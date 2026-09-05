"""Decide what preprocessing this dataset and this model actually need.

Preprocessing is not a fixed recipe. What is required depends on three things
at once: the dataset's problems, the semantic type of each column, and the
model that will consume the output. A random forest needs no scaling; a KNN
model is useless without it. This module reasons over all three.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, Decision, SemanticType, TaskType,
)
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.registry.base import ModelSpec

# Skew above this makes a transform worth suggesting for scale-sensitive models.
SKEW_THRESHOLD = 1.0
HIGH_CARDINALITY_ONEHOT_LIMIT = 15


def recommend_pipeline(
    profile: DatasetProfile,
    task_type: TaskType,
    target: str | None = None,
    model_spec: ModelSpec | None = None,
    context: BusinessContext | None = None,
    feature_columns: list[str] | None = None,
    aggressiveness: str = "standard",   # minimal | standard | thorough
) -> tuple[PreprocessingPipeline, list[Decision]]:
    """Build a preprocessing pipeline and the decision log explaining it."""
    context = context or BusinessContext()
    decisions: list[Decision] = []
    pipeline = PreprocessingPipeline(name=f"auto_{task_type.value}")

    features = _resolve_features(profile, target, context, feature_columns)
    dropped = _drop_useless(pipeline, profile, target, context, features, decisions)
    # Later stages must not reference columns that stage one has already removed.
    features = [c for c in features if c not in dropped]
    by_type = _group_by_semantic(profile, features)

    _handle_duplicates(pipeline, profile, decisions)
    _handle_missing(pipeline, profile, by_type, model_spec, decisions, aggressiveness)
    _engineer_features(pipeline, profile, by_type, decisions, aggressiveness)
    _handle_outliers(pipeline, profile, by_type, model_spec, decisions, aggressiveness)
    _transform_distributions(pipeline, profile, by_type, model_spec, decisions, aggressiveness)
    _encode_categoricals(pipeline, profile, by_type, task_type, model_spec, decisions)
    _scale(pipeline, profile, by_type, model_spec, decisions)
    _reduce_dimensions(pipeline, profile, by_type, model_spec, decisions, aggressiveness)

    return pipeline, decisions


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _resolve_features(
    profile: DatasetProfile,
    target: str | None,
    context: BusinessContext,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns:
        return [c for c in feature_columns if c in profile.columns]
    if context.include_variables:
        return [c for c in context.include_variables if c in profile.columns and c != target]
    excluded = set(context.exclude_variables)
    return [
        name for name, col in profile.columns.items()
        if name != target
        and name not in excluded
        and col.semantic_type not in {SemanticType.CONSTANT, SemanticType.UNKNOWN}
    ]


def _group_by_semantic(profile: DatasetProfile, features: list[str]) -> dict[SemanticType, list[str]]:
    grouped: dict[SemanticType, list[str]] = {}
    for name in features:
        col = profile.columns.get(name)
        if col is None:
            continue
        grouped.setdefault(col.semantic_type, []).append(name)
    return grouped


def _numeric(by_type: dict[SemanticType, list[str]]) -> list[str]:
    return by_type.get(SemanticType.NUMERIC_CONTINUOUS, []) + by_type.get(SemanticType.NUMERIC_DISCRETE, [])


def _log(decisions: list[Decision], stage: str, decision: str, reason: str,
         evidence: list[str] | None = None, rejected: list[dict[str, str]] | None = None,
         confidence: Confidence = Confidence.MODERATE) -> None:
    decisions.append(
        Decision(
            stage=stage, decision=decision, reason=reason,
            evidence=evidence or [], rejected=rejected or [], confidence=confidence,
        )
    )


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _drop_useless(pipeline, profile, target, context, features, decisions) -> set[str]:
    drop: list[str] = []
    reasons: list[str] = []

    for name in profile.identifier_columns:
        if name != target and name in profile.columns:
            drop.append(name)
            reasons.append(f"'{name}' is an identifier ({profile.columns[name].n_unique} distinct values)")
    for name in profile.constant_columns:
        if name != target:
            drop.append(name)
            reasons.append(f"'{name}' never varies")
    for name, col in profile.columns.items():
        if name != target and col.is_near_constant and name not in drop:
            drop.append(name)
            reasons.append(f"'{name}' is {col.dominant_pct:.1f}% one value")
    for suspect in profile.leakage_suspects:
        name = suspect["column"]
        if name != target and name not in drop and suspect.get("severity") == "critical":
            drop.append(name)
            reasons.append(f"'{name}' {suspect['reason']}")
    for name in context.exclude_variables:
        if name in profile.columns and name not in drop:
            drop.append(name)
            reasons.append(f"'{name}' was excluded by you")

    drop = [c for c in dict.fromkeys(drop) if c != target]
    if drop:
        pipeline.add("drop_columns", columns=drop, by="ai")
        _log(
            decisions, "preprocessing", f"Drop {len(drop)} column(s) before modelling",
            "These columns cannot generalise, or would leak the answer.",
            evidence=reasons[:12], confidence=Confidence.HIGH,
        )
    return set(drop)


def _handle_duplicates(pipeline, profile, decisions) -> None:
    if profile.n_duplicate_rows > 0:
        pipeline.add("drop_duplicates", by="ai")
        _log(
            decisions, "preprocessing", "Remove duplicate rows",
            "Duplicate rows inflate apparent sample size and can appear in both training and test data.",
            evidence=[f"{profile.n_duplicate_rows} duplicate rows ({profile.duplicate_pct:.2f}% of the dataset)"],
            confidence=Confidence.HIGH,
        )


def _handle_missing(pipeline, profile, by_type, model_spec, decisions, aggressiveness) -> None:
    numeric = _numeric(by_type)
    categorical = [
        c for t in (SemanticType.CATEGORICAL_NOMINAL, SemanticType.CATEGORICAL_ORDINAL,
                    SemanticType.BINARY, SemanticType.HIGH_CARDINALITY_CATEGORICAL)
        for c in by_type.get(t, [])
    ]
    numeric_missing = [c for c in numeric if profile.columns[c].missing_pct > 0]
    categorical_missing = [c for c in categorical if profile.columns[c].missing_pct > 0]

    if not numeric_missing and not categorical_missing:
        return

    if model_spec is not None and model_spec.handles_missing and aggressiveness != "thorough":
        _log(
            decisions, "preprocessing", "Leave missing values for the model to handle",
            f"{model_spec.name} handles missing values natively, and its own split-based treatment "
            "usually beats imputing a value that was never observed.",
            evidence=[f"{len(numeric_missing) + len(categorical_missing)} column(s) contain gaps"],
            confidence=Confidence.MODERATE,
        )
        return

    if numeric_missing:
        worst = max(profile.columns[c].missing_pct for c in numeric_missing)
        skewed = [c for c in numeric_missing
                  if abs(profile.columns[c].skewness or 0) > SKEW_THRESHOLD]
        correlated = _has_correlated_numerics(profile, numeric_missing)

        if worst > 20 and correlated and len(numeric) <= 40 and profile.n_rows <= 50_000:
            pipeline.add("impute_knn", columns=numeric, n_neighbors=5, by="ai")
            choice, reason = (
                "KNN imputation for numeric columns",
                "Missingness is substantial and the numeric columns are correlated, so similar "
                "rows carry real information about the missing value.",
            )
            rejected = [
                {"option": "Median imputation", "reason": "Would ignore the relationships between columns"},
                {"option": "Dropping the rows", "reason": f"Would discard up to {worst:.0f}% of the data"},
            ]
        elif skewed:
            pipeline.add("impute_median", columns=numeric, by="ai")
            choice, reason = (
                "Median imputation for numeric columns",
                "Several of these columns are skewed, and the median is not dragged by the tail.",
            )
            rejected = [{"option": "Mean imputation", "reason": "Skew and outliers would pull the mean away from typical values"}]
        else:
            pipeline.add("impute_median", columns=numeric, by="ai")
            choice, reason = (
                "Median imputation for numeric columns",
                "A safe default: it never invents an out-of-range value and is unaffected by outliers.",
            )
            rejected = [{"option": "Mean imputation", "reason": "Marginally different here; the median is the safer default"}]

        _log(
            decisions, "preprocessing", choice, reason,
            evidence=[f"'{c}': {profile.columns[c].missing_pct:.1f}% missing" for c in numeric_missing[:8]],
            rejected=rejected,
        )

    if categorical_missing:
        heavy = [c for c in categorical_missing if profile.columns[c].missing_pct > 10]
        if heavy:
            pipeline.add("impute_constant", columns=categorical, fill_value="__missing__", by="ai")
            choice = "Treat missing categories as their own level"
            reason = (
                "More than 10% of values are missing in at least one column. At that level the "
                "absence is usually informative, so it is modelled rather than papered over."
            )
        else:
            pipeline.add("impute_mode", columns=categorical, by="ai")
            choice = "Impute categorical gaps with the most frequent level"
            reason = "Missingness is light, so filling with the dominant category barely shifts the distribution."
        _log(
            decisions, "preprocessing", choice, reason,
            evidence=[f"'{c}': {profile.columns[c].missing_pct:.1f}% missing" for c in categorical_missing[:8]],
        )


def _has_correlated_numerics(profile: DatasetProfile, columns: list[str]) -> bool:
    wanted = set(columns)
    return any(abs(r) > 0.4 and (a in wanted or b in wanted) for a, b, r in profile.correlation_pairs[:40])


def _engineer_features(pipeline, profile, by_type, decisions, aggressiveness) -> None:
    dates = by_type.get(SemanticType.DATETIME, [])
    if dates:
        pipeline.add("datetime_features", columns=dates, cyclical=True, by="ai")
        pipeline.add("drop_columns", columns=dates, by="ai")
        _log(
            decisions, "preprocessing", f"Expand {len(dates)} date column(s) into calendar features",
            "A raw timestamp means nothing to most models. Year, month, weekday and quarter do, "
            "and cyclical sin/cos terms stop December and January looking eleven months apart.",
            evidence=[f"'{c}' spans {profile.columns[c].min_date} to {profile.columns[c].max_date}" for c in dates],
            confidence=Confidence.HIGH,
        )

    texts = by_type.get(SemanticType.TEXT, [])
    if texts:
        pipeline.add("text_features", columns=texts, by="ai")
        pipeline.add("drop_columns", columns=texts, by="ai")
        _log(
            decisions, "preprocessing", f"Summarise {len(texts)} text column(s) numerically",
            "Length, word count and vocabulary richness are cheap, interpretable signals.",
            evidence=[f"'{c}' averages {profile.columns[c].mean_token_count:.1f} words" for c in texts
                      if profile.columns[c].mean_token_count],
            rejected=[{"option": "TF-IDF vectorisation",
                       "reason": "Adds thousands of columns and makes the model far harder to explain; "
                                 "add it deliberately if the wording itself matters"}],
            confidence=Confidence.MODERATE,
        )


def _handle_outliers(pipeline, profile, by_type, model_spec, decisions, aggressiveness) -> None:
    numeric = _numeric(by_type)
    outliery = [c for c in numeric if profile.columns[c].outlier_pct >= 5.0]
    if not outliery:
        return
    if model_spec is not None and (model_spec.robust_to_outliers or model_spec.family in {"tree", "ensemble"}):
        _log(
            decisions, "preprocessing", "Leave outliers untouched",
            f"{model_spec.name} splits on rank order rather than magnitude, so extreme values do not distort it.",
            evidence=[f"'{c}': {profile.columns[c].outlier_pct:.1f}% outside the IQR fences" for c in outliery[:6]],
        )
        return
    if aggressiveness == "minimal":
        return
    pipeline.add("winsorize", columns=outliery, lower_quantile=0.01, upper_quantile=0.99, by="ai")
    _log(
        decisions, "preprocessing", f"Winsorise {len(outliery)} numeric column(s) at the 1st/99th percentile",
        "The chosen model is sensitive to magnitude, so a handful of extremes would dominate the fit. "
        "Clipping keeps every row while limiting their pull.",
        evidence=[f"'{c}': {profile.columns[c].n_outliers_iqr} outliers ({profile.columns[c].outlier_pct:.1f}%)"
                  for c in outliery[:6]],
        rejected=[{"option": "Deleting outlier rows",
                   "reason": "Extremes are often the most commercially interesting cases; deleting them hides real behaviour"}],
    )


def _transform_distributions(pipeline, profile, by_type, model_spec, decisions, aggressiveness) -> None:
    if aggressiveness == "minimal":
        return
    numeric = _numeric(by_type)
    skewed = [c for c in numeric if abs(profile.columns[c].skewness or 0) > SKEW_THRESHOLD]
    if not skewed:
        return
    scale_sensitive = model_spec is None or model_spec.requires_scaling or model_spec.family in {"linear", "statistical", "kernel", "distance", "neural"}
    if not scale_sensitive:
        _log(
            decisions, "preprocessing", "Skip distribution transforms",
            f"{model_spec.name} is invariant to monotonic transformations, so reshaping the "
            "distribution would change nothing except interpretability.",
            evidence=[f"'{c}' skew {profile.columns[c].skewness:.2f}" for c in skewed[:5]],
        )
        return
    all_positive = [c for c in skewed if (profile.columns[c].minimum or 0) > 0]
    if len(all_positive) == len(skewed):
        pipeline.add("log_transform", columns=skewed, by="ai")
        choice, reason = (
            f"Log-transform {len(skewed)} skewed column(s)",
            "All are strictly positive and right-skewed. A log makes the relationship closer to "
            "linear and stops large values dominating.",
        )
        rejected = [{"option": "Yeo-Johnson", "reason": "Unnecessary here; a log is simpler to explain to stakeholders"}]
    else:
        pipeline.add("yeo_johnson", columns=skewed, standardize=False, by="ai")
        choice, reason = (
            f"Yeo-Johnson transform {len(skewed)} skewed column(s)",
            "Some contain zero or negative values, where a log or Box-Cox would fail.",
        )
        rejected = [{"option": "Log transform", "reason": "Undefined for the non-positive values present"}]
    _log(
        decisions, "preprocessing", choice, reason,
        evidence=[f"'{c}' skew {profile.columns[c].skewness:.2f}" for c in skewed[:8]],
        rejected=rejected,
    )


def _encode_categoricals(pipeline, profile, by_type, task_type, model_spec, decisions) -> None:
    ordinal = by_type.get(SemanticType.CATEGORICAL_ORDINAL, [])
    nominal = by_type.get(SemanticType.CATEGORICAL_NOMINAL, [])
    binary = by_type.get(SemanticType.BINARY, [])
    high_card = by_type.get(SemanticType.HIGH_CARDINALITY_CATEGORICAL, [])

    if ordinal:
        pipeline.add("ordinal_encode", columns=ordinal, by="ai")
        _log(
            decisions, "preprocessing", f"Ordinal-encode {len(ordinal)} ordered column(s)",
            "These levels have a genuine order, so integer codes preserve real information "
            "that one-hot encoding would throw away.",
            evidence=[f"'{c}': {list(profile.columns[c].top_values)[:5]}" for c in ordinal[:5]],
        )

    binary_like = [c for c in binary if not pd.api.types.is_numeric_dtype(pd.Series(dtype=profile.columns[c].dtype))]
    onehot_targets = [c for c in nominal if profile.columns[c].n_unique <= HIGH_CARDINALITY_ONEHOT_LIMIT] + binary_like
    wide_nominal = [c for c in nominal if profile.columns[c].n_unique > HIGH_CARDINALITY_ONEHOT_LIMIT]

    if onehot_targets:
        drop = "first" if (model_spec and model_spec.family in {"linear", "statistical"}) else None
        pipeline.add("one_hot_encode", columns=onehot_targets, drop=drop, by="ai")
        reason = "Each level becomes its own indicator, which imposes no false ordering."
        if drop == "first":
            reason += (
                " One level is dropped as the reference category, because a linear model with "
                "all levels present is perfectly collinear (the dummy-variable trap)."
            )
        _log(
            decisions, "preprocessing", f"One-hot encode {len(onehot_targets)} categorical column(s)", reason,
            evidence=[f"'{c}': {profile.columns[c].n_unique} levels" for c in onehot_targets[:8]],
        )

    encode_by_target = high_card + wide_nominal
    if encode_by_target:
        if task_type.is_supervised and task_type is not TaskType.TIME_SERIES_FORECAST:
            pipeline.add("target_encode", columns=encode_by_target, smoothing=10.0, by="ai")
            choice = f"Target-encode {len(encode_by_target)} high-cardinality column(s)"
            reason = (
                "One-hot encoding these would add hundreds of near-empty columns. Target encoding "
                "replaces each level with its smoothed average outcome, and it is fitted inside each "
                "cross-validation fold so it cannot leak."
            )
            rejected = [
                {"option": "One-hot encoding", "reason": f"Would create roughly "
                 f"{sum(profile.columns[c].n_unique for c in encode_by_target)} columns"},
            ]
        else:
            pipeline.add("frequency_encode", columns=encode_by_target, by="ai")
            choice = f"Frequency-encode {len(encode_by_target)} high-cardinality column(s)"
            reason = "There is no target to encode against, so each level is represented by how common it is."
            rejected = [{"option": "One-hot encoding", "reason": "Too many levels to expand"}]
        _log(
            decisions, "preprocessing", choice, reason,
            evidence=[f"'{c}': {profile.columns[c].n_unique} distinct levels" for c in encode_by_target[:8]],
            rejected=rejected,
        )


def _scale(pipeline, profile, by_type, model_spec, decisions) -> None:
    if model_spec is not None and not model_spec.requires_scaling:
        _log(
            decisions, "preprocessing", "Skip scaling",
            f"{model_spec.name} is scale-invariant, so standardising would only make the "
            "coefficients harder to read against the original units.",
        )
        return
    numeric = _numeric(by_type)
    if not numeric:
        return
    outliery = [c for c in numeric if profile.columns[c].outlier_pct >= 5.0]
    if len(outliery) >= max(1, len(numeric) // 3):
        pipeline.add("robust_scale", columns="__numeric__", by="ai")
        choice, reason = (
            "Robust scaling (median and IQR)",
            "A third or more of the numeric columns carry heavy tails; using the median and IQR "
            "means the outliers do not set the scale for everything else.",
        )
        rejected = [{"option": "Standard z-score scaling", "reason": "The mean and standard deviation are themselves distorted by the outliers"}]
    else:
        pipeline.add("standard_scale", columns="__numeric__", by="ai")
        choice, reason = (
            "Standardise all numeric features (z-score)",
            f"{model_spec.name if model_spec else 'The selected model family'} compares features by "
            "magnitude, so columns on larger scales would otherwise dominate purely because of their units.",
        )
        rejected = [{"option": "Min-max scaling", "reason": "More sensitive to outliers and to values outside the training range"}]
    _log(decisions, "preprocessing", choice, reason,
         evidence=[f"{len(numeric)} numeric feature(s) to scale"], rejected=rejected,
         confidence=Confidence.HIGH)


def _reduce_dimensions(pipeline, profile, by_type, model_spec, decisions, aggressiveness) -> None:
    numeric = _numeric(by_type)
    severe_vif = {k: v for k, v in profile.multicollinearity.items() if v >= 10 and k in numeric}
    wide = profile.n_columns > max(30, profile.n_rows / 10)

    if severe_vif and len(severe_vif) >= 3 and aggressiveness == "thorough":
        pipeline.add("pca_reduce", columns="__numeric__", n_components=0.95, by="ai")
        _log(
            decisions, "preprocessing", "Reduce numeric features with PCA (95% variance retained)",
            "Several predictors are so collinear that individual coefficients would be unreliable. "
            "PCA replaces them with uncorrelated components.",
            evidence=[f"VIF {name} = {value:.1f}" for name, value in list(severe_vif.items())[:6]],
            rejected=[{"option": "Dropping correlated columns",
                       "reason": "Which column of a correlated pair to keep is an arbitrary choice"},
                      {"option": "Ridge regularisation",
                       "reason": "A good alternative — it keeps the original variables; choose it if "
                                 "you need to interpret individual predictors"}],
            confidence=Confidence.MODERATE,
        )
    elif severe_vif:
        _log(
            decisions, "preprocessing", "Flag multicollinearity but keep the original variables",
            "Interpretable coefficients are usually worth more than the small accuracy gain from "
            "reducing dimensions. Prefer a regularised model (Ridge / Elastic Net) instead.",
            evidence=[f"VIF {name} = {value:.1f}" for name, value in list(severe_vif.items())[:6]],
            confidence=Confidence.MODERATE,
        )
    elif wide and aggressiveness == "thorough":
        _log(
            decisions, "preprocessing", "Consider feature selection",
            f"{profile.n_columns} columns against {profile.n_rows} rows is wide enough that some "
            "features will look predictive by chance alone.",
            evidence=[f"{profile.n_columns} columns / {profile.n_rows} rows"],
            confidence=Confidence.MODERATE,
        )
