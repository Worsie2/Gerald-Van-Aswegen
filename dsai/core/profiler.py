"""The Data Understanding Engine.

Takes a raw DataFrame and produces a :class:`DatasetProfile` — the structured
internal representation every other engine reasons over. It answers "what is
this data?" before anybody is asked "what model do you want?".

Everything here is measured, never assumed. User beliefs live in
:class:`~dsai.core.schema.BusinessContext` and are applied afterwards by
:func:`apply_user_overrides`.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import (
    BusinessContext,
    ColumnProfile,
    DatasetProfile,
    QualityIssue,
    RelationshipFinding,
    SemanticType,
)

# Tunables collected in one place so they can be adjusted per project.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "identifier_unique_ratio": 0.95,
    "high_cardinality_abs": 50,
    "high_cardinality_ratio": 0.5,
    "near_constant_dominance": 0.99,
    "discrete_max_unique": 20,
    "text_min_mean_tokens": 3.0,
    "text_min_mean_chars": 25.0,
    "missing_warning_pct": 5.0,
    "missing_critical_pct": 40.0,
    "outlier_warning_pct": 5.0,
    "skew_warning": 1.0,
    "high_correlation": 0.85,
    "leakage_correlation": 0.98,
    "vif_warning": 5.0,
    "vif_critical": 10.0,
    "imbalance_warning": 0.20,     # minority share below this = imbalanced
    "imbalance_critical": 0.05,
    "small_sample_rows": 100,
    "wide_data_ratio": 0.5,        # columns / rows
    "max_profile_rows": 200_000,   # sample above this for speed
    "max_corr_columns": 60,
}

_ID_NAME_PATTERN = re.compile(
    r"^(id|uuid|guid|key|code|ref|reference|number|no|nr|index|idx)$"
    r"|(_id|_uuid|_guid|_key|_code|_ref|_no|_nr|_number)$"
    r"|^(id_|uuid_|key_)",
    re.IGNORECASE,
)
_DATE_NAME_PATTERN = re.compile(
    r"(date|time|timestamp|datetime|day|month|year|quarter|week|period|dt)$"
    r"|^(date|time|timestamp|datetime|dt)",
    re.IGNORECASE,
)
_ORDINAL_TOKENS = [
    {"low", "medium", "high"},
    {"low", "med", "high"},
    {"small", "medium", "large"},
    {"poor", "fair", "good", "excellent"},
    {"bronze", "silver", "gold", "platinum"},
    {"never", "rarely", "sometimes", "often", "always"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"very low", "low", "medium", "high", "very high"},
    {"beginner", "intermediate", "advanced"},
]
_TARGETY_NAME_PATTERN = re.compile(
    r"(target|label|outcome|response|churn|converted|conversion|purchase|purchased|"
    r"buy|bought|revenue|sales|spend|spending|price|amount|value|score|rating|"
    r"default|fraud|risk|success|survived|status|class|result|profit|margin|"
    r"quantity|demand|yield|attrition|renewal|subscribe|click|approved)",
    re.IGNORECASE,
)
_LEAKY_NAME_PATTERN = re.compile(
    r"(predicted|prediction|forecast|score_|_score$|probability|propensity|"
    r"post_|after_|final_|actual_|_flag_computed|model_)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# type inference
# --------------------------------------------------------------------------

def _try_parse_datetime(series: pd.Series) -> pd.Series | None:
    """Attempt datetime parsing on an object column without pandas noise."""
    sample = series.dropna()
    if sample.empty:
        return None
    sample = sample.head(500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
            except Exception:
                return None
    if parsed.notna().mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except Exception:
            return pd.to_datetime(series, errors="coerce")


def _looks_ordinal(values: set[str]) -> bool:
    lowered = {str(v).strip().lower() for v in values}
    for pattern in _ORDINAL_TOKENS:
        if lowered and lowered.issubset(pattern) and len(lowered) >= 2:
            return True
    # graded labels such as "1 - Poor", "Grade A"
    if all(re.match(r"^(grade|level|tier|band|stage)\s*[a-z0-9]", v) for v in lowered):
        return len(lowered) >= 2
    return False


def infer_semantic_type(
    series: pd.Series,
    name: str,
    n_rows: int,
    thresholds: dict[str, float],
) -> tuple[SemanticType, pd.Series]:
    """Return the semantic type plus a (possibly converted) working series."""
    working = series
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    if n_rows == 0 or non_null.empty:
        return SemanticType.CONSTANT, working
    if n_unique <= 1:
        return SemanticType.CONSTANT, working

    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME, working
    if isinstance(series.dtype, pd.PeriodDtype):
        return SemanticType.DATETIME, working

    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BINARY, working

    if pd.api.types.is_numeric_dtype(series):
        if n_unique == 2:
            return SemanticType.BINARY, working
        unique_ratio = n_unique / max(len(non_null), 1)
        is_intlike = bool(
            pd.api.types.is_integer_dtype(series)
            or np.allclose(non_null.astype(float) % 1, 0, atol=1e-9)
        )
        if is_intlike and unique_ratio >= thresholds["identifier_unique_ratio"] and n_rows > 20:
            if _ID_NAME_PATTERN.search(name) or _is_monotonic_sequence(non_null):
                return SemanticType.IDENTIFIER, working
        if is_intlike and n_unique <= thresholds["discrete_max_unique"]:
            return SemanticType.NUMERIC_DISCRETE, working
        return SemanticType.NUMERIC_CONTINUOUS, working

    # object / string / categorical
    as_str = non_null.astype(str)
    if _DATE_NAME_PATTERN.search(name) or _has_date_shape(as_str):
        parsed = _try_parse_datetime(series)
        if parsed is not None:
            return SemanticType.DATETIME, parsed

    numeric_try = pd.to_numeric(
        as_str.str.replace(r"[,\s%]", "", regex=True).str.replace(r"^[R$€£]", "", regex=True),
        errors="coerce",
    )
    if numeric_try.notna().mean() > 0.95:
        converted = pd.to_numeric(
            series.astype(str)
            .str.replace(r"[,\s%]", "", regex=True)
            .str.replace(r"^[R$€£]", "", regex=True),
            errors="coerce",
        )
        return infer_semantic_type(converted, name, n_rows, thresholds)[0], converted

    unique_ratio = n_unique / max(len(non_null), 1)
    mean_chars = float(as_str.str.len().mean())
    mean_tokens = float(as_str.str.split().str.len().mean())

    if n_unique == 2:
        return SemanticType.BINARY, working
    if unique_ratio >= thresholds["identifier_unique_ratio"] and n_rows > 20:
        if mean_tokens < thresholds["text_min_mean_tokens"]:
            return SemanticType.IDENTIFIER, working
    if (
        mean_tokens >= thresholds["text_min_mean_tokens"]
        and mean_chars >= thresholds["text_min_mean_chars"]
    ):
        return SemanticType.TEXT, working
    if n_unique <= 15 and _looks_ordinal(set(as_str.unique()[:30])):
        return SemanticType.CATEGORICAL_ORDINAL, working
    if (
        n_unique > thresholds["high_cardinality_abs"]
        or unique_ratio > thresholds["high_cardinality_ratio"]
    ):
        return SemanticType.HIGH_CARDINALITY_CATEGORICAL, working
    return SemanticType.CATEGORICAL_NOMINAL, working


def _is_monotonic_sequence(series: pd.Series) -> bool:
    try:
        arr = series.astype(float).to_numpy()
    except Exception:
        return False
    if arr.size < 5:
        return False
    diffs = np.diff(np.sort(arr))
    return bool(np.all(diffs > 0) and np.median(diffs) <= 2)


def _has_date_shape(as_str: pd.Series) -> bool:
    sample = as_str.head(50)
    pattern = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
    return bool(sample.str.match(pattern).mean() > 0.8)


# --------------------------------------------------------------------------
# per-column profiling
# --------------------------------------------------------------------------

def profile_column(
    series: pd.Series,
    name: str,
    n_rows: int,
    thresholds: dict[str, float],
) -> tuple[ColumnProfile, pd.Series]:
    semantic, working = infer_semantic_type(series, name, n_rows, thresholds)
    non_null = working.dropna()
    n_missing = int(working.isna().sum())
    n_unique = int(non_null.nunique()) if not non_null.empty else 0

    prof = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        semantic_type=semantic,
        n_rows=n_rows,
        n_missing=n_missing,
        missing_pct=round(100.0 * n_missing / n_rows, 3) if n_rows else 0.0,
        n_unique=n_unique,
        unique_pct=round(100.0 * n_unique / n_rows, 3) if n_rows else 0.0,
        cardinality_ratio=round(n_unique / max(len(non_null), 1), 4),
    )
    prof.is_constant = n_unique <= 1

    if not non_null.empty:
        counts = non_null.value_counts()
        prof.dominant_value = counts.index[0]
        prof.dominant_pct = round(100.0 * float(counts.iloc[0]) / len(non_null), 3)
        prof.is_near_constant = (
            prof.dominant_pct >= thresholds["near_constant_dominance"] * 100
            and not prof.is_constant
        )
        if semantic in {
            SemanticType.BINARY,
            SemanticType.CATEGORICAL_NOMINAL,
            SemanticType.CATEGORICAL_ORDINAL,
            SemanticType.HIGH_CARDINALITY_CATEGORICAL,
        }:
            prof.top_values = {str(k): int(v) for k, v in counts.head(10).items()}

    if semantic in {SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE} and not non_null.empty:
        values = pd.to_numeric(non_null, errors="coerce").dropna().astype(float)
        if not values.empty:
            q1, med, q3 = (float(values.quantile(q)) for q in (0.25, 0.5, 0.75))
            prof.mean = float(values.mean())
            prof.std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            prof.minimum, prof.q1, prof.median = float(values.min()), q1, med
            prof.q3, prof.maximum = q3, float(values.max())
            prof.skewness = float(values.skew()) if len(values) > 2 else 0.0
            prof.kurtosis = float(values.kurt()) if len(values) > 3 else 0.0
            prof.zeros_pct = round(100.0 * float((values == 0).mean()), 3)
            prof.negative_pct = round(100.0 * float((values < 0).mean()), 3)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = int(((values < lo) | (values > hi)).sum())
                prof.n_outliers_iqr = n_out
                prof.outlier_pct = round(100.0 * n_out / len(values), 3)

    if semantic is SemanticType.DATETIME and not non_null.empty:
        dt = pd.to_datetime(non_null, errors="coerce").dropna()
        if not dt.empty:
            prof.min_date = str(dt.min())
            prof.max_date = str(dt.max())
            freq, missing_periods = _infer_frequency(dt)
            prof.inferred_frequency = freq
            prof.n_missing_periods = missing_periods

    if semantic is SemanticType.TEXT and not non_null.empty:
        prof.mean_token_count = float(non_null.astype(str).str.split().str.len().mean())

    return prof, working


def _infer_frequency(dt: pd.Series) -> tuple[str | None, int | None]:
    """Best-effort frequency inference plus a count of gaps in the sequence."""
    unique_sorted = pd.Series(pd.to_datetime(dt.unique())).sort_values()
    if len(unique_sorted) < 3:
        return None, None
    try:
        freq = pd.infer_freq(pd.DatetimeIndex(unique_sorted))
    except (ValueError, TypeError):
        freq = None
    deltas = unique_sorted.diff().dropna()
    if deltas.empty:
        return freq, None
    median_delta = deltas.median()
    if freq is None:
        days = median_delta / pd.Timedelta(days=1)
        freq = (
            "hourly" if days < 0.9 else
            "daily" if days < 2 else
            "weekly" if days < 9 else
            "monthly" if days < 45 else
            "quarterly" if days < 135 else
            "yearly" if days < 500 else "irregular"
        )
    if median_delta > pd.Timedelta(0):
        span = unique_sorted.iloc[-1] - unique_sorted.iloc[0]
        expected = int(span / median_delta) + 1
        gaps = max(0, expected - len(unique_sorted))
    else:
        gaps = 0
    return freq, gaps


# --------------------------------------------------------------------------
# dataset-level analysis
# --------------------------------------------------------------------------

def _correlation_analysis(
    frame: pd.DataFrame,
    numeric_cols: list[str],
    thresholds: dict[str, float],
) -> tuple[list[tuple[str, str, float]], list[RelationshipFinding]]:
    pairs: list[tuple[str, str, float]] = []
    findings: list[RelationshipFinding] = []
    cols = numeric_cols[: int(thresholds["max_corr_columns"])]
    if len(cols) < 2:
        return pairs, findings
    corr = frame[cols].corr(method="pearson", numeric_only=True)
    spearman = frame[cols].corr(method="spearman", numeric_only=True)
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            pairs.append((a, b, round(float(r), 4)))
            key = tuple(sorted((a, b)))
            if abs(r) >= thresholds["high_correlation"] and key not in seen:
                seen.add(key)
                findings.append(
                    RelationshipFinding(
                        kind="correlation",
                        columns=[a, b],
                        statistic=round(float(r), 4),
                        method="pearson",
                        strength="very strong" if abs(r) >= 0.95 else "strong",
                        message=f"'{a}' and '{b}' move together (r = {r:.2f}).",
                    )
                )
            rs = spearman.loc[a, b]
            # A big rank-correlation gap over Pearson is a nonlinearity signal.
            if not pd.isna(rs) and abs(rs) - abs(r) > 0.15 and abs(rs) > 0.4:
                findings.append(
                    RelationshipFinding(
                        kind="nonlinearity",
                        columns=[a, b],
                        statistic=round(float(abs(rs) - abs(r)), 4),
                        method="spearman_vs_pearson",
                        strength="moderate",
                        message=(
                            f"'{a}' vs '{b}' looks monotonic but not straight-line "
                            f"(Spearman {rs:.2f} vs Pearson {r:.2f})."
                        ),
                    )
                )
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:200], findings


def _vif_scores(frame: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    """Variance inflation factors via R^2 of each column on the others."""
    cols = [c for c in numeric_cols if frame[c].notna().sum() > 10]
    if len(cols) < 3:
        return {}
    data = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(cols) + 5:
        return {}
    data = data.loc[:, data.std(ddof=0) > 0]
    if data.shape[1] < 3:
        return {}
    scores: dict[str, float] = {}
    matrix = data.to_numpy(dtype=float)
    matrix = (matrix - matrix.mean(0)) / matrix.std(0)
    for idx, col in enumerate(data.columns):
        y = matrix[:, idx]
        x = np.delete(matrix, idx, axis=1)
        x = np.column_stack([np.ones(len(x)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ beta
            ss_res = float(resid @ resid)
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            scores[col] = round(float(1.0 / max(1e-9, 1.0 - r2)), 2) if r2 < 1 else 999.0
        except np.linalg.LinAlgError:
            continue
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def _target_candidates(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Score every column on how plausible it is as a dependent variable."""
    candidates: list[dict[str, Any]] = []
    n_rows = profile.n_rows
    for name, col in profile.columns.items():
        if col.semantic_type in {SemanticType.IDENTIFIER, SemanticType.CONSTANT,
                                 SemanticType.TEXT, SemanticType.DATETIME}:
            continue
        score, reasons = 0.0, []
        if _TARGETY_NAME_PATTERN.search(name):
            score += 0.35
            reasons.append("name resembles an outcome variable")
        if col.semantic_type is SemanticType.BINARY:
            score += 0.25
            reasons.append("binary outcome")
        elif col.semantic_type is SemanticType.NUMERIC_CONTINUOUS:
            score += 0.20
            reasons.append("continuous numeric")
        elif col.semantic_type is SemanticType.CATEGORICAL_NOMINAL and col.n_unique <= 10:
            score += 0.15
            reasons.append("low-cardinality categorical")
        if col.missing_pct == 0:
            score += 0.10
            reasons.append("fully populated")
        elif col.missing_pct > 30:
            score -= 0.20
            reasons.append(f"{col.missing_pct:.0f}% missing")
        # last column is conventionally the label in tabular exports
        if list(profile.columns).index(name) == len(profile.columns) - 1:
            score += 0.08
            reasons.append("last column in the file")
        if col.is_near_constant:
            score -= 0.4
            reasons.append("almost no variation")
        if score <= 0:
            continue
        task = _implied_task(col)
        candidates.append(
            {
                "column": name,
                "score": round(min(1.0, score), 3),
                "implied_task": task,
                "reasons": reasons,
                "semantic_type": col.semantic_type.value,
                "n_unique": col.n_unique,
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:10]


def _implied_task(col: ColumnProfile) -> str:
    if col.semantic_type is SemanticType.BINARY:
        return "binary_classification"
    if col.semantic_type in {SemanticType.CATEGORICAL_NOMINAL, SemanticType.CATEGORICAL_ORDINAL}:
        return "multiclass_classification"
    if col.semantic_type is SemanticType.NUMERIC_DISCRETE and col.n_unique <= 10:
        return "multiclass_classification"
    return "regression"


def _leakage_suspects(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    target: str | None,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Flag columns that would not exist at prediction time, or are the target in disguise."""
    suspects: list[dict[str, Any]] = []
    for name, col in profile.columns.items():
        if _LEAKY_NAME_PATTERN.search(name):
            suspects.append(
                {
                    "column": name,
                    "reason": "name suggests a post-outcome or model-generated field",
                    "severity": "warning",
                }
            )
    if target and target in frame.columns:
        tgt = pd.to_numeric(frame[target], errors="coerce")
        if tgt.notna().sum() > 10:
            for name in profile.numeric_columns:
                if name == target:
                    continue
                other = pd.to_numeric(frame[name], errors="coerce")
                joined = pd.concat([tgt, other], axis=1).dropna()
                if len(joined) < 10:
                    continue
                r = joined.corr().iloc[0, 1]
                if pd.notna(r) and abs(r) >= thresholds["leakage_correlation"]:
                    suspects.append(
                        {
                            "column": name,
                            "reason": (
                                f"correlates {r:.3f} with the target '{target}' — "
                                "likely a restatement of the outcome"
                            ),
                            "severity": "critical",
                        }
                    )
    return suspects


def _detect_quality_issues(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    n_rows = profile.n_rows

    if n_rows < thresholds["small_sample_rows"]:
        issues.append(
            QualityIssue(
                code="small_sample",
                severity="critical" if n_rows < 30 else "warning",
                columns=[],
                message=(
                    f"Only {n_rows} rows. Model estimates will be unstable and "
                    "cross-validated results will vary a lot between runs."
                ),
                detail={"n_rows": n_rows},
                suggested_action="Prefer simple models, repeated cross-validation, and wide uncertainty language.",
            )
        )
    if profile.n_columns > 0 and n_rows > 0:
        ratio = profile.n_columns / n_rows
        if ratio > thresholds["wide_data_ratio"]:
            issues.append(
                QualityIssue(
                    code="wide_data",
                    severity="warning",
                    columns=[],
                    message=(
                        f"{profile.n_columns} columns for {n_rows} rows "
                        f"(ratio {ratio:.2f}). High risk of overfitting."
                    ),
                    detail={"ratio": round(ratio, 3)},
                    suggested_action="Use regularisation, feature selection or dimensionality reduction.",
                )
            )

    if profile.n_duplicate_rows:
        issues.append(
            QualityIssue(
                code="duplicate_rows",
                severity="warning" if profile.duplicate_pct < 5 else "critical",
                columns=[],
                message=f"{profile.n_duplicate_rows} duplicate rows ({profile.duplicate_pct:.2f}%).",
                detail={"count": profile.n_duplicate_rows},
                suggested_action="Remove duplicates unless repeated rows are genuine observations.",
            )
        )

    heavy_missing = [n for n, c in profile.columns.items() if c.missing_pct >= thresholds["missing_critical_pct"]]
    some_missing = [
        n for n, c in profile.columns.items()
        if thresholds["missing_warning_pct"] <= c.missing_pct < thresholds["missing_critical_pct"]
    ]
    if heavy_missing:
        issues.append(
            QualityIssue(
                code="severe_missingness",
                severity="critical",
                columns=heavy_missing,
                message=f"{len(heavy_missing)} column(s) are at least {thresholds['missing_critical_pct']:.0f}% missing.",
                detail={c: profile.columns[c].missing_pct for c in heavy_missing},
                suggested_action="Consider dropping these columns, or model missingness explicitly.",
            )
        )
    if some_missing:
        issues.append(
            QualityIssue(
                code="missing_values",
                severity="warning",
                columns=some_missing,
                message=f"{len(some_missing)} column(s) have moderate missingness.",
                detail={c: profile.columns[c].missing_pct for c in some_missing},
                suggested_action="Impute; median for skewed numerics, KNN/iterative when missingness is patterned.",
            )
        )

    if profile.constant_columns:
        issues.append(
            QualityIssue(
                code="constant_columns",
                severity="info",
                columns=profile.constant_columns,
                message=f"{len(profile.constant_columns)} column(s) carry no information.",
                detail={},
                suggested_action="Drop them; they cannot contribute to any model.",
            )
        )

    near_const = [n for n, c in profile.columns.items() if c.is_near_constant]
    if near_const:
        issues.append(
            QualityIssue(
                code="near_constant_columns",
                severity="info",
                columns=near_const,
                message=f"{len(near_const)} column(s) are ≥99% a single value.",
                detail={c: profile.columns[c].dominant_pct for c in near_const},
                suggested_action="Usually safe to drop unless the rare value is the point of the analysis.",
            )
        )

    outliery = [
        n for n, c in profile.columns.items()
        if c.outlier_pct >= thresholds["outlier_warning_pct"]
    ]
    if outliery:
        issues.append(
            QualityIssue(
                code="outliers",
                severity="warning",
                columns=outliery,
                message=f"{len(outliery)} numeric column(s) have >{thresholds['outlier_warning_pct']:.0f}% IQR outliers.",
                detail={c: profile.columns[c].outlier_pct for c in outliery},
                suggested_action="Inspect before removing — outliers are often the most interesting rows.",
            )
        )

    skewed = [
        n for n, c in profile.columns.items()
        if c.skewness is not None and abs(c.skewness) > thresholds["skew_warning"]
    ]
    if skewed:
        issues.append(
            QualityIssue(
                code="skewed_distributions",
                severity="info",
                columns=skewed,
                message=f"{len(skewed)} numeric column(s) are noticeably skewed.",
                detail={c: round(profile.columns[c].skewness or 0, 2) for c in skewed},
                suggested_action="Log / Yeo-Johnson transforms help linear models; tree models do not care.",
            )
        )

    strong_pairs = [(a, b, r) for a, b, r in profile.correlation_pairs if abs(r) >= thresholds["high_correlation"]]
    if strong_pairs:
        issues.append(
            QualityIssue(
                code="high_correlation",
                severity="warning",
                columns=sorted({c for pair in strong_pairs for c in pair[:2]}),
                message=f"{len(strong_pairs)} variable pair(s) correlate above {thresholds['high_correlation']:.2f}.",
                detail={f"{a} ~ {b}": r for a, b, r in strong_pairs[:15]},
                suggested_action="Drop one of each pair, or use PCA / ridge / elastic net.",
            )
        )

    bad_vif = {k: v for k, v in profile.multicollinearity.items() if v >= thresholds["vif_warning"]}
    if bad_vif:
        worst = max(bad_vif.values())
        issues.append(
            QualityIssue(
                code="multicollinearity",
                severity="critical" if worst >= thresholds["vif_critical"] else "warning",
                columns=list(bad_vif),
                message=f"{len(bad_vif)} predictor(s) show inflated variance (max VIF {worst:.1f}).",
                detail=bad_vif,
                suggested_action="Coefficients are unreliable; regularise or reduce dimensions before interpreting them.",
            )
        )

    hi_card = [
        n for n, c in profile.columns.items()
        if c.semantic_type is SemanticType.HIGH_CARDINALITY_CATEGORICAL
    ]
    if hi_card:
        issues.append(
            QualityIssue(
                code="high_cardinality",
                severity="warning",
                columns=hi_card,
                message=f"{len(hi_card)} categorical column(s) have many distinct levels.",
                detail={c: profile.columns[c].n_unique for c in hi_card},
                suggested_action="One-hot encoding will explode; use target/frequency encoding or grouping.",
            )
        )

    for name in profile.identifier_columns:
        pass  # reported as a single info issue below
    if profile.identifier_columns:
        issues.append(
            QualityIssue(
                code="identifier_columns",
                severity="info",
                columns=profile.identifier_columns,
                message=f"{len(profile.identifier_columns)} column(s) look like identifiers.",
                detail={},
                suggested_action="Exclude from modelling — they memorise rows rather than generalise.",
            )
        )

    if profile.leakage_suspects:
        crit = [s for s in profile.leakage_suspects if s.get("severity") == "critical"]
        issues.append(
            QualityIssue(
                code="possible_leakage",
                severity="critical" if crit else "warning",
                columns=[s["column"] for s in profile.leakage_suspects],
                message=f"{len(profile.leakage_suspects)} column(s) may leak the outcome.",
                detail={s["column"]: s["reason"] for s in profile.leakage_suspects},
                suggested_action="Confirm each column is knowable *before* the outcome occurs; otherwise exclude it.",
            )
        )
    return issues


def class_imbalance(series: pd.Series, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Describe the class balance of a categorical target."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    counts = series.dropna().value_counts()
    if counts.empty:
        return {"imbalanced": False, "counts": {}}
    shares = counts / counts.sum()
    minority = float(shares.min())
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "shares": {str(k): round(float(v), 4) for k, v in shares.items()},
        "minority_class": str(shares.idxmin()),
        "minority_share": round(minority, 4),
        "imbalance_ratio": round(float(shares.max() / max(minority, 1e-9)), 2),
        "imbalanced": bool(minority < thresholds["imbalance_warning"]),
        "severely_imbalanced": bool(minority < thresholds["imbalance_critical"]),
        "n_classes": int(len(counts)),
    }


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def profile_dataset(
    frame: pd.DataFrame,
    name: str = "dataset",
    target: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[DatasetProfile, pd.DataFrame]:
    """Profile a DataFrame.

    Returns the profile and a *typed* copy of the frame where columns that were
    detected as dates or disguised numerics have been converted.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if frame is None or frame.shape[1] == 0:
        raise ValueError("Cannot profile an empty DataFrame (no columns).")

    full_rows = len(frame)
    working_frame = frame
    sampled = False
    if full_rows > thresholds["max_profile_rows"]:
        working_frame = frame.sample(int(thresholds["max_profile_rows"]), random_state=0)
        sampled = True

    n_rows = len(working_frame)
    profile = DatasetProfile(
        name=name,
        n_rows=full_rows,
        n_columns=int(frame.shape[1]),
        memory_mb=round(float(frame.memory_usage(deep=True).sum()) / 1e6, 3),
        sampled=sampled,
        sample_rows=n_rows if sampled else None,
    )

    typed = {}
    for col in working_frame.columns:
        col_profile, converted = profile_column(working_frame[col], str(col), n_rows, thresholds)
        col_profile.n_rows = full_rows
        profile.columns[str(col)] = col_profile
        typed[str(col)] = converted
    typed_frame = pd.DataFrame(typed, index=working_frame.index)

    try:
        dupes = int(working_frame.duplicated().sum())
    except TypeError:  # unhashable cell contents
        dupes = int(working_frame.astype(str).duplicated().sum())
    profile.n_duplicate_rows = dupes
    profile.duplicate_pct = round(100.0 * dupes / n_rows, 3) if n_rows else 0.0

    profile.identifier_columns = profile.names_by_semantic(SemanticType.IDENTIFIER)
    profile.datetime_columns = profile.names_by_semantic(SemanticType.DATETIME)
    profile.text_columns = profile.names_by_semantic(SemanticType.TEXT)
    profile.constant_columns = [n for n, c in profile.columns.items() if c.is_constant]

    numeric_cols = profile.numeric_columns
    profile.correlation_pairs, rels = _correlation_analysis(typed_frame, numeric_cols, thresholds)
    profile.relationships = rels
    profile.multicollinearity = _vif_scores(typed_frame, numeric_cols)
    profile.target_candidates = _target_candidates(profile, typed_frame, thresholds)
    profile.leakage_suspects = _leakage_suspects(profile, typed_frame, target, thresholds)
    profile.quality_issues = _detect_quality_issues(profile, typed_frame, thresholds)
    return profile, typed_frame


def apply_user_overrides(profile: DatasetProfile, context: BusinessContext) -> DatasetProfile:
    """Fold the user's corrections into a profile without losing what was measured.

    The user is the authority on meaning; the profiler is the authority on
    measurement. Overrides are marked so downstream text can attribute them.
    """
    for column, meaning in context.variable_meanings.items():
        if column in profile.columns:
            profile.columns[column].user_description = meaning
    for column in context.exclude_variables:
        if column in profile.columns:
            profile.columns[column].notes.append("User asked for this variable to be excluded.")
    if context.target_variable and context.target_variable in profile.columns:
        profile.columns[context.target_variable].notes.append("User nominated this as the target variable.")
    return profile


def override_semantic_type(profile: DatasetProfile, column: str, new_type: SemanticType) -> DatasetProfile:
    """Let the user correct the profiler's type inference."""
    if column not in profile.columns:
        raise KeyError(f"'{column}' is not in this dataset.")
    col = profile.columns[column]
    if col.semantic_type is not new_type:
        col.notes.append(f"Type changed by user from {col.semantic_type.value} to {new_type.value}.")
        col.semantic_type = new_type
        col.semantic_type_overridden = True
    return profile
