"""Natural-language commands, turned into analytical workflows.

Parsing is deterministic and rule-based, so the same question always produces
the same plan and the platform works with no API key. Where an LLM is
configured, it is used to *narrate* results — never to invent them — and the
narration is clearly labelled as such.

Every parsed command produces an :class:`Intent` the user can inspect *before*
it executes. For anything expensive or destructive that inspection is required,
not optional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, Objective, SemanticType, TaskType,
)


@dataclass
class Intent:
    """What the platform understood, and what it will do about it."""

    action: str                        # analyse | describe | model | segment | forecast |
                                       # anomaly | correlate | test | explain | recommend |
                                       # filter | compare | unknown
    original: str = ""
    task_type: TaskType | None = None
    target: str | None = None
    columns: list[str] = field(default_factory=list)
    group_by: str | None = None
    n_clusters: int | None = None
    horizon: int | None = None
    model_keys: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: Confidence = Confidence.MODERATE
    plan_preview: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    estimated_cost: str = "low"        # low | medium | high
    clarification: str = ""
    unresolved: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"**Understood as:** {self.action.replace('_', ' ')}"]
        if self.task_type:
            lines.append(f"**Analysis type:** {self.task_type.value.replace('_', ' ')}")
        if self.target:
            lines.append(f"**Target variable:** `{self.target}`")
        if self.columns:
            lines.append(f"**Variables:** {', '.join(f'`{c}`' for c in self.columns)}")
        if self.group_by:
            lines.append(f"**Grouped by:** `{self.group_by}`")
        if self.n_clusters:
            lines.append(f"**Segments:** {self.n_clusters}")
        if self.horizon:
            lines.append(f"**Forecast horizon:** {self.horizon} period(s)")
        if self.filters:
            lines.append(f"**Filters:** {self.filters}")
        if self.plan_preview:
            lines.append("\n**What will run:**")
            lines += [f"{i + 1}. {step}" for i, step in enumerate(self.plan_preview)]
        if self.unresolved:
            lines.append("\n**Could not resolve:** " + "; ".join(self.unresolved))
        if self.clarification:
            lines.append(f"\n**Needs clarification:** {self.clarification}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

ACTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("segment", [r"\bsegment", r"\bcluster", r"\bgroup(s|ing)? (the|my|these)", r"\bpersonas?\b",
                 r"\bnatural groups\b", r"\btypes? of (customer|client|user|product)"]),
    ("forecast", [r"\bforecast\b", r"\bpredict .*(next|coming|future)", r"\bnext (month|quarter|year|week|day)",
                  r"\bproject(ion)? (forward|ahead)", r"\bhow (much|many) .* next\b"]),
    ("anomaly", [r"\banomal", r"\boutlier", r"\bunusual\b", r"\bstrange\b", r"\bsuspicious\b",
                 r"\bodd\b", r"\bfraud\b", r"\bdoes ?n[o']t fit\b"]),
    ("model", [r"\bpredict\b", r"\bmodel\b", r"\bclassif", r"\bregress", r"\bestimate\b",
               r"\bwhich .*(will|are likely)", r"\bbuild a model\b", r"\btry (every|all)"]),
    ("correlate", [r"\bcorrelat", r"\brelationship between\b", r"\brelated to\b",
                   r"\bassociat(ed|ion) (with|between)\b", r"\blinked to\b"]),
    ("test", [r"\bsignifican(t|ce|tly)\b", r"\bhypothesis\b", r"\bt.?test\b", r"\banova\b", r"\bchi.?square\b",
              r"\bis there a difference\b", r"\bcompare .* (groups|between)\b", r"\bdiffer\b"]),
    ("explain", [r"\bwhy did you\b", r"\bexplain\b", r"\bwhy (are|is|did|do|does|has|have)\b",
                 r"\bwhat(?:'s| is) (?:behind|causing|driving)\b", r"\bwhat does .* mean\b", r"\bhow did you\b",
                 r"\bwhat drives\b", r"\bwhat (affects|influences|causes)\b", r"\bmost important\b",
                 r"\bwhich variables?\b"]),
    ("recommend", [r"\bwhat should i\b", r"\brecommend", r"\bwhat next\b", r"\bwhat to do\b",
                   r"\bsuggest\b", r"\binvestigate next\b", r"\bwhich .*(remove|drop)\b"]),
    ("rank", [r"\b(most|least) (valuable|profitable|important|active|engaged|expensive|costly)\b",
              r"\bwho (are|is) (my|the|our) (best|top|worst|biggest)\b",
              r"\btop \d+ (customer|client|product|region|segment)",
              r"\brank (the|my|these|customers|products)\b",
              r"\bbiggest (spender|customer|contributor)"]),
    ("describe", [r"\bdescribe\b", r"\bsummar(y|ise|ize)\b", r"\boverview\b", r"\bwhat('| i)s in\b",
                  r"\bhow many\b", r"\bshow me\b", r"\bprofile\b", r"\bdistribution of\b"]),
    ("analyse", [r"\banaly[sz]e (this|the|my)\b", r"\brun (a full|the) analysis\b",
                 r"\bdo the analysis\b", r"\bfull analysis\b", r"\bwork it out\b"]),
    ("compare", [r"\bcompare (models|algorithms)\b", r"\bwhich model\b", r"\bbest model\b",
                 r"\bmodel comparison\b", r"\btry (different|another)\b"]),
]

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_K_PATTERN = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}"
    r"(?:segments?|clusters?|groups?|personas?)\b", re.IGNORECASE
)
_HORIZON_PATTERN = re.compile(
    r"\b(?:next|coming|following)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)?\s*"
    r"(day|days|week|weeks|month|months|quarter|quarters|year|years|period|periods)\b", re.IGNORECASE
)
_TOP_N_PATTERN = re.compile(r"\b(?:top|first|best)\s+(\d+)\b", re.IGNORECASE)

MODEL_KEYWORDS = {
    "random forest": ["random_forest_regressor", "random_forest_classifier"],
    "xgboost": ["xgboost_regressor", "xgboost_classifier"],
    "lightgbm": ["lightgbm_regressor", "lightgbm_classifier"],
    "linear regression": ["linear_regression", "ols_statsmodels"],
    "logistic regression": ["logistic_regression"],
    "ridge": ["ridge", "ridge_classifier"],
    "lasso": ["lasso"],
    "elastic net": ["elastic_net"],
    "decision tree": ["decision_tree_regressor", "decision_tree_classifier"],
    "gradient boosting": ["gradient_boosting_regressor", "gradient_boosting_classifier"],
    "neural network": ["mlp_regressor", "mlp_classifier"],
    "svm": ["svr_rbf", "svc_rbf"],
    "knn": ["knn_regressor", "knn_classifier"],
    "naive bayes": ["gaussian_nb"],
    "k-means": ["kmeans"],
    "kmeans": ["kmeans"],
    "dbscan": ["dbscan"],
    "hierarchical": ["agglomerative"],
    "arima": ["arima", "auto_arima"],
    "sarima": ["sarima"],
    "holt": ["holt_winters"],
    "prophet": ["unobserved_components"],  # nearest available equivalent
    "isolation forest": ["isolation_forest"],
}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_command(
    text: str,
    profile: DatasetProfile,
    context: BusinessContext | None = None,
    last_run: Any = None,
) -> Intent:
    """Turn a question into an inspectable plan."""
    lowered = text.lower().strip()
    intent = Intent(action="unknown", original=text)
    if not lowered:
        intent.clarification = "Nothing was asked."
        return intent

    intent.action = _detect_action(lowered)
    mentioned = _columns_mentioned(lowered, profile)
    intent.columns = mentioned
    intent.filters = _extract_filters(text, profile)
    intent.parameters = _extract_parameters(lowered)

    handler: Callable[[Intent, str, DatasetProfile, Any, BusinessContext | None], Intent] = {
        "segment": _plan_segment,
        "forecast": _plan_forecast,
        "anomaly": _plan_anomaly,
        "model": _plan_model,
        "correlate": _plan_correlate,
        "test": _plan_test,
        "explain": _plan_explain,
        "recommend": _plan_recommend,
        "describe": _plan_describe,
        "analyse": _plan_analyse,
        "compare": _plan_compare,
        "rank": _plan_rank,
    }.get(intent.action, _plan_unknown)

    intent = handler(intent, lowered, profile, last_run, context)
    intent.model_keys = _requested_models(lowered, intent.task_type)
    if intent.model_keys:
        intent.plan_preview.append(f"Use the model(s) you named: {', '.join(intent.model_keys)}")

    intent.requires_confirmation = intent.estimated_cost in {"medium", "high"} or bool(intent.clarification)
    return intent


def _detect_action(text: str) -> str:
    for action, patterns in ACTION_PATTERNS:
        if any(re.search(p, text) for p in patterns):
            return action
    return "unknown"


#: Suffixes stripped when matching a column name, so "churn" finds "churned"
#: and "sale" finds "sales". Deliberately short — aggressive stemming produces
#: false matches that are worse than a missed one.
_STEM_SUFFIXES = ("ed", "s", "ing", "_flag", "_amount", "_total", "_value")


def _stem(word: str) -> str:
    for suffix in _STEM_SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _columns_mentioned(text: str, profile: DatasetProfile) -> list[str]:
    """Longest names first, so 'annual spend' beats 'spend'."""
    hits: list[str] = []
    words = set(re.findall(r"[a-z_]+", text))
    stems = {_stem(w) for w in words}
    for name in sorted(profile.columns, key=len, reverse=True):
        lowered = name.lower()
        readable = lowered.replace("_", " ")
        if not readable:
            continue
        matched = (
            re.search(rf"\b{re.escape(readable)}\b", text)
            or re.search(rf"\b{re.escape(lowered)}\b", text)
            # stem match, e.g. the user typed "churn" and the column is "churned"
            or (len(lowered) > 3 and _stem(lowered) in stems)
        )
        if matched and not any(lowered in h.lower() and name != h for h in hits):
            hits.append(name)
    return hits


def _extract_filters(text: str, profile: DatasetProfile) -> dict[str, Any]:
    """Simple comparisons the user typed, e.g. 'where income > 50000'."""
    filters: dict[str, Any] = {}
    pattern = re.compile(
        r"\b(\w+)\s*(>=|<=|>|<|=|is|equals?)\s*['\"]?([\w.\- ]+?)['\"]?(?=\s|$|,|\.)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        column, operator, value = match.group(1), match.group(2), match.group(3).strip()
        resolved = next(
            (c for c in profile.columns if c.lower() == column.lower()
             or c.lower().replace("_", " ") == column.lower()),
            None,
        )
        if resolved is None:
            continue
        operator = {"is": "==", "equals": "==", "equal": "==", "=": "=="}.get(operator.lower(), operator)
        try:
            typed_value: Any = float(value)
        except ValueError:
            typed_value = value
        filters[resolved] = {"operator": operator, "value": typed_value}
    return filters


def _extract_parameters(text: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    top_n = _TOP_N_PATTERN.search(text)
    if top_n:
        parameters["top_n"] = int(top_n.group(1))
    if re.search(r"\bfast\b|\bquick", text):
        parameters["time_budget"] = "fast"
    if re.search(r"\bthorough\b|\bevery\b|\ball (suitable|possible)\b|\bexhaustive", text):
        parameters["time_budget"] = "thorough"
        parameters["max_models"] = 15
    if re.search(r"\bexplain|\binterpretab|\bsimple model\b|\btransparent", text):
        parameters["interpretability_need"] = "high"
    if re.search(r"\btune|\boptimi[sz]e (the )?(hyper)?parameters?\b", text):
        parameters["tune_hyperparameters"] = True
    return parameters


def _requested_models(text: str, task_type: TaskType | None = None) -> list[str]:
    """Models the user named, filtered to those that fit the inferred task.

    "Use random forest" on a regression problem must not also offer the
    classifier — naming a family is not naming a task.
    """
    from dsai.registry.base import REGISTRY

    found: list[str] = []
    for keyword, keys in MODEL_KEYWORDS.items():
        if keyword in text:
            for key in keys:
                if key not in REGISTRY:
                    continue
                if task_type is not None and task_type not in REGISTRY.get(key).task_types:
                    continue
                found.append(key)
    return list(dict.fromkeys(found))


# --------------------------------------------------------------------------
# per-action planners
# --------------------------------------------------------------------------

def _numeric_targets(profile: DatasetProfile, mentioned: list[str]) -> str | None:
    for name in mentioned:
        column = profile.columns.get(name)
        if column is not None and column.is_numeric:
            return name
    for candidate in profile.target_candidates:
        if candidate["implied_task"] == "regression":
            return candidate["column"]
    return None


def _plan_segment(intent, text, profile, last_run, context) -> Intent:
    intent.task_type = TaskType.CLUSTERING
    match = _K_PATTERN.search(text)
    if match:
        raw = match.group(1)
        intent.n_clusters = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw.lower())
    features = intent.columns or [
        c for c in profile.modelling_columns
        if profile.columns[c].semantic_type is not SemanticType.DATETIME
    ]
    intent.columns = features
    intent.estimated_cost = "medium"
    intent.plan_preview = [
        f"Use {len(features)} variable(s) as clustering inputs",
        "Scale them so no variable dominates purely because of its units",
        (f"Create exactly {intent.n_clusters} segments as you asked"
         if intent.n_clusters else
         "Sweep the number of segments and compare elbow, silhouette, Calinski-Harabasz "
         "and Davies-Bouldin to choose it"),
        "Run several clustering algorithms and compare their internal validity",
        "Profile each segment in the original units, not standard deviations",
        "Report what makes each segment distinctive, and flag any too small to act on",
    ]
    if not intent.n_clusters:
        intent.confidence = Confidence.MODERATE
    return intent


def _plan_forecast(intent, text, profile, last_run, context) -> Intent:
    intent.task_type = TaskType.TIME_SERIES_FORECAST
    match = _HORIZON_PATTERN.search(text)
    if match:
        raw = match.group(1)
        intent.horizon = int(raw) if raw and raw.isdigit() else _NUMBER_WORDS.get((raw or "one").lower(), 1)
    intent.target = _numeric_targets(profile, intent.columns)
    time_column = next((c for c in intent.columns if c in profile.datetime_columns), None)
    time_column = time_column or (profile.datetime_columns[0] if profile.datetime_columns else None)
    intent.parameters["time_column"] = time_column
    intent.estimated_cost = "high"

    if time_column is None:
        intent.clarification = (
            "No date column was found, so there is no time dimension to project along. "
            "Tell me which column holds the date, or convert it first."
        )
        intent.confidence = Confidence.LOW
        return intent
    if intent.target is None:
        intent.clarification = "Which numeric column should be forecast?"
        intent.confidence = Confidence.LOW
        return intent

    intent.plan_preview = [
        f"Build the series: `{intent.target}` ordered by `{time_column}`",
        "Test for trend, seasonality, stationarity, gaps and structural breaks",
        f"Forecast {intent.horizon or 'the default'} period(s) ahead",
        "Compare statistical forecasters (exponential smoothing, ARIMA/SARIMA, Theta, state-space) "
        "against machine-learning forecasters on lag features",
        "Include a naive last-value baseline — any model that cannot beat it is not adding value",
        "Back-test with a rolling origin, never with random folds",
        "Report prediction intervals where the model provides them",
    ]
    return intent


def _plan_anomaly(intent, text, profile, last_run, context) -> Intent:
    intent.task_type = TaskType.ANOMALY_DETECTION
    features = [c for c in (intent.columns or profile.numeric_columns) if c in profile.columns]
    intent.columns = features or profile.numeric_columns
    intent.estimated_cost = "medium"
    intent.plan_preview = [
        f"Use {len(intent.columns)} numeric variable(s)",
        "Scale them so distance-based detectors are not dominated by one unit",
        "Run several detectors: Isolation Forest, Local Outlier Factor, robust Mahalanobis distance, "
        "and simple Z-score/IQR rules for comparability",
        "Flag rows and give each an anomaly score",
        "Explain each flagged row by which variables are unusual and by how much",
    ]
    intent.clarification = (
        "There are no labels, so 'anomalous' means 'unlike the rest of this data'. Whether a "
        "flagged row is actually a problem is a judgement only you can make."
    )
    return intent


def _plan_model(intent, text, profile, last_run, context) -> Intent:
    target = None
    for name in intent.columns:
        column = profile.columns.get(name)
        if column is not None and column.semantic_type not in {SemanticType.IDENTIFIER, SemanticType.DATETIME}:
            target = name
            break
    target = target or (context.target_variable if context else None)
    if target is None and profile.target_candidates:
        target = profile.target_candidates[0]["column"]

    if target is None:
        intent.clarification = "Which column should be predicted?"
        intent.confidence = Confidence.LOW
        intent.estimated_cost = "low"
        return intent

    intent.target = target
    column = profile.columns[target]
    if column.semantic_type is SemanticType.BINARY:
        intent.task_type = TaskType.BINARY_CLASSIFICATION
    elif column.is_categorical or (column.semantic_type is SemanticType.NUMERIC_DISCRETE and column.n_unique <= 10):
        intent.task_type = TaskType.MULTICLASS_CLASSIFICATION
    else:
        intent.task_type = TaskType.REGRESSION

    intent.estimated_cost = "high" if intent.parameters.get("time_budget") == "thorough" else "medium"
    intent.plan_preview = [
        f"Treat this as {intent.task_type.value.replace('_', ' ')} on `{target}`",
        "Check for leakage: any column that would not exist before the outcome occurs",
        "Design preprocessing to suit each candidate model, not a fixed recipe",
        f"Train and cross-validate {intent.parameters.get('max_models', 8)} candidate model(s), "
        "including a baseline that ignores every predictor",
        "Rank on performance, generalisation, stability, interpretability and cost — not the raw score alone",
        "Explain the selected model and check its residuals",
    ]
    return intent


def _plan_correlate(intent, text, profile, last_run, context) -> Intent:
    intent.task_type = TaskType.EXPLORATORY
    columns = [c for c in intent.columns if profile.columns[c].is_numeric] or profile.numeric_columns
    intent.columns = columns
    intent.estimated_cost = "low"
    intent.plan_preview = [
        f"Compute Pearson, Spearman and Kendall correlations across {len(columns)} numeric variable(s)",
        "Report the coefficient, sample size and p-value for each pair",
        "Flag pairs where the rank correlation is much stronger than the linear one — that is curvature",
        "Compute VIF to show which variables are carrying the same information",
    ]
    intent.clarification = (
        "Correlation shows what moves together. It cannot show what causes what — a third "
        "variable, reverse causation or coincidence produce the same number."
    )
    return intent


def _plan_test(intent, text, profile, last_run, context) -> Intent:
    intent.task_type = TaskType.HYPOTHESIS_TESTING
    numeric = [c for c in intent.columns if profile.columns[c].is_numeric]
    categorical = [c for c in intent.columns if profile.columns[c].is_categorical]
    intent.estimated_cost = "low"

    if numeric and categorical:
        intent.parameters["value_column"] = numeric[0]
        intent.group_by = categorical[0]
        groups = profile.columns[categorical[0]].n_unique
        test = "t-test (or Mann-Whitney if the data is not normal)" if groups == 2 else \
               "one-way ANOVA (or Kruskal-Wallis if assumptions fail)"
        intent.plan_preview = [
            f"Compare `{numeric[0]}` across the {groups} level(s) of `{categorical[0]}`",
            "Check normality (Shapiro-Wilk) and equal variance (Levene) first",
            f"Run a {test}",
            "Report the effect size alongside the p-value, and say whether the difference is "
            "large enough to matter in practice",
        ]
    elif len(numeric) >= 2:
        intent.plan_preview = [
            f"Test whether `{numeric[0]}` and `{numeric[1]}` are correlated",
            "Report r, the p-value, and the share of variation they share",
        ]
    elif len(categorical) >= 2:
        intent.plan_preview = [
            f"Test whether `{categorical[0]}` and `{categorical[1]}` are independent (chi-square)",
            "Fall back to Fisher's exact test if any expected cell count is below 5",
            "Report Cramér's V as the effect size",
        ]
    else:
        intent.clarification = (
            "Which variables should be compared? Name a numeric measure and a grouping variable, "
            "or two variables to test for association."
        )
        intent.confidence = Confidence.LOW
    return intent


def _plan_explain(intent, text, profile, last_run, context) -> Intent:
    intent.action = "explain"
    intent.estimated_cost = "low"
    if last_run is None:
        # "What drives revenue?" with nothing run yet is a request to find out,
        # not an error. Fall through to modelling, and say so.
        target = _numeric_targets(profile, intent.columns)
        if target is None and profile.target_candidates:
            target = profile.target_candidates[0]["column"]
        if target is not None:
            intent.action = "model"
            fallback = _plan_model(intent, text, profile, last_run, context)
            fallback.plan_preview.insert(
                0, f"No analysis has been run yet, so one will be run first to answer this about `{target}`"
            )
            fallback.plan_preview.append("Then report the ranked drivers and how each relates to the outcome")
            return fallback
        intent.clarification = "There is no completed analysis to explain yet, and no obvious outcome variable to model."
        intent.confidence = Confidence.LOW
        return intent
    intent.plan_preview = [
        "Show the ranked variable importances and the method used to compute them",
        "Show the partial dependence for the strongest driver",
        "Show the decision log: what was chosen, why, and what was rejected",
        "State the difference between association and causation for these findings",
    ]
    return intent


def _plan_recommend(intent, text, profile, last_run, context) -> Intent:
    intent.estimated_cost = "low"
    if last_run is None:
        intent.plan_preview = [
            "No analysis has been run yet, so recommendations will be based on the data profile alone:",
            "data-quality fixes, which objectives this dataset can support, and what to collect next",
        ]
    else:
        intent.plan_preview = [
            "Show the evidence-linked recommendations from the last run",
            "Grouped by risk, data quality, actions and what to investigate next",
            "Each with its evidence, confidence and caveats",
        ]
    return intent


def _plan_describe(intent, text, profile, last_run, context) -> Intent:
    intent.task_type = TaskType.EXPLORATORY
    intent.estimated_cost = "low"
    columns = intent.columns or profile.modelling_columns[:10]
    intent.columns = columns
    intent.plan_preview = [
        f"Summarise {len(columns)} variable(s): centre, spread, shape, missingness and outliers",
        "Show distributions where the shape is not symmetric",
        "Report the data-quality issues found and what they mean for any analysis",
    ]
    return intent


def _plan_analyse(intent, text, profile, last_run, context) -> Intent:
    intent.action = "analyse"
    intent.estimated_cost = "high"
    intent.plan_preview = [
        "Profile every variable and assess data quality",
        "Identify what analytical objectives this data supports, and pick the strongest",
        "Design preprocessing suited to the data and the candidate models",
        "Train and cross-validate a shortlist spanning several model families, including a baseline",
        "Compare them on performance, generalisation, stability, interpretability and cost",
        "Explain the selected model and check its assumptions",
        "Generate findings, run validation checks against them, and produce recommendations",
    ]
    intent.confidence = Confidence.HIGH
    return intent


def _plan_compare(intent, text, profile, last_run, context) -> Intent:
    intent.action = "compare"
    intent.estimated_cost = "high"
    intent.parameters.setdefault("max_models", 12)
    intent.plan_preview = [
        "Widen the shortlist to cover every suitable model family",
        "Train and cross-validate each under the same preprocessing discipline",
        "Present the full tournament with every metric appropriate to this problem",
        "Recommend one, and separately name the highest-scoring and the simplest acceptable model",
    ]
    return intent


def _plan_rank(intent, text, profile, last_run, context) -> Intent:
    """"Who are my best customers?" — a ranking question, not a modelling one."""
    intent.task_type = TaskType.EXPLORATORY
    intent.estimated_cost = "low"
    value = _numeric_targets(profile, intent.columns)
    intent.target = value
    intent.parameters.setdefault("top_n", 20)

    if value is None:
        intent.clarification = (
            "Ranked by what? Name the numeric column that defines 'valuable' — spend, revenue, "
            "margin, lifetime value — because they can rank the same customers very differently."
        )
        intent.confidence = Confidence.LOW
        return intent

    identifier = profile.identifier_columns[0] if profile.identifier_columns else None
    intent.plan_preview = [
        f"Rank rows by `{value}`" + (f", labelled by `{identifier}`" if identifier else ""),
        f"Show the top {intent.parameters['top_n']}",
        f"Show what share of total `{value}` the top rows account for",
        "Compare the top group against the rest on the other variables",
    ]
    intent.clarification = (
        f"This ranks by `{value}` as it stands today. Highest current value is not the same as "
        "highest potential, highest margin, or lowest risk of leaving — those are different questions."
    )
    return intent


def _plan_unknown(intent, text, profile, last_run, context) -> Intent:
    intent.confidence = Confidence.LOW
    intent.clarification = (
        "I could not tell what kind of analysis you want. Try naming the goal directly — "
        "for example: 'predict annual spend', 'segment these customers into five groups', "
        "'forecast sales for the next six months', 'find unusual customers', or "
        "'is the difference between regions significant?'"
    )
    intent.plan_preview = ["Nothing will run until the question is clearer."]
    return intent


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def intent_to_objective(intent: Intent, profile: DatasetProfile) -> Objective | None:
    """Convert an approved intent into an objective the orchestrator can run."""
    if intent.task_type is None:
        return None
    return Objective(
        task_type=intent.task_type,
        target=intent.target,
        features=intent.columns if intent.task_type in
                 {TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION, TaskType.DIMENSIONALITY_REDUCTION}
                 else [],
        time_column=intent.parameters.get("time_column"),
        group_column=intent.group_by,
        horizon=intent.horizon,
        n_clusters=intent.n_clusters,
        rationale=f"You asked: “{intent.original}”",
        confidence=intent.confidence,
        source="user_supplied",
        priority=1.0,
    )


def apply_filters(frame: pd.DataFrame, filters: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the comparisons parsed out of the question, reporting what each did."""
    working = frame
    log: list[str] = []
    for column, condition in filters.items():
        if column not in working.columns:
            continue
        operator, value = condition["operator"], condition["value"]
        before = len(working)
        try:
            series = working[column]
            if operator == "==":
                mask = series.astype(str).str.lower() == str(value).lower()
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                mask = {
                    ">": numeric > value, ">=": numeric >= value,
                    "<": numeric < value, "<=": numeric <= value,
                }[operator]
            working = working[mask.fillna(False)]
            log.append(f"`{column}` {operator} {value}: {before} → {len(working)} rows")
        except Exception as exc:
            log.append(f"Could not apply `{column}` {operator} {value}: {exc}")
    return working, log


def answer_question(
    text: str,
    frame: pd.DataFrame,
    profile: DatasetProfile,
    context: BusinessContext | None = None,
    last_run: Any = None,
) -> dict[str, Any]:
    """Answer a question that needs no model run: descriptive and statistical queries."""
    intent = parse_command(text, profile, context, last_run)
    working, filter_log = apply_filters(frame, intent.filters)

    if intent.action == "describe":
        from dsai.statistics.descriptive import describe_categorical, describe_numeric

        numeric = [c for c in intent.columns if c in working.columns
                   and pd.api.types.is_numeric_dtype(working[c])]
        categorical = [c for c in intent.columns if c in working.columns and c not in numeric]
        return {
            "intent": intent,
            "filters": filter_log,
            "numeric_summary": describe_numeric(working, numeric) if numeric else None,
            "categorical_summary": describe_categorical(working, categorical) if categorical else None,
            "answer": _describe_answer(working, numeric, categorical, profile),
        }

    if intent.action == "correlate":
        from dsai.statistics.descriptive import correlation_pairs

        pairs = correlation_pairs(working, columns=intent.columns, min_abs=0.1)
        top = pairs.head(10) if not pairs.empty else pairs
        return {
            "intent": intent,
            "filters": filter_log,
            "correlations": top,
            "answer": _correlation_answer(top),
        }

    if intent.action == "test":
        return {"intent": intent, "filters": filter_log, "answer": _run_test(intent, working, profile)}

    return {"intent": intent, "filters": filter_log, "answer": None}


def _describe_answer(frame: pd.DataFrame, numeric: list[str], categorical: list[str],
                     profile: DatasetProfile) -> str:
    lines = [f"{len(frame):,} rows across {frame.shape[1]} columns."]
    for column in numeric[:5]:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        lines.append(
            f"**{column}**: median {series.median():,.4g}, mean {series.mean():,.4g}, "
            f"ranging {series.min():,.4g} to {series.max():,.4g}"
            + (f". Skewed ({series.skew():.1f}), so the median is the better summary."
               if abs(series.skew()) > 1 else ".")
        )
    for column in categorical[:5]:
        counts = frame[column].value_counts()
        if counts.empty:
            continue
        lines.append(
            f"**{column}**: {len(counts)} distinct value(s); most common is "
            f"'{counts.index[0]}' at {counts.iloc[0] / counts.sum():.1%}."
        )
    return "\n\n".join(lines)


def _correlation_answer(pairs: pd.DataFrame) -> str:
    if pairs.empty:
        return "No pair of numeric variables correlates above 0.1 — they carry largely independent information."
    lines = ["Strongest relationships found:"]
    for row in pairs.head(5).itertuples():
        significance = "statistically significant" if getattr(row, "significant_at_5pct", False) else "not statistically significant"
        lines.append(
            f"- **{row.variable_1}** and **{row.variable_2}**: r = {row.coefficient:.3f} "
            f"({row.strength}, {significance}, n = {row.n})"
        )
    lines.append(
        "\nThese are associations. Any of them could be one variable driving the other, both being "
        "driven by something unmeasured, or coincidence — the correlation itself cannot distinguish those."
    )
    return "\n".join(lines)


def _run_test(intent: Intent, frame: pd.DataFrame, profile: DatasetProfile) -> str:
    from dsai.statistics import tests as T

    value = intent.parameters.get("value_column")
    group = intent.group_by
    if value and group and value in frame.columns and group in frame.columns:
        n_groups = frame[group].nunique()
        if n_groups == 2:
            levels = frame[group].dropna().unique()[:2]
            a = frame.loc[frame[group] == levels[0], value]
            b = frame.loc[frame[group] == levels[1], value]
            result = T.t_test(a, b, names=(str(levels[0]), str(levels[1])))
            alternative = T.mann_whitney(a, b, names=(str(levels[0]), str(levels[1])))
            return (
                result.summarise() + "\n\n" + result.conclusion
                + "\n\nNon-parametric cross-check: " + alternative.summarise()
            )
        result = T.anova(frame, value, group)
        alternative = T.kruskal_wallis(frame, value, group)
        return result.summarise() + "\n\n" + result.conclusion + "\n\nNon-parametric cross-check: " + alternative.summarise()

    numeric = [c for c in intent.columns if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric) >= 2:
        result = T.correlation_test(frame[numeric[0]], frame[numeric[1]])
        return result.summarise() + "\n\n" + result.conclusion

    categorical = [c for c in intent.columns if c in frame.columns and c not in numeric]
    if len(categorical) >= 2:
        result = T.chi_square(frame, categorical[0], categorical[1])
        return result.summarise() + "\n\n" + result.conclusion
    return intent.clarification or "Not enough information to run a test."
