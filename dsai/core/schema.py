"""Core data structures shared across every engine in the platform.

Everything the engines exchange is a plain dataclass so that a run can be
serialised to JSON for reproducibility, diffed between experiments, and
rendered by any front end (Streamlit, CLI, report builder).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
from dataclasses import dataclass, field
from typing import Any


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonMixin:
    """Adds dict/JSON serialisation to any dataclass."""

    def to_dict(self) -> dict[str, Any]:
        return {k: _jsonable(v) for k, v in dataclasses.asdict(self).items()}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# --------------------------------------------------------------------------
# Semantic vocabulary
# --------------------------------------------------------------------------


class SemanticType(str, enum.Enum):
    """What a column *means*, as opposed to how pandas stores it."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    BINARY = "binary"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    HIGH_CARDINALITY_CATEGORICAL = "high_cardinality_categorical"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


NUMERIC_SEMANTICS = {
    SemanticType.NUMERIC_CONTINUOUS,
    SemanticType.NUMERIC_DISCRETE,
}
CATEGORICAL_SEMANTICS = {
    SemanticType.BINARY,
    SemanticType.CATEGORICAL_NOMINAL,
    SemanticType.CATEGORICAL_ORDINAL,
    SemanticType.HIGH_CARDINALITY_CATEGORICAL,
}


class TaskType(str, enum.Enum):
    """The analytical problem the platform is being asked to solve."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    TIME_SERIES_FORECAST = "time_series_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    ASSOCIATION_RULES = "association_rules"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    EXPLORATORY = "exploratory"

    @property
    def is_supervised(self) -> bool:
        return self in {
            TaskType.REGRESSION,
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.TIME_SERIES_FORECAST,
        }

    @property
    def is_classification(self) -> bool:
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }


class Confidence(str, enum.Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    SPECULATIVE = "speculative"


class EvidenceKind(str, enum.Enum):
    """Keeps observed fact, statistical test, model output and interpretation apart.

    The platform must never present an interpretation as if it were a measured
    fact, so every statement carries the kind of backing it actually has.
    """

    OBSERVED = "observed_in_data"
    STATISTICAL = "statistical_test"
    MODEL = "model_derived"
    INTERPRETATION = "ai_interpretation"
    USER_ASSUMPTION = "user_assumption"


# --------------------------------------------------------------------------
# Column and dataset profiles
# --------------------------------------------------------------------------


@dataclass
class ColumnProfile(JsonMixin):
    name: str
    dtype: str
    semantic_type: SemanticType = SemanticType.UNKNOWN
    n_rows: int = 0
    n_missing: int = 0
    missing_pct: float = 0.0
    n_unique: int = 0
    unique_pct: float = 0.0
    is_constant: bool = False
    is_near_constant: bool = False
    dominant_value: Any = None
    dominant_pct: float = 0.0
    # numeric summaries
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    maximum: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zeros_pct: float | None = None
    negative_pct: float | None = None
    n_outliers_iqr: int = 0
    outlier_pct: float = 0.0
    # categorical summaries
    top_values: dict[str, int] = field(default_factory=dict)
    cardinality_ratio: float = 0.0
    # datetime summaries
    min_date: str | None = None
    max_date: str | None = None
    inferred_frequency: str | None = None
    n_missing_periods: int | None = None
    # text summaries
    mean_token_count: float | None = None
    # provenance / user overrides
    user_description: str | None = None
    semantic_type_overridden: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.semantic_type in NUMERIC_SEMANTICS

    @property
    def is_categorical(self) -> bool:
        return self.semantic_type in CATEGORICAL_SEMANTICS


@dataclass
class QualityIssue(JsonMixin):
    code: str
    severity: str  # "critical" | "warning" | "info"
    columns: list[str]
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    suggested_action: str | None = None


@dataclass
class RelationshipFinding(JsonMixin):
    kind: str  # "correlation" | "association" | "group_difference" | "nonlinearity"
    columns: list[str]
    statistic: float
    method: str
    p_value: float | None = None
    strength: str = "weak"
    message: str = ""


@dataclass
class DatasetProfile(JsonMixin):
    """The structured internal profile every downstream engine reasons over."""

    name: str = "dataset"
    n_rows: int = 0
    n_columns: int = 0
    memory_mb: float = 0.0
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    n_duplicate_rows: int = 0
    duplicate_pct: float = 0.0
    quality_issues: list[QualityIssue] = field(default_factory=list)
    relationships: list[RelationshipFinding] = field(default_factory=list)
    correlation_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    multicollinearity: dict[str, float] = field(default_factory=dict)
    target_candidates: list[dict[str, Any]] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    leakage_suspects: list[dict[str, Any]] = field(default_factory=list)
    profiled_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    sampled: bool = False
    sample_rows: int | None = None

    # -- convenience accessors -------------------------------------------------
    def names_by_semantic(self, *types: SemanticType) -> list[str]:
        wanted = set(types)
        return [n for n, c in self.columns.items() if c.semantic_type in wanted]

    @property
    def numeric_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.is_numeric]

    @property
    def categorical_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.is_categorical]

    @property
    def modelling_columns(self) -> list[str]:
        """Columns that are plausible model inputs (excludes ids/constants)."""
        skip = {SemanticType.IDENTIFIER, SemanticType.CONSTANT, SemanticType.UNKNOWN}
        return [n for n, c in self.columns.items() if c.semantic_type not in skip]

    def issues_by_severity(self, severity: str) -> list[QualityIssue]:
        return [i for i in self.quality_issues if i.severity == severity]

    @property
    def quality_score(self) -> float:
        """0-100 heuristic score. Deliberately blunt: it flags, it does not judge."""
        penalty = 0.0
        for issue in self.quality_issues:
            penalty += {"critical": 12.0, "warning": 5.0, "info": 1.0}.get(issue.severity, 1.0)
        return round(max(0.0, 100.0 - penalty), 1)


# --------------------------------------------------------------------------
# User-supplied business context
# --------------------------------------------------------------------------


@dataclass
class BusinessContext(JsonMixin):
    """Everything the *user* asserts. Never mixed with what was measured."""

    description: str = ""
    data_source: str = ""
    business_problem: str = ""
    research_question: str = ""
    desired_outcome: str = ""
    industry: str = ""
    time_period: str = ""
    geography: str = ""
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    variable_meanings: dict[str, str] = field(default_factory=dict)
    include_variables: list[str] = field(default_factory=list)
    exclude_variables: list[str] = field(default_factory=list)
    target_variable: str | None = None
    analysis_style: str = ""  # exploratory | predictive | explanatory | decision
    currency: str = "ZAR"
    stated_objective: str = ""

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.description, self.business_problem, self.research_question,
                self.desired_outcome, self.stated_objective, self.target_variable,
            ]
        )

    def as_prompt_block(self) -> str:
        lines = []
        mapping = {
            "What the data represents": self.description,
            "Where it came from": self.data_source,
            "Business problem": self.business_problem,
            "Research question": self.research_question,
            "Desired outcome": self.desired_outcome,
            "Industry": self.industry,
            "Time period": self.time_period,
            "Geography": self.geography,
            "Analysis style": self.analysis_style,
            "Stated objective": self.stated_objective,
        }
        for label, value in mapping.items():
            if value:
                lines.append(f"- {label}: {value}")
        for a in self.assumptions:
            lines.append(f"- User assumption: {a}")
        for l in self.limitations:
            lines.append(f"- User-stated limitation: {l}")
        for var, meaning in self.variable_meanings.items():
            lines.append(f"- Variable '{var}' means: {meaning}")
        return "\n".join(lines) if lines else "(no user context supplied)"


# --------------------------------------------------------------------------
# Objectives, decisions and findings
# --------------------------------------------------------------------------


@dataclass
class Objective(JsonMixin):
    task_type: TaskType
    target: str | None = None
    features: list[str] = field(default_factory=list)
    time_column: str | None = None
    group_column: str | None = None
    horizon: int | None = None
    n_clusters: int | None = None
    rationale: str = ""
    confidence: Confidence = Confidence.MODERATE
    source: str = "ai_detected"  # ai_detected | user_supplied | user_override
    priority: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        if self.target:
            return f"{self.task_type.value} on '{self.target}'"
        return self.task_type.value


@dataclass
class Decision(JsonMixin):
    """One auditable entry in the AI Decision Log.

    Deliberately a *conclusion + reason + evidence* record: it exposes what was
    decided and why, without dumping private reasoning.
    """

    stage: str
    decision: str
    reason: str
    evidence: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    confidence: Confidence = Confidence.MODERATE
    overridable: bool = True
    overridden_by_user: bool = False
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))


@dataclass
class Finding(JsonMixin):
    title: str
    detail: str
    kind: EvidenceKind
    #: Stable, content-derived identifier. Assigned by the evidence ledger so a
    #: claim can be cited and traced after it has left the application.
    id: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: Confidence = Confidence.MODERATE
    columns: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)


@dataclass
class Recommendation(JsonMixin):
    action: str
    reason: str
    #: Stable, content-derived identifier. See Finding.id.
    id: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: Confidence = Confidence.MODERATE
    expected_impact: str = ""
    caveats: list[str] = field(default_factory=list)
    traceable_to: list[str] = field(default_factory=list)  # experiment / analysis ids
    category: str = "action"  # action | data_quality | further_analysis | risk


@dataclass
class TraceEvent(JsonMixin):
    step: str
    status: str = "done"  # done | running | warning | failed | skipped
    detail: str = ""
    elapsed_s: float | None = None
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
