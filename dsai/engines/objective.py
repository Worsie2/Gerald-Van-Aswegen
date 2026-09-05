"""Objective detection: work out what analytical problem is actually on the table.

Two sources feed in, and they are kept apart deliberately:

* what the **user said** they want (authoritative, but often vague), and
* what the **data supports** (measured, but blind to intent).

When the two disagree, the user wins on intent and the data wins on feasibility
— and the disagreement is surfaced rather than silently resolved.
"""

from __future__ import annotations

import re
from typing import Any

from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, Objective, SemanticType, TaskType,
)

# Phrases that reliably indicate an analytical intent. Matched against the
# user's own words, so the vocabulary is deliberately business-facing.
INTENT_PATTERNS: dict[TaskType, list[str]] = {
    TaskType.REGRESSION: [
        r"\bpredict\b.*\b(amount|value|spend|spending|revenue|sales|price|cost|income|score|quantity|demand|volume|profit|margin)\b",
        r"\bestimate\b", r"\bhow much\b", r"\bforecast the (amount|value|level)\b",
        r"\bwhat drives\b", r"\bwhat (affects|influences|explains)\b.*\b(revenue|sales|spend|price|value)\b",
    ],
    TaskType.BINARY_CLASSIFICATION: [
        r"\b(churn|attrition|default|fraud|conversion|convert|retain|renew|cancel)\b",
        r"\bwhich .* (will|are likely to|likely)\b", r"\bpredict whether\b", r"\byes.?no\b",
        r"\bprobability (of|that)\b", r"\bwho will\b", r"\brisk of\b",
    ],
    TaskType.MULTICLASS_CLASSIFICATION: [
        r"\bclassify\b", r"\bcategor(y|ise|ize)\b", r"\bwhich (type|category|class|group) \b",
        r"\bassign .* to\b", r"\blabel\b",
    ],
    TaskType.CLUSTERING: [
        r"\bsegment(s|ed|ing|ation)?\b", r"\bcluster(s|ed|ing)?\b", r"\bgroup(s|ing)?\b",
        r"\bpersonas?\b", r"\btypes? of (customer|client|user|product)\b",
        r"\bnatural groups\b", r"\bwho are my\b",
    ],
    TaskType.TIME_SERIES_FORECAST: [
        r"\bforecast\b", r"\bnext (month|quarter|year|week|day|period)\b",
        r"\bfuture\b", r"\bover time\b", r"\btrend\b", r"\bseasonal\b",
        r"\bpredict .* (sales|demand|revenue) for\b",
    ],
    TaskType.ANOMALY_DETECTION: [
        r"\banomal(y|ies|ous)\b", r"\boutliers?\b", r"\bunusual\b", r"\bstrange\b",
        r"\bsuspicious\b", r"\bfraud detection\b", r"\bodd\b", r"\bdoes not fit\b",
    ],
    TaskType.ASSOCIATION_RULES: [
        r"\bbasket\b", r"\bbought together\b", r"\bpurchase together\b",
        r"\bcross.?sell\b", r"\bassociation rules?\b", r"\bwhat else\b", r"\bco.?occur\b",
    ],
    TaskType.DIMENSIONALITY_REDUCTION: [
        r"\breduce (the )?(dimensions?|variables?|features?)\b", r"\bpca\b",
        r"\bfactor analysis\b", r"\blatent\b", r"\bcompress\b",
    ],
    TaskType.HYPOTHESIS_TESTING: [
        r"\bsignificant\b", r"\bhypothesis\b", r"\bt.?test\b", r"\banova\b",
        r"\bis there a difference\b", r"\bcompare .* (groups|between)\b",
        r"\bchi.?square\b", r"\bcorrelat(ed|ion) between\b",
    ],
    TaskType.EXPLORATORY: [
        r"\bexplore\b", r"\bunderstand\b", r"\bwhat('| i)s in\b", r"\bhave a look\b",
        r"\banalyse this\b", r"\banalyze this\b", r"\boverview\b", r"\bsummar(y|ise|ize)\b",
    ],
}

_HORIZON_PATTERN = re.compile(
    r"\b(?:next|coming|following)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)?\s*"
    r"(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}
# Allows "five segments", "five customer segments", "5 distinct customer groups".
_K_PATTERN = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}(?:segments?|clusters?|groups?|personas?)\b",
    re.IGNORECASE,
)


def detect_objectives(
    profile: DatasetProfile,
    context: BusinessContext | None = None,
    max_objectives: int = 5,
) -> list[Objective]:
    """Propose ranked analytical objectives for this dataset.

    Returns several, not one: most datasets support more than one useful
    question, and forcing a single choice up front hides the alternatives.
    """
    context = context or BusinessContext()
    text = " ".join(
        filter(None, [
            context.stated_objective, context.business_problem, context.research_question,
            context.desired_outcome, context.description, context.analysis_style,
        ])
    ).lower()

    stated = _objectives_from_text(text, profile, context)
    inferred = _objectives_from_data(profile, context)

    merged: dict[str, Objective] = {}
    for objective in stated + inferred:
        key = f"{objective.task_type.value}:{objective.target or ''}"
        existing = merged.get(key)
        if existing is None or objective.priority > existing.priority:
            merged[key] = objective
        elif existing.source == "ai_detected" and objective.source == "user_supplied":
            objective.priority = max(objective.priority, existing.priority)
            merged[key] = objective

    ordered = sorted(merged.values(), key=lambda o: o.priority, reverse=True)
    if not ordered:
        ordered = [
            Objective(
                task_type=TaskType.EXPLORATORY,
                rationale="No clear objective was stated and the data does not point strongly to one, "
                          "so start by exploring what is there.",
                confidence=Confidence.LOW,
                priority=0.3,
            )
        ]
    return ordered[:max_objectives]


def _objectives_from_text(text: str, profile: DatasetProfile, context: BusinessContext) -> list[Objective]:
    if not text.strip() and not context.target_variable:
        return []
    out: list[Objective] = []
    matched: dict[TaskType, list[str]] = {}
    for task, patterns in INTENT_PATTERNS.items():
        hits = [p for p in patterns if re.search(p, text)]
        if hits:
            matched[task] = hits

    explicit_target = context.target_variable if context.target_variable in profile.columns else None
    mentioned = _columns_mentioned(text, profile)

    for task, hits in matched.items():
        target = explicit_target
        if target is None and task.is_supervised:
            target = _best_target_for_task(profile, task, mentioned)
        if task.is_supervised and target is None:
            continue
        if task is TaskType.TIME_SERIES_FORECAST and not profile.datetime_columns:
            continue

        objective = Objective(
            task_type=_refine_task_for_target(task, profile, target),
            target=target,
            time_column=profile.datetime_columns[0] if profile.datetime_columns else None,
            rationale=f"You asked for this: your description matches {len(hits)} phrase(s) "
                      f"indicating {task.value.replace('_', ' ')}.",
            confidence=Confidence.HIGH if explicit_target or len(hits) > 1 else Confidence.MODERATE,
            source="user_supplied",
            priority=(0.35 if task is TaskType.EXPLORATORY else 0.9 + 0.05 * min(len(hits), 2)),
        )
        if objective.task_type is TaskType.TIME_SERIES_FORECAST:
            objective.horizon = _extract_horizon(text)
        if objective.task_type is TaskType.CLUSTERING:
            objective.n_clusters = _extract_k(text)
        out.append(objective)

    if explicit_target and not any(o.target == explicit_target for o in out):
        col = profile.columns[explicit_target]
        out.append(
            Objective(
                task_type=_task_for_column(col),
                target=explicit_target,
                rationale=f"You nominated '{explicit_target}' as the variable to model.",
                confidence=Confidence.HIGH,
                source="user_supplied",
                priority=0.95,
            )
        )
    return out


def _objectives_from_data(profile: DatasetProfile, context: BusinessContext) -> list[Objective]:
    """What the data itself supports, regardless of what anybody asked for."""
    out: list[Objective] = []
    excluded = set(context.exclude_variables)

    for candidate in profile.target_candidates[:3]:
        name = candidate["column"]
        if name in excluded:
            continue
        col = profile.columns[name]
        out.append(
            Objective(
                task_type=_task_for_column(col),
                target=name,
                rationale=(
                    f"'{name}' looks like an outcome variable: "
                    + "; ".join(candidate["reasons"][:3]) + "."
                ),
                confidence=Confidence.MODERATE if candidate["score"] > 0.5 else Confidence.LOW,
                source="ai_detected",
                priority=0.4 + 0.3 * candidate["score"],
            )
        )

    numeric_and_categorical = len(profile.numeric_columns) + len(profile.categorical_columns)
    if numeric_and_categorical >= 3 and profile.n_rows >= 50:
        out.append(
            Objective(
                task_type=TaskType.CLUSTERING,
                features=profile.modelling_columns,
                rationale=(
                    f"{numeric_and_categorical} usable variables across {profile.n_rows} rows is enough "
                    "to look for natural groupings, whether or not a target exists."
                ),
                confidence=Confidence.MODERATE,
                source="ai_detected",
                priority=0.45,
            )
        )

    if profile.datetime_columns and profile.numeric_columns and profile.n_rows >= 30:
        time_col = profile.datetime_columns[0]
        freq = profile.columns[time_col].inferred_frequency or "irregular"
        numeric_target = _most_variable_numeric(profile)
        out.append(
            Objective(
                task_type=TaskType.TIME_SERIES_FORECAST,
                target=numeric_target,
                time_column=time_col,
                rationale=(
                    f"'{time_col}' gives the data a time dimension ({freq} spacing), so values "
                    "can be projected forward."
                ),
                confidence=Confidence.MODERATE,
                source="ai_detected",
                priority=0.5,
            )
        )

    if len(profile.numeric_columns) >= 3 and profile.n_rows >= 50:
        out.append(
            Objective(
                task_type=TaskType.ANOMALY_DETECTION,
                features=profile.numeric_columns,
                rationale="There are enough numeric variables to detect rows that behave unlike the rest.",
                confidence=Confidence.LOW,
                source="ai_detected",
                priority=0.3,
            )
        )

    if len(profile.numeric_columns) >= 8:
        out.append(
            Objective(
                task_type=TaskType.DIMENSIONALITY_REDUCTION,
                features=profile.numeric_columns,
                rationale=f"{len(profile.numeric_columns)} numeric variables is enough that a compressed "
                          "view may be easier to reason about than the raw columns.",
                confidence=Confidence.LOW,
                source="ai_detected",
                priority=0.25,
            )
        )

    out.append(
        Objective(
            task_type=TaskType.EXPLORATORY,
            rationale="Always available: describe the data, its distributions and its relationships.",
            confidence=Confidence.HIGH,
            source="ai_detected",
            priority=0.2,
        )
    )
    return out


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _task_for_column(col) -> TaskType:
    if col.semantic_type is SemanticType.BINARY:
        return TaskType.BINARY_CLASSIFICATION
    if col.semantic_type in {SemanticType.CATEGORICAL_NOMINAL, SemanticType.CATEGORICAL_ORDINAL}:
        return TaskType.MULTICLASS_CLASSIFICATION
    if col.semantic_type is SemanticType.NUMERIC_DISCRETE and col.n_unique <= 10:
        return TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _refine_task_for_target(task: TaskType, profile: DatasetProfile, target: str | None) -> TaskType:
    """Correct a stated task that the target column cannot actually support."""
    if target is None or target not in profile.columns:
        return task
    if not task.is_supervised or task is TaskType.TIME_SERIES_FORECAST:
        return task
    implied = _task_for_column(profile.columns[target])
    if task.is_classification and implied is TaskType.REGRESSION:
        return TaskType.REGRESSION
    if task is TaskType.REGRESSION and implied.is_classification:
        return implied
    if task.is_classification:
        return implied
    return task


def _best_target_for_task(profile: DatasetProfile, task: TaskType, mentioned: list[str]) -> str | None:
    for name in mentioned:
        col = profile.columns.get(name)
        if col is None:
            continue
        if task is TaskType.REGRESSION and col.is_numeric:
            return name
        if task.is_classification and (col.is_categorical or col.n_unique <= 20):
            return name
        if task is TaskType.TIME_SERIES_FORECAST and col.is_numeric:
            return name
    for candidate in profile.target_candidates:
        implied = candidate["implied_task"]
        if task is TaskType.REGRESSION and implied == "regression":
            return candidate["column"]
        if task.is_classification and implied in {"binary_classification", "multiclass_classification"}:
            return candidate["column"]
        if task is TaskType.TIME_SERIES_FORECAST and implied == "regression":
            return candidate["column"]
    return None


def _columns_mentioned(text: str, profile: DatasetProfile) -> list[str]:
    """Column names the user referred to, longest first so 'annual spend' beats 'spend'."""
    hits = []
    for name in sorted(profile.columns, key=len, reverse=True):
        readable = name.lower().replace("_", " ")
        if readable and (readable in text or name.lower() in text):
            hits.append(name)
    return hits


def _most_variable_numeric(profile: DatasetProfile) -> str | None:
    best, best_cv = None, -1.0
    for name in profile.numeric_columns:
        col = profile.columns[name]
        if not col.mean or not col.std:
            continue
        cv = abs(col.std / col.mean) if col.mean else 0.0
        if 0 < cv < 10 and cv > best_cv:
            best, best_cv = name, cv
    return best or (profile.numeric_columns[0] if profile.numeric_columns else None)


def _extract_horizon(text: str) -> int | None:
    match = _HORIZON_PATTERN.search(text)
    if not match:
        return None
    raw, unit = match.group(1), match.group(2).lower()
    if raw is None:
        count = 1
    elif raw.isdigit():
        count = int(raw)
    else:
        count = _NUMBER_WORDS.get(raw, 1)
    return count  # in the units of the series' own frequency


def _extract_k(text: str) -> int | None:
    match = _K_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw.lower())


def describe_objectives(objectives: list[Objective]) -> str:
    lines = []
    for i, o in enumerate(objectives, 1):
        source = "you asked for this" if o.source.startswith("user") else "detected from the data"
        lines.append(f"{i}. {o.label()} — {o.rationale} ({source}, confidence: {o.confidence.value})")
    return "\n".join(lines)
