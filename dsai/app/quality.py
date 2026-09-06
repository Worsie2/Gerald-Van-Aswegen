"""Data quality, broken into the dimensions a reader can act on.

The single score on ``DatasetProfile`` is deliberately blunt — it flags, it does
not judge. "88 out of 100" tells a reader nothing about *what* to fix, so this
splits the same evidence into dimensions and computes each from the profile it
already carries. Nothing new is measured here; this is a second reading of the
measurements already taken.
"""

from __future__ import annotations

from typing import Any

__all__ = ["quality_breakdown", "QUALITY_DIMENSIONS"]

#: Which issue codes count against which dimension, and what to say when they do.
QUALITY_DIMENSIONS: list[tuple[str, tuple[str, ...], str]] = [
    ("Completeness", ("severe_missingness", "missing_values"),
     "How much of the data is actually there."),
    ("Uniqueness", ("duplicate_rows", "identifier_columns"),
     "Whether rows repeat, and whether a column is really just a row label."),
    ("Consistency", ("constant_columns", "near_constant_columns", "high_cardinality",
                     "mixed_types"),
     "Whether each column carries the kind of variation a variable should."),
    ("Distribution", ("outliers", "skewed_distributions"),
     "Whether values are spread in a way models can work with."),
    ("Independence", ("high_correlation", "multicollinearity", "possible_leakage"),
     "Whether variables duplicate each other, or give away the answer."),
    ("Sufficiency", ("small_sample", "wide_data"),
     "Whether there is enough data to support a conclusion."),
]

#: What one issue of each severity costs its dimension.
_PENALTY = {"critical": 45.0, "warning": 18.0, "info": 5.0}


def quality_breakdown(profile: Any) -> list[tuple[str, float, str]]:
    """``(dimension, score out of 100, note)`` for each quality dimension.

    A dimension with nothing against it scores 100 — which is a statement that
    nothing was *detected*, not a guarantee that nothing is wrong, and the note
    says so where it matters.
    """
    by_code: dict[str, list[Any]] = {}
    for issue in getattr(profile, "quality_issues", []):
        by_code.setdefault(issue.code, []).append(issue)

    rows: list[tuple[str, float, str]] = []
    for name, codes, description in QUALITY_DIMENSIONS:
        hits = [issue for code in codes for issue in by_code.get(code, [])]
        penalty = sum(_PENALTY.get(issue.severity, 5.0) for issue in hits)
        score = max(0.0, 100.0 - penalty)
        if hits:
            worst = min(hits, key=lambda i: ["critical", "warning", "info"].index(i.severity)
                        if i.severity in ("critical", "warning", "info") else 3)
            note = worst.message
        else:
            note = f"Nothing detected. {description}"
        rows.append((name, score, note))
    return rows
