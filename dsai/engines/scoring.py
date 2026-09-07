"""Apply a model that has already been trained to data it has never seen.

Training a model and never using it is the most common way an analysis stops
short of being worth anything. This is the other half: take the fitted pipeline
from a completed run, point it at a new file, and get predictions back.

The hard part is not the prediction. It is everything that has to be true before
the prediction means anything:

* the new data must carry the columns the model was fitted on, under the same
  names, holding the same kinds of value;
* a column the model never saw is harmless and a column it needs and cannot find
  is fatal, and those two have to be told apart plainly; and
* new data that no longer resembles the training data will still produce
  confident numbers, which is exactly when a prediction is most dangerous — so
  the drift check runs whether or not anyone asked for it.

Nothing here refits anything. The transformations applied are the ones learned
during training, which is the only way a prediction on new data is comparable to
the scores the model was judged on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import TaskType

__all__ = ["SchemaCheck", "ScoringResult", "check_schema", "score_new_data", "drift_report"]


@dataclass
class SchemaCheck:
    """Whether this data can be scored by this model, and what is wrong if not."""

    ok: bool = True
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    type_changed: list[dict[str, str]] = field(default_factory=list)
    unseen_categories: dict[str, list[str]] = field(default_factory=dict)
    new_missingness: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok and not self.notes:
            return "The new data matches what the model was trained on."
        if not self.ok:
            return (f"{len(self.missing)} required column(s) are absent, so this data cannot be "
                    "scored by this model.")
        return f"The data can be scored, with {len(self.notes)} thing(s) worth knowing first."


@dataclass
class ScoringResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    schema: SchemaCheck = field(default_factory=SchemaCheck)
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_scored: int = 0
    n_skipped: int = 0
    model_name: str = ""
    caveats: list[str] = field(default_factory=list)


def _is_identifier_like(series: pd.Series) -> bool:
    """A label column whose values are mostly unique — a customer ID, an order number.

    Numeric columns are never identifiers by this test even though a continuous
    one has a distinct value in almost every row: uniqueness is what continuous
    *means*, and excluding those would throw away the columns drift most needs
    to watch.
    """
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return non_null.nunique() / len(non_null) > 0.9


def check_schema(
    new_frame: pd.DataFrame,
    training_frame: pd.DataFrame | None,
    required: list[str],
    target: str | None = None,
) -> SchemaCheck:
    """Compare the new data against what the model expects, before scoring anything."""
    check = SchemaCheck()
    present = set(new_frame.columns)
    needed = [c for c in required if c != target]

    check.missing = [c for c in needed if c not in present]
    check.extra = sorted(present - set(needed) - ({target} if target else set()))
    check.ok = not check.missing

    if check.missing:
        check.notes.append(
            f"Absent and required: {', '.join(f'`{c}`' for c in check.missing[:10])}. "
            "The model cannot make a prediction without them."
        )
    if check.extra:
        check.notes.append(
            f"{len(check.extra)} column(s) in the new data were not part of the model and are "
            "ignored: " + ", ".join(f"`{c}`" for c in check.extra[:8])
            + (" …" if len(check.extra) > 8 else "")
        )

    if training_frame is None:
        return check

    for column in needed:
        if column not in present or column not in training_frame.columns:
            continue
        old_numeric = pd.api.types.is_numeric_dtype(training_frame[column])
        new_numeric = pd.api.types.is_numeric_dtype(new_frame[column])
        if old_numeric != new_numeric:
            check.type_changed.append({
                "column": column,
                "was": "numeric" if old_numeric else "text or category",
                "now": "numeric" if new_numeric else "text or category",
            })
            continue

        if not new_numeric and not _is_identifier_like(training_frame[column]):
            # Every value in an identifier column is unseen by definition.
            # Reporting that as a warning trains the reader to ignore the
            # warning that matters.
            seen = set(training_frame[column].astype("string").dropna().unique())
            arriving = set(new_frame[column].astype("string").dropna().unique())
            unseen = sorted(arriving - seen)
            if unseen:
                check.unseen_categories[column] = unseen[:20]

        old_gaps = float(training_frame[column].isna().mean())
        new_gaps = float(new_frame[column].isna().mean())
        if new_gaps > max(0.05, old_gaps * 2):
            check.new_missingness[column] = round(new_gaps, 4)

    if check.type_changed:
        check.notes.append(
            "Column(s) changed kind between training and now: "
            + "; ".join(f"`{c['column']}` was {c['was']}, is now {c['now']}" for c in check.type_changed[:5])
            + ". Predictions on these will be unreliable at best."
        )
    if check.unseen_categories:
        total = sum(len(v) for v in check.unseen_categories.values())
        check.notes.append(
            f"{total} category value(s) appear that the model never saw during training, across "
            f"{len(check.unseen_categories)} column(s). Encoders treat these as unknown; the model "
            "has learned nothing about them."
        )
    if check.new_missingness:
        check.notes.append(
            "Missingness has risen sharply in "
            + ", ".join(f"`{c}` ({v:.0%})" for c, v in list(check.new_missingness.items())[:5])
            + ". Imputation will fill those gaps with values learned from the training data, which "
            "is a bigger assumption the more of them there are."
        )
    return check


#: Below this many new rows, a difference in the mix of a categorical column is
#: as likely to be sampling noise as a real change, so the report says so rather
#: than presenting chance as drift.
MIN_ROWS_FOR_DRIFT = 30


def drift_report(new_frame: pd.DataFrame, training_frame: pd.DataFrame,
                 columns: list[str]) -> pd.DataFrame:
    """How far each column has moved since training, in terms a reader can act on.

    Deliberately simple statistics rather than a formal test: the question here
    is "has this changed enough to worry about", and a standardised difference in
    means answers it more usefully than a p-value that will be significant on any
    large enough sample.

    Identifiers and timestamps are skipped. Every customer ID in a new file is
    new, and every date in it is later — reporting that as drift buries the
    columns where a change would actually mean something.
    """
    small = len(new_frame) < MIN_ROWS_FOR_DRIFT
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in new_frame.columns or column not in training_frame.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(training_frame[column]):
            continue
        if _is_identifier_like(training_frame[column]):
            continue
        old, new = training_frame[column], new_frame[column]
        if pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new):
            old_clean, new_clean = old.dropna(), new.dropna()
            if old_clean.empty or new_clean.empty:
                continue
            spread = old_clean.std()
            shift = (new_clean.mean() - old_clean.mean()) / spread if spread > 1e-12 else 0.0
            rows.append({
                "Column": column,
                "Training mean": round(float(old_clean.mean()), 4),
                "New mean": round(float(new_clean.mean()), 4),
                "Shift (std devs)": round(float(shift), 3),
                "Reading": ("too few new rows to tell" if small
                            else _read_shift(abs(shift))),
            })
        else:
            old_share = old.astype("string").value_counts(normalize=True)
            new_share = new.astype("string").value_counts(normalize=True)
            combined = old_share.index.union(new_share.index)
            distance = float(
                (old_share.reindex(combined).fillna(0) - new_share.reindex(combined).fillna(0))
                .abs().sum() / 2
            )
            rows.append({
                "Column": column,
                "Training mean": "—",
                "New mean": "—",
                "Shift (std devs)": round(distance, 3),
                "Reading": ("too few new rows to tell" if small
                            else _read_mix(distance)),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.reindex(
            frame["Shift (std devs)"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    return frame


def _read_shift(shift: float) -> str:
    if shift < 0.1:
        return "essentially unchanged"
    if shift < 0.3:
        return "slightly shifted"
    if shift < 0.8:
        return "noticeably different — worth checking why"
    return "very different from training data — treat predictions with suspicion"


def _read_mix(distance: float) -> str:
    if distance < 0.05:
        return "essentially unchanged"
    if distance < 0.15:
        return "the mix has shifted a little"
    if distance < 0.35:
        return "the mix has changed materially"
    return "a different population — treat predictions with suspicion"


def score_new_data(
    model: Any,
    new_frame: pd.DataFrame,
    result: Any,
    training_frame: pd.DataFrame | None = None,
    keep_columns: list[str] | None = None,
) -> ScoringResult:
    """Predict on *new_frame* with an already-fitted model.

    ``model`` is whatever :meth:`AIDataScientist.fitted_model` returned — a
    scikit-learn Pipeline for a supervised run, or a ``(preprocessor, estimator)``
    pair for clustering and anomaly detection.
    """
    out = ScoringResult(model_name=result.model_name)
    required = list(result.features or [])
    target = result.target
    out.schema = check_schema(new_frame, training_frame, required, target)
    if not out.schema.ok:
        return out

    usable = new_frame.reindex(columns=[c for c in required if c != target])
    # Rows with nothing in them cannot be scored by anything, and quietly
    # dropping them would leave the caller unable to line predictions back up
    # against their own file.
    empty = usable.isna().all(axis=1)
    out.n_skipped = int(empty.sum())
    scored_index = usable.index[~empty]
    usable = usable.loc[scored_index]
    if usable.empty:
        out.caveats.append("Every row was empty across the columns the model needs.")
        return out

    predictions = pd.DataFrame(index=scored_index)
    if keep_columns:
        for column in keep_columns:
            if column in new_frame.columns:
                predictions[column] = new_frame.loc[scored_index, column]

    task = result.task_type
    if isinstance(model, tuple):
        preprocessor, estimator = model
        matrix = preprocessor.transform(usable) if preprocessor is not None else usable
        matrix = np.asarray(matrix, dtype=float)
        if task is TaskType.ANOMALY_DETECTION:
            flags = estimator.predict(matrix)
            predictions["anomaly"] = ["yes" if int(f) == -1 else "no" for f in flags]
            if hasattr(estimator, "score_samples"):
                predictions["anomaly_score"] = np.asarray(estimator.score_samples(matrix), dtype=float)
            out.caveats.append(
                "An anomaly flag is a statement about how unusual a row is relative to the "
                "training data, not a statement that anything is wrong with it."
            )
        else:
            predictions["segment"] = np.asarray(estimator.predict(matrix)).astype(int)
            out.caveats.append(
                "Segments are assigned by nearest centre. A row unlike anything in the training "
                "data is still assigned to its closest segment, however far away that is."
            )
    else:
        predicted = model.predict(usable)
        column_name = f"predicted_{target}" if target else "prediction"
        predictions[column_name] = np.ravel(predicted)

        if task.is_classification and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(usable)
                classes = list(getattr(model, "classes_", range(proba.shape[1])))
                predictions["confidence"] = np.max(proba, axis=1).round(4)
                for index, label in enumerate(classes):
                    predictions[f"probability_{label}"] = proba[:, index].round(4)
                out.caveats.append(
                    "A probability is only as trustworthy as the model's calibration. Check the "
                    "calibration table on the Models page before treating one as a real likelihood."
                )
            except Exception:
                pass

    out.predictions = predictions
    out.n_scored = len(predictions)

    if training_frame is not None:
        comparable = [c for c in required if c != target]
        out.drift = drift_report(new_frame, training_frame, comparable)
        worrying = out.drift[out.drift["Reading"].str.contains("suspicion|materially", na=False)] \
            if not out.drift.empty else out.drift
        if len(worrying):
            out.caveats.append(
                f"{len(worrying)} column(s) have moved substantially since training. A model "
                "predicts confidently on data it does not recognise, so that is precisely when "
                "these numbers deserve least trust."
            )

    if out.n_skipped:
        out.caveats.append(
            f"{out.n_skipped} row(s) were empty across every column the model needs and were not "
            "scored. They are absent from the output rather than filled with a guess."
        )
    return out
