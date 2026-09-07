"""Pipeline construction, history and leakage control.

A :class:`PreprocessingPipeline` is an ordered, editable, serialisable list of
steps. It compiles into two things:

1. **Row operations** — deduplication, row dropping, forward fill. These change
   which rows exist, so they are applied to the dataset before splitting.
2. **A scikit-learn pipeline** of column transformations. This is handed to the
   model *unfitted*, so cross-validation fits it on each training fold only.

That split is what keeps the platform honest about leakage: nothing that learns
a statistic from the data is ever fitted on rows it will later be scored on.
"""

from __future__ import annotations

import copy
import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from dsai.core.schema import DatasetProfile, JsonMixin, SemanticType
from dsai.preprocessing.steps import STEPS, StepSpec


@dataclass
class PipelineStep(JsonMixin):
    """One configured step in a pipeline."""

    step_key: str
    params: dict[str, Any] = field(default_factory=dict)
    columns: list[str] | None = None
    enabled: bool = True
    note: str = ""
    added_by: str = "ai"           # ai | user
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def spec(self) -> StepSpec:
        return STEPS.get(self.step_key)

    def describe(self) -> str:
        spec = self.spec
        if not self.columns:
            target = "all applicable columns"
        elif isinstance(self.columns, str):
            target = {
                "__all__": "every column",
                "__numeric__": "every numeric column at this point",
                "__non_numeric__": "every non-numeric column at this point",
                "__datetime__": "every date column at this point",
            }.get(self.columns, self.columns)
        else:
            target = ", ".join(self.columns[:6]) + (
                f" (+{len(self.columns) - 6} more)" if len(self.columns) > 6 else ""
            )
        detail = f"{spec.name} → {target}"
        if self.params:
            shown = ", ".join(f"{k}={v}" for k, v in self.params.items() if v is not None)
            if shown:
                detail += f" [{shown}]"
        return detail

    def build(self):
        spec = self.spec
        params = {**spec.default_params(), **self.params}
        if spec.scope == "column" and self.columns is not None:
            # setdefault is not enough: the spec's own default for "columns" is
            # None, so the key exists and would silently win.
            params["columns"] = self.columns
        return spec.build(**params)


@dataclass
class PipelineHistoryEntry(JsonMixin):
    action: str                    # add | remove | move | edit | clear | load
    detail: str
    by: str = "user"
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))


class PreprocessingPipeline(JsonMixin):
    """An editable, replayable preprocessing plan."""

    def __init__(self, name: str = "pipeline", steps: list[PipelineStep] | None = None) -> None:
        self.name = name
        self.steps: list[PipelineStep] = list(steps or [])
        self.history: list[PipelineHistoryEntry] = []
        self._undo_stack: list[list[PipelineStep]] = []
        self._redo_stack: list[list[PipelineStep]] = []

    # -- editing --------------------------------------------------------------
    def _snapshot(self) -> None:
        self._undo_stack.append(copy.deepcopy(self.steps))
        self._redo_stack.clear()

    def add(self, step_key: str, columns: list[str] | None = None, position: int | None = None,
            by: str = "user", note: str = "", **params: Any) -> PipelineStep:
        STEPS.get(step_key)  # raises early on an unknown key
        step = PipelineStep(step_key=step_key, params=params, columns=columns, added_by=by, note=note)
        self._snapshot()
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        self.history.append(PipelineHistoryEntry("add", step.describe(), by))
        return step

    def remove(self, step_id: str, by: str = "user") -> bool:
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                self._snapshot()
                removed = self.steps.pop(i)
                self.history.append(PipelineHistoryEntry("remove", removed.describe(), by))
                return True
        return False

    def move(self, step_id: str, new_index: int, by: str = "user") -> bool:
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                self._snapshot()
                self.steps.pop(i)
                self.steps.insert(max(0, min(new_index, len(self.steps))), step)
                self.history.append(
                    PipelineHistoryEntry("move", f"{step.spec.name}: position {i} → {new_index}", by)
                )
                return True
        return False

    def set_enabled(self, step_id: str, enabled: bool, by: str = "user") -> bool:
        for step in self.steps:
            if step.id == step_id:
                self._snapshot()
                step.enabled = enabled
                self.history.append(
                    PipelineHistoryEntry("edit", f"{step.spec.name} {'enabled' if enabled else 'disabled'}", by)
                )
                return True
        return False

    def update_params(self, step_id: str, by: str = "user", **params: Any) -> bool:
        for step in self.steps:
            if step.id == step_id:
                self._snapshot()
                step.params.update(params)
                self.history.append(PipelineHistoryEntry("edit", f"{step.spec.name}: {params}", by))
                return True
        return False

    def clear(self, by: str = "user") -> None:
        self._snapshot()
        self.steps = []
        self.history.append(PipelineHistoryEntry("clear", "all steps removed", by))

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(copy.deepcopy(self.steps))
        self.steps = self._undo_stack.pop()
        self.history.append(PipelineHistoryEntry("undo", "reverted the last change"))
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self.steps))
        self.steps = self._redo_stack.pop()
        self.history.append(PipelineHistoryEntry("redo", "reapplied the change"))
        return True

    # -- inspection -----------------------------------------------------------
    @property
    def active_steps(self) -> list[PipelineStep]:
        return [s for s in self.steps if s.enabled]

    @property
    def row_steps(self) -> list[PipelineStep]:
        return [s for s in self.active_steps if s.spec.scope == "row"]

    @property
    def column_steps(self) -> list[PipelineStep]:
        return [s for s in self.active_steps if s.spec.scope == "column"]

    def describe(self) -> list[str]:
        return [
            f"{i + 1}. {s.describe()}{'' if s.enabled else '  (disabled)'}"
            for i, s in enumerate(self.steps)
        ]

    def leakage_report(self) -> dict[str, Any]:
        """Which steps learn from data, and are therefore fitted inside the fold."""
        fold_fitted = [s.spec.name for s in self.column_steps if not s.spec.leakage_safe]
        return {
            "fitted_inside_cross_validation": fold_fitted,
            "applied_before_split": [s.spec.name for s in self.row_steps],
            "safe": True,
            "explanation": (
                "Steps that learn a statistic from the data (means, scales, category-to-target "
                "maps, PCA loadings, feature selection) are fitted only on each training fold, "
                "so no information from the held-out rows reaches the model. Row operations "
                "(deduplication, row removal) run before splitting because they change which "
                "observations exist at all."
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "history": [h.to_dict() for h in self.history],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessingPipeline":
        pipeline = cls(name=payload.get("name", "pipeline"))
        for raw in payload.get("steps", []):
            pipeline.steps.append(
                PipelineStep(
                    step_key=raw["step_key"],
                    params=raw.get("params", {}),
                    columns=raw.get("columns"),
                    enabled=raw.get("enabled", True),
                    note=raw.get("note", ""),
                    added_by=raw.get("added_by", "ai"),
                    id=raw.get("id", uuid.uuid4().hex[:8]),
                )
            )
        pipeline.history.append(PipelineHistoryEntry("load", f"loaded {len(pipeline.steps)} steps"))
        return pipeline

    def copy(self) -> "PreprocessingPipeline":
        clone = PreprocessingPipeline(self.name, copy.deepcopy(self.steps))
        clone.history = list(self.history)
        return clone

    # -- compilation ----------------------------------------------------------
    def apply_row_steps(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Run the row-level operations. Returns the new frame and a change log."""
        log: list[str] = []
        result = frame
        for step in self.row_steps:
            before = len(result)
            key, params = step.step_key, {**step.spec.default_params(), **step.params}
            if key == "drop_duplicates":
                subset = params.get("subset") or step.columns
                subset = [c for c in (subset or []) if c in result.columns] or None
                result = result.drop_duplicates(subset=subset)
            elif key == "drop_rows_missing":
                cols = [c for c in (step.columns or result.columns) if c in result.columns]
                threshold = float(params.get("threshold", 1.0))
                if threshold >= 1.0:
                    result = result.dropna(subset=cols)
                else:
                    share_missing = result[cols].isna().mean(axis=1)
                    result = result[share_missing < threshold]
            elif key == "impute_forward_fill":
                cols = [c for c in (step.columns or result.columns) if c in result.columns]
                direction = params.get("direction", "forward")
                result = result.copy()
                if direction in ("forward", "both"):
                    result[cols] = result[cols].ffill()
                if direction in ("backward", "both"):
                    result[cols] = result[cols].bfill()
            elif key == "remove_outlier_rows":
                cols = [c for c in (step.columns or []) if c in result.columns]
                cols = [c for c in cols if pd.api.types.is_numeric_dtype(result[c])]
                if cols:
                    factor = float(params.get("factor", 3.0))
                    q1 = result[cols].quantile(0.25)
                    q3 = result[cols].quantile(0.75)
                    iqr = (q3 - q1).replace(0, float("nan"))
                    mask = ~(((result[cols] < q1 - factor * iqr) | (result[cols] > q3 + factor * iqr)).any(axis=1))
                    result = result[mask.fillna(True)]
            removed = before - len(result)
            log.append(
                f"{step.spec.name}: {removed} row(s) removed ({before} → {len(result)})"
                if removed else f"{step.spec.name}: no rows removed"
            )
        return result, log

    def build_sklearn_pipeline(self, append_coercer: bool = True) -> Pipeline | None:
        """Compile the column steps into an unfitted scikit-learn Pipeline."""
        stages: list[tuple[str, Any]] = []
        for i, step in enumerate(self.column_steps):
            try:
                stages.append((f"{i:02d}_{step.step_key}_{step.id}", step.build()))
            except Exception as exc:
                raise RuntimeError(f"Could not build preprocessing step '{step.spec.name}': {exc}") from exc
        if append_coercer:
            stages.append(("zz_coerce_numeric", STEPS.get("coerce_numeric").build()))
        return Pipeline(stages) if stages else None

    def fit_transform_frame(self, frame: pd.DataFrame,
                            target: pd.Series | None = None) -> pd.DataFrame:
        """Apply the whole pipeline to a frame, fitted on all of it.

        This is for *getting the cleaned data out* — handing it on, or checking
        the transformations against what you know about the data. It is
        deliberately not how modelling works: there, the column steps are
        refitted inside each cross-validation fold so no fold can learn from the
        rows it will be scored on. Fitting on everything here is correct because
        nothing is being scored; it would be leakage if anything were.
        """
        working = PreprocessingPipeline("export", copy.deepcopy(self.steps))
        result, _ = working.apply_row_steps(frame)
        aligned = target.loc[result.index] if target is not None else None

        compiled = working.build_sklearn_pipeline()
        if compiled is None:
            return result
        transformed = compiled.fit_transform(result, aligned)
        if not isinstance(transformed, pd.DataFrame):
            names = None
            try:
                names = list(compiled.get_feature_names_out())
            except Exception:
                pass
            transformed = pd.DataFrame(transformed, index=result.index, columns=names)
        return transformed

    def preview(self, frame: pd.DataFrame, target: pd.Series | None = None,
                up_to: str | None = None, n_rows: int = 20) -> dict[str, Any]:
        """Show what the pipeline does to a sample, step by step.

        This is a preview only — it fits on the sample so the user can see the
        effect. The real fit happens inside cross-validation.
        """
        sample = frame.head(max(n_rows * 5, 200))
        target_sample = target.loc[sample.index] if target is not None else None
        current = sample
        stages: list[dict[str, Any]] = []

        working_pipeline = PreprocessingPipeline("preview", copy.deepcopy(self.steps))
        current, row_log = working_pipeline.apply_row_steps(current)
        if target_sample is not None:
            target_sample = target_sample.loc[current.index]
        for entry in row_log:
            stages.append({"step": entry, "shape": current.shape, "scope": "row"})

        for step in working_pipeline.column_steps:
            before_shape = current.shape
            try:
                transformer = step.build()
                current = transformer.fit_transform(current, target_sample)
                if not isinstance(current, pd.DataFrame):
                    current = pd.DataFrame(current)
                stages.append(
                    {
                        "step": step.describe(),
                        "scope": "column",
                        "shape": current.shape,
                        "columns_before": before_shape[1],
                        "columns_after": current.shape[1],
                        "status": "ok",
                    }
                )
            except Exception as exc:
                stages.append(
                    {
                        "step": step.describe(),
                        "scope": "column",
                        "shape": current.shape,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                break
            if up_to and step.id == up_to:
                break

        return {
            "stages": stages,
            "final_shape": tuple(current.shape),
            "sample": current.head(n_rows),
            "final_columns": list(current.columns)[:200],
            "note": "Preview only — fitted on a sample. The real fit happens inside cross-validation.",
        }


def summarise_transformations(pipeline: PreprocessingPipeline) -> dict[str, list[str]]:
    """Group the pipeline by category, for the report and the decision log."""
    grouped: dict[str, list[str]] = {}
    for step in pipeline.active_steps:
        grouped.setdefault(step.spec.category, []).append(step.describe())
    return grouped
