"""Reproducibility: record enough that any result can be reconstructed.

An analysis nobody can re-run is an anecdote. Every run captures its dataset
fingerprint, preprocessing steps, model, hyper-parameters, seed, split strategy,
metrics, library versions, the platform's decisions and the user's overrides.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dsai.core.schema import JsonMixin


@dataclass
class RunManifest(JsonMixin):
    """The complete record of one analysis."""

    run_id: str = ""
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    dataset: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)
    preprocessing_spec: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    models_evaluated: list[dict[str, Any]] = field(default_factory=list)
    selected_model: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    ai_decisions: list[dict[str, Any]] = field(default_factory=list)
    user_overrides: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    dsai_version: str = "0.1.0"

    def fingerprint(self) -> str:
        """A short hash over the parts that determine the result.

        Two runs with the same fingerprint should produce the same numbers; if
        they do not, something outside this record changed.
        """
        payload = json.dumps(
            {
                "dataset_hash": self.dataset.get("content_hash"),
                "objective": self.objective,
                "preprocessing": self.preprocessing,
                "validation": self.validation,
                "seed": self.random_seed,
                "model": self.selected_model.get("key"),
                "hyperparameters": self.selected_model.get("hyperparameters"),
            },
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def render(self) -> str:
        """Human-readable manifest for the appendix of a report."""
        lines = [
            "REPRODUCIBILITY MANIFEST",
            "=" * 60,
            f"Run ID:        {self.run_id}",
            f"Fingerprint:   {self.fingerprint()}",
            f"Created:       {self.created_at}",
            f"Duration:      {self.duration_s}s",
            "",
            "DATASET",
            f"  Name:        {self.dataset.get('name', '—')}",
            f"  Source:      {self.dataset.get('kind', '—')} {self.dataset.get('path') or self.dataset.get('query') or ''}",
            f"  Shape:       {self.dataset.get('n_rows', '—')} rows × {self.dataset.get('n_columns', '—')} columns",
            f"  Content hash:{self.dataset.get('content_hash', '—')}",
            "",
            "OBJECTIVE",
            f"  Task:        {self.objective.get('task_type', '—')}",
            f"  Target:      {self.objective.get('target', '—')}",
            f"  Source:      {self.objective.get('source', '—')}",
            "",
            "PREPROCESSING",
        ]
        lines += [f"  {i + 1}. {step}" for i, step in enumerate(self.preprocessing)] or ["  (none)"]
        lines += [
            "",
            "VALIDATION",
            f"  Strategy:    {self.validation.get('strategy', '—')}",
            f"  Splits:      {self.validation.get('n_splits', '—')}",
            f"  Random seed: {self.random_seed}",
            "",
            "SELECTED MODEL",
            f"  {self.selected_model.get('name', '—')} ({self.selected_model.get('key', '—')})",
        ]
        for name, value in (self.selected_model.get("hyperparameters") or {}).items():
            lines.append(f"    {name} = {value}")
        lines += ["", "METRICS"]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value}")
        lines += ["", f"MODELS EVALUATED ({len(self.models_evaluated)})"]
        for model in self.models_evaluated:
            lines.append(f"  {model.get('name')}: {model.get('primary_score')} ({model.get('status')})")
        lines += ["", "ENVIRONMENT"]
        for name, value in self.environment.items():
            lines.append(f"  {name}: {value}")
        if self.user_overrides:
            lines += ["", "USER OVERRIDES"]
            for override in self.user_overrides:
                lines.append(f"  {override.get('stage')}: {override.get('detail')}")
        if self.warnings:
            lines += ["", "WARNINGS"]
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def build_manifest(run: Any) -> RunManifest:
    """Extract a manifest from a completed :class:`AnalysisRun`."""
    from dsai.engines import metrics as M

    manifest = RunManifest(run_id=run.id, duration_s=run.duration_s)
    manifest.created_at = run.created_at or manifest.created_at

    if run.source is not None:
        manifest.dataset = run.source.to_dict()
    manifest.dataset.setdefault("name", run.dataset_name)
    if run.profile is not None:
        manifest.dataset.setdefault("n_rows", run.profile.n_rows)
        manifest.dataset.setdefault("n_columns", run.profile.n_columns)
        manifest.dataset["quality_score"] = run.profile.quality_score
        manifest.dataset["profiled_at"] = run.profile.profiled_at

    manifest.context = run.context.to_dict() if run.context else {}
    if run.objective is not None:
        manifest.objective = run.objective.to_dict()
    if run.pipeline is not None:
        manifest.preprocessing = [s.describe() for s in run.pipeline.active_steps]
        manifest.preprocessing_spec = run.pipeline.to_dict()
    if run.plan is not None:
        manifest.validation = dict(run.plan.validation_strategy)
        manifest.validation["primary_metric"] = run.plan.primary_metric
    manifest.random_seed = run.settings.random_state
    manifest.environment = dict(run.environment)
    manifest.warnings = list(run.warnings)

    primary = run.plan.primary_metric if run.plan else ""
    manifest.models_evaluated = [
        {
            "key": r.model_key,
            "name": r.model_name,
            "status": r.status,
            "primary_metric": primary,
            "primary_score": r.primary(primary) if primary else None,
            "training_time_s": r.training_time_s,
            "hyperparameters": r.hyperparameters,
            "warnings": r.warnings,
        }
        for r in run.results
    ]
    if run.best is not None:
        manifest.selected_model = {
            "key": run.best.model_key,
            "name": run.best.model_name,
            "hyperparameters": run.best.hyperparameters,
            "n_train": run.best.n_train,
            "n_test": run.best.n_test,
            "features": run.best.features,
        }
        manifest.metrics = {
            "cross_validated": run.best.validation.mean_scores,
            "cross_validated_std": run.best.validation.std_scores,
            "hold_out_test": run.best.test_scores,
            "training": run.best.train_scores,
        }

    manifest.ai_decisions = [d.to_dict() for d in run.decisions]
    manifest.user_overrides = [d.to_dict() for d in run.decisions if d.overridden_by_user]
    return manifest


def environment_snapshot() -> dict[str, str]:
    """Library versions that materially affect results."""
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for module_name, key in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit_learn"),
        ("scipy", "scipy"), ("statsmodels", "statsmodels"), ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("mlxtend", "mlxtend"),
    ]:
        try:
            module = __import__(module_name)
            snapshot[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return snapshot


def save_manifest(manifest: RunManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> RunManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = RunManifest()
    for key, value in payload.items():
        if hasattr(manifest, key):
            setattr(manifest, key, value)
    return manifest
