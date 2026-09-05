"""Project persistence: save an analysis and come back to it later.

A project directory holds the dataset, the context, saved pipelines, the run
manifests and the fitted model. It is plain files — JSON and CSV/Parquet — so
it can be inspected, diffed and version-controlled rather than being an opaque
blob.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dsai.core.schema import BusinessContext, JsonMixin
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.repro.provenance import RunManifest, build_manifest, load_manifest

PROJECT_FILE = "project.json"
DATA_DIR = "data"
RUNS_DIR = "runs"
PIPELINES_DIR = "pipelines"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


@dataclass
class ProjectMeta(JsonMixin):
    name: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    dataset_file: str | None = None
    dataset_rows: int = 0
    dataset_columns: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)
    saved_pipelines: list[str] = field(default_factory=list)
    dsai_version: str = "0.1.0"


class Project:
    """A saved workspace: dataset, context, pipelines, runs and models."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta = ProjectMeta(name=self.path.name)
        self._frame: pd.DataFrame | None = None

    # -- lifecycle ------------------------------------------------------------
    @classmethod
    def create(
        cls,
        path: str | Path,
        frame: pd.DataFrame | None = None,
        name: str | None = None,
        description: str = "",
        context: BusinessContext | None = None,
    ) -> "Project":
        project = cls(path)
        project.path.mkdir(parents=True, exist_ok=True)
        for sub in (DATA_DIR, RUNS_DIR, PIPELINES_DIR, MODELS_DIR, REPORTS_DIR):
            (project.path / sub).mkdir(exist_ok=True)
        project.meta.name = name or project.path.name
        project.meta.description = description
        if context is not None:
            project.meta.context = context.to_dict()
        if frame is not None:
            project.set_data(frame)
        project.save()
        return project

    @classmethod
    def open(cls, path: str | Path) -> "Project":
        project = cls(path)
        meta_path = project.path / PROJECT_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No project found at {project.path}. Create one with Project.create()."
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if hasattr(project.meta, key):
                setattr(project.meta, key, value)
        return project

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        return (Path(path) / PROJECT_FILE).exists()

    def save(self) -> Path:
        self.meta.updated_at = _dt.datetime.now().isoformat(timespec="seconds")
        target = self.path / PROJECT_FILE
        target.write_text(json.dumps(self.meta.to_dict(), indent=2, default=str), encoding="utf-8")
        return target

    def delete(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    # -- data -----------------------------------------------------------------
    def set_data(self, frame: pd.DataFrame, filename: str = "dataset.parquet") -> Path:
        target = self.path / DATA_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(target, index=False)
        except Exception:
            # Parquet needs pyarrow and cannot hold every dtype; CSV always works.
            target = target.with_suffix(".csv")
            frame.to_csv(target, index=False)
        self.meta.dataset_file = target.name
        self.meta.dataset_rows = len(frame)
        self.meta.dataset_columns = int(frame.shape[1])
        self._frame = frame
        self.save()
        return target

    def data(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        if not self.meta.dataset_file:
            raise FileNotFoundError("This project has no dataset saved yet.")
        target = self.path / DATA_DIR / self.meta.dataset_file
        self._frame = pd.read_parquet(target) if target.suffix == ".parquet" else pd.read_csv(target)
        return self._frame

    # -- context --------------------------------------------------------------
    def set_context(self, context: BusinessContext) -> None:
        self.meta.context = context.to_dict()
        self.save()

    def context(self) -> BusinessContext:
        context = BusinessContext()
        for key, value in (self.meta.context or {}).items():
            if hasattr(context, key):
                setattr(context, key, value)
        return context

    # -- pipelines ------------------------------------------------------------
    def save_pipeline(self, pipeline: PreprocessingPipeline, name: str | None = None) -> Path:
        name = name or pipeline.name
        target = self.path / PIPELINES_DIR / f"{_slug(name)}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(pipeline.to_dict(), indent=2, default=str), encoding="utf-8")
        if name not in self.meta.saved_pipelines:
            self.meta.saved_pipelines.append(name)
            self.save()
        return target

    def load_pipeline(self, name: str) -> PreprocessingPipeline:
        target = self.path / PIPELINES_DIR / f"{_slug(name)}.json"
        if not target.exists():
            raise FileNotFoundError(f"No saved pipeline called '{name}' in this project.")
        return PreprocessingPipeline.from_dict(json.loads(target.read_text(encoding="utf-8")))

    def list_pipelines(self) -> list[str]:
        return sorted(p.stem for p in (self.path / PIPELINES_DIR).glob("*.json"))

    # -- runs -----------------------------------------------------------------
    def save_run(self, run: Any, save_model: bool = True) -> Path:
        manifest = build_manifest(run)
        target = self.path / RUNS_DIR / f"{run.id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": manifest.to_dict(),
            "trace": [e.to_dict() for e in run.trace.events],
            "findings": [f.to_dict() for f in run.findings],
            "recommendations": [r.to_dict() for r in run.recommendations],
            "decisions": [d.to_dict() for d in run.decisions],
            "tournament": run.tournament.to_dict() if run.tournament else None,
            "self_check": run.self_check.to_dict() if run.self_check else None,
            "summary": run.summary(),
            "status": run.status,
        }
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if run.id not in self.meta.run_ids:
            self.meta.run_ids.append(run.id)
        self.save()
        return target

    def load_run(self, run_id: str) -> dict[str, Any]:
        target = self.path / RUNS_DIR / f"{run_id}.json"
        if not target.exists():
            raise FileNotFoundError(f"No run '{run_id}' saved in this project.")
        return json.loads(target.read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        """Every saved run, newest first, with just enough to pick one."""
        rows = []
        for path in (self.path / RUNS_DIR).glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                manifest = payload.get("manifest", {})
                rows.append(
                    {
                        "run_id": manifest.get("run_id", path.stem),
                        "created_at": manifest.get("created_at", ""),
                        "task": (manifest.get("objective") or {}).get("task_type", ""),
                        "target": (manifest.get("objective") or {}).get("target", ""),
                        "model": (manifest.get("selected_model") or {}).get("name", ""),
                        "fingerprint": RunManifest(**{
                            k: v for k, v in manifest.items() if k in RunManifest.__dataclass_fields__
                        }).fingerprint(),
                        "summary": payload.get("summary", ""),
                        "status": payload.get("status", ""),
                    }
                )
            except Exception:
                continue
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    def save_model(self, model: Any, run_id: str) -> Path | None:
        """Persist a fitted model with joblib, if it is available."""
        try:
            import joblib
        except ImportError:
            return None
        target = self.path / MODELS_DIR / f"{run_id}.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(model, target)
        except Exception:
            return None
        return target

    def load_model(self, run_id: str) -> Any:
        import joblib

        target = self.path / MODELS_DIR / f"{run_id}.joblib"
        if not target.exists():
            raise FileNotFoundError(f"No saved model for run '{run_id}'.")
        return joblib.load(target)

    def save_report(self, content: str, name: str, extension: str = "md") -> Path:
        target = self.path / REPORTS_DIR / f"{_slug(name)}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def summary(self) -> str:
        return (
            f"Project '{self.meta.name}' at {self.path}\n"
            f"  Dataset: {self.meta.dataset_rows:,} rows × {self.meta.dataset_columns} columns\n"
            f"  Runs: {len(self.meta.run_ids)}   Saved pipelines: {len(self.meta.saved_pipelines)}\n"
            f"  Created {self.meta.created_at}, last updated {self.meta.updated_at}"
        )


def list_projects(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    rows = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or not Project.exists(candidate):
            continue
        try:
            project = Project.open(candidate)
            rows.append(
                {
                    "name": project.meta.name,
                    "path": str(candidate),
                    "rows": project.meta.dataset_rows,
                    "columns": project.meta.dataset_columns,
                    "runs": len(project.meta.run_ids),
                    "updated_at": project.meta.updated_at,
                }
            )
        except Exception:
            continue
    return sorted(rows, key=lambda r: r["updated_at"], reverse=True)


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    return keep[:80] or "unnamed"
