"""Session state for the workspace UI.

Keeping every mutable object behind one accessor means the pages stay thin and
the state is inspectable in one place rather than scattered across widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dsai.core.schema import BusinessContext, DatasetProfile, Objective
from dsai.dataio.loaders import DataSource
from dsai.engines.orchestrator import AIDataScientist, AnalysisRun, RunSettings
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.registry.base import load_builtin_models

STATE_KEY = "dsai_workspace"


@dataclass
class Workspace:
    """Everything the UI is currently working on."""

    frame: pd.DataFrame | None = None
    typed_frame: pd.DataFrame | None = None
    source: DataSource | None = None
    dataset_name: str = ""
    profile: DatasetProfile | None = None
    context: BusinessContext = field(default_factory=BusinessContext)
    objectives: list[Objective] = field(default_factory=list)
    objective: Objective | None = None
    pipeline: PreprocessingPipeline | None = None
    settings: RunSettings = field(default_factory=RunSettings)
    run: AnalysisRun | None = None
    runs: list[AnalysisRun] = field(default_factory=list)
    scientist: AIDataScientist | None = None
    project_path: str | None = None
    mode: str = "guided"                 # guided | advanced
    theme: str = "light"
    chat: list[dict[str, Any]] = field(default_factory=list)
    notices: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.frame is not None and not self.frame.empty

    @property
    def has_run(self) -> bool:
        return self.run is not None and self.run.status == "complete"

    def reset_analysis(self) -> None:
        self.run = None
        self.objective = None
        self.pipeline = None

    def notify(self, level: str, message: str) -> None:
        self.notices.append((level, message))

    def take_notices(self) -> list[tuple[str, str]]:
        notices, self.notices = self.notices, []
        return notices


def workspace() -> Workspace:
    """The single Workspace for this browser session."""
    import streamlit as st

    if STATE_KEY not in st.session_state:
        state = Workspace()
        state.scientist = AIDataScientist(registry=load_builtin_models())
        st.session_state[STATE_KEY] = state
    return st.session_state[STATE_KEY]


def scientist() -> AIDataScientist:
    state = workspace()
    if state.scientist is None:
        state.scientist = AIDataScientist(registry=load_builtin_models())
    return state.scientist
