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
class Accessibility:
    """How the interface should present itself, per user.

    These are not preferences about taste. Each one answers a barrier someone
    actually hits: text too small to read, contrast too low to distinguish,
    motion that makes a page unusable, and charts that carry information no
    screen reader can reach.
    """

    #: Multiplies the whole type scale. 1.0 | 1.15 | 1.3 | 1.5
    text_scale: float = 1.0
    #: Pushes ink and borders to maximum contrast against the surface.
    high_contrast: bool = False
    #: Suppresses every transition and animation in the app.
    reduce_motion: bool = False
    #: Shows the numbers behind every chart without needing to expand anything.
    always_show_tables: bool = False
    #: Underlines links so they are not distinguished by colour alone.
    underline_links: bool = False

    @property
    def any_enabled(self) -> bool:
        return (self.text_scale != 1.0 or self.high_contrast or self.reduce_motion
                or self.always_show_tables or self.underline_links)


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
    #: Several named pipelines can be held at once and compared. ``pipeline``
    #: below is a view onto whichever is active, so everything that predates
    #: multiple pipelines keeps working unchanged.
    pipelines: dict[str, PreprocessingPipeline] = field(default_factory=dict)
    active_pipeline: str = "default"
    settings: RunSettings = field(default_factory=RunSettings)
    run: AnalysisRun | None = None
    runs: list[AnalysisRun] = field(default_factory=list)
    scientist: AIDataScientist | None = None
    project_path: str | None = None
    mode: str = "guided"                 # guided | advanced
    #: Drives the interface chrome and the charts together, from the same tokens.
    #: Dark by default — it is what the base theme in config.toml is set to, so
    #: the default costs no override, and it is the mode this workspace is
    #: designed around.
    theme: str = "dark"                  # light | dark
    #: Presentation settings that remove barriers rather than express taste.
    access: Accessibility = field(default_factory=Accessibility)
    chat: list[dict[str, Any]] = field(default_factory=list)
    notices: list[tuple[str, str]] = field(default_factory=list)

    # -- pipelines ------------------------------------------------------------
    @property
    def pipeline(self) -> PreprocessingPipeline | None:
        """The pipeline currently being edited and used for analysis."""
        return self.pipelines.get(self.active_pipeline)

    @pipeline.setter
    def pipeline(self, value: PreprocessingPipeline | None) -> None:
        if value is None:
            self.pipelines.pop(self.active_pipeline, None)
        else:
            self.pipelines[self.active_pipeline] = value

    def add_pipeline(self, name: str, pipeline: PreprocessingPipeline,
                     activate: bool = True) -> str:
        """Store a pipeline under a name, keeping existing names distinct."""
        base, suffix = name.strip() or "pipeline", 1
        unique = base
        while unique in self.pipelines and self.pipelines[unique] is not pipeline:
            suffix += 1
            unique = f"{base} ({suffix})"
        pipeline.name = unique
        self.pipelines[unique] = pipeline
        if activate:
            self.active_pipeline = unique
        return unique

    def remove_pipeline(self, name: str) -> None:
        self.pipelines.pop(name, None)
        if self.active_pipeline == name:
            self.active_pipeline = next(iter(self.pipelines), "default")

    @property
    def pipeline_names(self) -> list[str]:
        return list(self.pipelines)

    @property
    def has_data(self) -> bool:
        return self.frame is not None and not self.frame.empty

    @property
    def has_run(self) -> bool:
        return self.run is not None and self.run.status == "complete"

    def reset_analysis(self) -> None:
        self.run = None
        self.objective = None
        self.pipelines = {}
        self.active_pipeline = "default"

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
