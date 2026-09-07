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
    #: Column type corrections the user made. Kept apart from the profile so a
    #: re-profile does not silently undo them — the user is the authority on
    #: what a column means, and the profiler only on what it measured.
    type_overrides: dict[str, Any] = field(default_factory=dict)
    #: Several named pipelines can be held at once and compared. ``pipeline``
    #: below is a view onto whichever is active, so everything that predates
    #: multiple pipelines keeps working unchanged.
    pipelines: dict[str, PreprocessingPipeline] = field(default_factory=dict)
    active_pipeline: str = "default"
    settings: RunSettings = field(default_factory=RunSettings)
    run: AnalysisRun | None = None
    runs: list[AnalysisRun] = field(default_factory=list)
    #: Summaries of each distinct state of the data seen this session. Never the
    #: rows — a version records shape, types, missingness and distributions.
    dataset_versions: list[Any] = field(default_factory=list)
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

    # -- lifecycle ------------------------------------------------------------
    def set_dataset(self, frame: pd.DataFrame, source: DataSource | None, name: str) -> None:
        """Load a new dataset, discarding everything derived from the last one.

        This exists because forgetting to do it is silent and awful: the frame
        changes, the profile does not, and every page downstream then describes
        columns that are no longer there. The only way out used to be restarting
        the app.
        """
        self.frame = frame
        self.source = source
        self.dataset_name = name
        self.forget_derived()

    def forget_derived(self) -> None:
        """Clear everything computed *from* the data. The data itself stays.

        The engine goes too. It caches the typed frame and every fitted model
        from the dataset it was working on, so keeping it across a change of
        dataset is how a model trained on one file ends up scoring another.
        """
        self.profile = None
        self.typed_frame = None
        self.objectives = []
        self.objective = None
        self.type_overrides = {}
        self.run = None
        self.runs = []
        self.pipelines = {}
        self.active_pipeline = "default"
        self.chat = []
        self.dataset_versions = []
        self.scientist = None
        # Widget values Streamlit is holding on our behalf. A selectbox keyed by
        # column name keeps its old selection, and a column that no longer
        # exists is not a valid option — which raises rather than degrading.
        self._clear_widget_state()

    def reset_analysis(self, reprofile: bool = False) -> None:
        """The results are stale, but the data is not.

        The default is for a change of business context: the measurements still
        stand, so the profile is kept, but what the platform thinks you want and
        everything built on that has to be worked out again.

        ``reprofile=True`` is for when the *meaning* of a column changed — a
        corrected type changes how that column is coerced, so the profile and
        the typed frame both have to be rebuilt. The user's corrections are held
        on ``type_overrides`` and re-applied, so rebuilding does not overrule
        them.
        """
        self.objectives = []
        self.objective = None
        self.run = None
        self.pipelines = {}
        self.active_pipeline = "default"
        self.scientist = None
        if reprofile:
            self.profile = None
            self.typed_frame = None
        self._clear_widget_state()

    @staticmethod
    def _clear_widget_state() -> None:
        """Drop the widget values that name columns of the previous dataset.

        Streamlit holds a keyed widget's last value across reruns. A selectbox
        keyed by column name therefore comes back pointing at a column the new
        dataset does not have, and an invalid option raises rather than
        degrading — which is the other half of why loading a second file used to
        require restarting the app.
        """
        try:
            import streamlit as st

            stale = [
                key for key in st.session_state
                if isinstance(key, str)
                and key.startswith(("_prep_", "_inspect_", "imp_", "impg_", "impv_",
                                    "_score_", "_preprocessing_", "_library_"))
            ]
            for key in stale:
                del st.session_state[key]
        except Exception:
            # Called outside a Streamlit run (a test, a script). Nothing to clear.
            pass

    def profile_dataset(self) -> None:
        """Profile the loaded frame, re-applying the user's type corrections.

        One place, so every caller gets the corrections back. Re-profiling
        without them would quietly overrule the user, which is the opposite of
        how this platform is supposed to treat a person's judgement.
        """
        from dsai.core.profiler import override_semantic_type

        run = scientist().understand(self.frame, self.dataset_name, self.context, self.source)
        profile = run.profile
        for column, semantic_type in self.type_overrides.items():
            try:
                override_semantic_type(profile, column, semantic_type)
            except KeyError:
                continue        # the column is gone; the correction no longer applies
        self.profile = profile
        self.typed_frame = scientist()._typed_frame
        self.objectives = run.objectives

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
