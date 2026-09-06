"""Analysis page: choose the objective, review the plan, run it."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    apply_theme, caveat, sidebar_chrome, inference, decision_panel, metric_row, override_notice, page_header, require_data, show_notices,
    trace_view, workflow_nav,
)
from dsai.app.state import scientist, workspace
from dsai.core.schema import Objective, TaskType
from dsai.engines.decision import Constraints, plan_analysis
from dsai.engines.orchestrator import RunSettings

st.set_page_config(page_title="Analysis · DSAI", page_icon="🧪", layout="wide")
state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Analysis",
    "Choose what to find out, review exactly what will run, then run it.",
    "Step 4 of 7",
)
workflow_nav("Analysis", state)

if not require_data(state):
    st.stop()

profile = state.profile
frame = state.typed_frame if state.typed_frame is not None else state.frame

# --------------------------------------------------------------------------
# mode
# --------------------------------------------------------------------------
mode = st.radio(
    "How do you want to work?",
    ["AI automatic", "AI assisted", "Manual / expert"],
    horizontal=True,
    captions=[
        "Do the analysis for me, end to end.",
        "Recommend an approach, let me approve or change it, then run.",
        "I choose the objective, models and settings myself.",
    ],
    help="All three use the same underlying engine — only how much you decide changes.",
)
mode_key = {"AI automatic": "automatic", "AI assisted": "assisted", "Manual / expert": "manual"}[mode]
state.mode = "advanced" if mode_key == "manual" else "guided"

st.divider()

# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
st.subheader("What do you want to find out?")
override_notice(
    "These are ranked suggestions. Pick any of them, or define your own below — the platform "
    "recommends, you decide."
)

objectives = state.objectives or []
labels = [
    f"{o.label()}  ·  {'you asked for this' if o.source.startswith('user') else 'detected from the data'}"
    for o in objectives
]
choice = st.radio("Suggested objectives", labels + ["Define my own"], index=0 if labels else 0)

if choice == "Define my own":
    columns = st.columns(3)
    task = columns[0].selectbox("Analysis type", list(TaskType),
                                format_func=lambda t: t.value.replace("_", " "))
    needs_target = task.is_supervised
    target = columns[1].selectbox(
        "Target variable", ["— none —"] + list(profile.columns),
        disabled=not needs_target,
    )
    time_column = columns[2].selectbox(
        "Time column", ["— none —"] + profile.datetime_columns,
        disabled=task is not TaskType.TIME_SERIES_FORECAST,
    )
    feature_columns = st.multiselect(
        "Variables to use (leave empty to use everything appropriate)",
        [c for c in profile.columns],
    )
    extra = st.columns(2)
    n_clusters = extra[0].number_input("Number of segments (0 = let the platform choose)", 0, 30, 0,
                                       disabled=task is not TaskType.CLUSTERING)
    horizon = extra[1].number_input("Forecast horizon (0 = default)", 0, 120, 0,
                                    disabled=task is not TaskType.TIME_SERIES_FORECAST)
    objective = Objective(
        task_type=task,
        target=None if target.startswith("—") else target,
        features=feature_columns,
        time_column=None if time_column.startswith("—") else time_column,
        n_clusters=int(n_clusters) or None,
        horizon=int(horizon) or None,
        rationale="You defined this objective directly.",
        source="user_override",
        priority=1.0,
    )
else:
    objective = objectives[labels.index(choice)]
    inference(objective.rationale, label="Why this objective")

state.objective = objective

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
st.subheader("Settings")
columns = st.columns(4)
max_models = columns[0].slider("Models to compare", 2, 20, state.settings.max_models)
time_budget = columns[1].selectbox("Time budget", ["fast", "balanced", "thorough"],
                                   index=["fast", "balanced", "thorough"].index(state.settings.time_budget))
interpretability = columns[2].selectbox(
    "How important is interpretability?", ["low", "moderate", "high", "critical"],
    index=["low", "moderate", "high", "critical"].index(state.settings.interpretability_need),
    help="'Critical' restricts the field to models whose mechanism can be read directly.",
)
test_size = columns[3].slider("Hold-out share", 0.1, 0.4, state.settings.test_size, 0.05)

advanced = st.expander("Advanced settings", expanded=state.mode == "advanced")
with advanced:
    columns = st.columns(3)
    random_state = columns[0].number_input("Random seed", 0, 10_000, state.settings.random_state,
                                           help="Recorded in the manifest so the run can be reproduced exactly.")
    tune = columns[1].checkbox("Tune hyper-parameters", state.settings.tune_hyperparameters,
                               help="Slower. Uses a randomised search inside cross-validation.")
    aggressiveness = columns[2].selectbox("Preprocessing depth", ["minimal", "standard", "thorough"],
                                          index=1)
    from dsai.registry.base import REGISTRY

    candidates = [s.key for s in REGISTRY.find(task_type=objective.task_type, only_available=True)]
    include = st.multiselect("Always include these models", candidates)
    exclude = st.multiselect("Never use these models", candidates)

settings = RunSettings(
    mode=mode_key,
    max_models=max_models,
    time_budget=time_budget,
    interpretability_need=interpretability,
    tune_hyperparameters=tune,
    test_size=test_size,
    random_state=int(random_state),
    preprocessing_aggressiveness=aggressiveness,
    include_models=include,
    exclude_models=exclude,
)
state.settings = settings

# --------------------------------------------------------------------------
# plan preview
# --------------------------------------------------------------------------
st.divider()
st.subheader("What will run")

plan = plan_analysis(
    profile, objective, state.context,
    Constraints(
        max_models=max_models, time_budget=time_budget,
        interpretability_need=interpretability,
        include_models=include, exclude_models=exclude,
    ),
    frame=frame,
)

metric_row([
    ("Models to train", str(len(plan.candidates)), "Chosen to span different model families"),
    ("Ranked on", plan.primary_metric, "The metric the comparison is ranked by"),
    ("Validation", plan.validation_strategy.get("strategy", "—").replace("_", " "),
     plan.validation_strategy.get("reason", "")),
    ("Folds", str(plan.validation_strategy.get("n_splits", "—")), "Cross-validation splits"),
])

for warning in plan.warnings:
    caveat(warning)

candidates_tab, reasoning_tab = st.tabs(["Candidate models", "Why these choices"])
with candidates_tab:
    import pandas as pd

    st.dataframe(
        pd.DataFrame([
            {
                "Model": c.name, "Family": c.family, "Fit score": round(c.score, 3),
                "Interpretability": c.interpretability, "Cost": c.cost,
                "Chosen because": c.reasons[0] if c.reasons else "general-purpose fit",
                "Concern": c.concerns[0] if c.concerns else "",
            }
            for c in plan.candidates
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "The shortlist deliberately spans model families. If a simple model matches a complex one, "
        "that is itself a finding."
    )
with reasoning_tab:
    decision_panel(plan.decisions)

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
st.divider()
label = {
    "automatic": "▶  Run autonomous analysis",
    "assisted": "▶  Run the approved plan",
    "manual": "▶  Run with my settings",
}[mode_key]

if st.button(label, type="primary", use_container_width=True):
    placeholder = st.empty()
    lines: list[str] = []

    def on_event(event):
        icons = {"done": "✓", "running": "…", "warning": "!", "failed": "✗", "skipped": "–"}
        detail = f" — {event.detail}" if event.detail else ""
        lines.append(f"{icons.get(event.status, '·')} {event.step}{detail}")
        placeholder.code("\n".join(lines[-25:]), language=None)

    engine = scientist()
    engine.trace_callback = on_event
    try:
        with st.spinner("Analysing…"):
            run = engine.understand(state.frame, state.dataset_name, state.context, state.source)
            run.objective = objective
            engine.plan(run, objective, settings)
            if state.pipeline is not None and state.mode == "advanced":
                run.pipeline = state.pipeline
            engine.execute(run)
            engine.interpret(run)
        state.run = run
        state.runs.append(run)
        state.pipeline = run.pipeline
        state.typed_frame = engine._typed_frame
        state.notify("success", f"Analysis complete in {run.duration_s}s. {run.summary()}")
        st.rerun()
    except Exception as exc:
        st.error(f"The analysis failed: {type(exc).__name__}: {exc}")
    finally:
        engine.trace_callback = None

if state.run is not None:
    st.divider()
    st.subheader("Last run")
    st.markdown(state.run.summary())
    trace_view(state.run.trace)
    st.markdown("Continue to **Models** to see the comparison, or **Insights** for the findings.")
