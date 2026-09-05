"""Preprocessing page: review, edit or build the pipeline."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    dataframe, decision_panel, override_notice, page_header, require_data, show_notices, workflow_nav,
)
from dsai.app.state import workspace
from dsai.core.schema import TaskType
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.preprocessing.recommender import recommend_pipeline
from dsai.preprocessing.steps import STEPS
from dsai.registry.base import REGISTRY

st.set_page_config(page_title="Preprocessing · DSAI", page_icon="🧹", layout="wide")
state = workspace()
show_notices(state)

page_header(
    "Preprocessing",
    "Preprocessing is not a fixed recipe — what a dataset needs depends on the model that will "
    "consume it. Steps that learn from the data are fitted inside each cross-validation fold, never "
    "on the full dataset.",
    "Step 3 of 7",
)
workflow_nav("Preprocessing")

if not require_data(state):
    st.stop()

profile = state.profile
frame = state.typed_frame if state.typed_frame is not None else state.frame
objective = state.objective or (state.objectives[0] if state.objectives else None)

if objective is None:
    st.info("No analytical objective is set. Choose one on the **Analysis** page first.")
    st.stop()

build_tab, edit_tab, preview_tab, saved_tab = st.tabs(
    ["Build", "Edit steps", "Preview effect", "Saved pipelines"]
)

with build_tab:
    columns = st.columns([2, 1, 1])
    model_keys = ["— general purpose —"] + [
        s.key for s in REGISTRY.find(task_type=objective.task_type, only_available=True)
    ]
    model_choice = columns[0].selectbox(
        "Design the pipeline for which model?", model_keys,
        help="A random forest needs no scaling and keeps its outliers. KNN needs both handled. "
             "The recommended pipeline differs accordingly.",
    )
    aggressiveness = columns[1].selectbox(
        "How much preprocessing?", ["minimal", "standard", "thorough"], index=1,
        help="Minimal: only what is necessary. Thorough: also transforms distributions and reduces dimensions.",
    )
    columns[2].write("")
    if columns[2].button("Build recommended pipeline", type="primary", use_container_width=True):
        spec = None if model_choice.startswith("—") else REGISTRY.get(model_choice)
        pipeline, decisions = recommend_pipeline(
            profile, objective.task_type, objective.target, spec, state.context,
            aggressiveness=aggressiveness,
        )
        state.pipeline = pipeline
        state.notify("success", f"Built a {len(pipeline.active_steps)}-step pipeline.")
        st.session_state["_preprocessing_decisions"] = decisions
        st.rerun()

    if st.button("Start from an empty pipeline"):
        state.pipeline = PreprocessingPipeline(name="manual")
        st.session_state["_preprocessing_decisions"] = []
        st.rerun()

    decisions = st.session_state.get("_preprocessing_decisions", [])
    if decisions:
        st.subheader("Why these steps")
        decision_panel(decisions)

pipeline = state.pipeline

with edit_tab:
    if pipeline is None:
        st.info("Build a pipeline first.")
    else:
        override_notice()
        st.subheader(f"Pipeline: {len(pipeline.steps)} step(s)")
        for index, step in enumerate(pipeline.steps):
            with st.container(border=True):
                columns = st.columns([6, 1, 1, 1, 1])
                columns[0].markdown(
                    f"**{index + 1}. {step.spec.name}**  \n"
                    f"<span style='color:#6b7280;font-size:0.85rem'>{step.describe()}</span>",
                    unsafe_allow_html=True,
                )
                if columns[1].button("↑", key=f"up_{step.id}", disabled=index == 0, help="Move earlier"):
                    pipeline.move(step.id, index - 1)
                    st.rerun()
                if columns[2].button("↓", key=f"down_{step.id}",
                                     disabled=index == len(pipeline.steps) - 1, help="Move later"):
                    pipeline.move(step.id, index + 1)
                    st.rerun()
                if columns[3].button("⏻", key=f"toggle_{step.id}",
                                     help="Disable" if step.enabled else "Enable"):
                    pipeline.set_enabled(step.id, not step.enabled)
                    st.rerun()
                if columns[4].button("✕", key=f"del_{step.id}", help="Remove"):
                    pipeline.remove(step.id)
                    st.rerun()
                if not step.spec.leakage_safe:
                    st.caption("↳ fitted inside each cross-validation fold, so it cannot leak")

        columns = st.columns(3)
        if columns[0].button("Undo", disabled=not pipeline._undo_stack):
            pipeline.undo()
            st.rerun()
        if columns[1].button("Redo", disabled=not pipeline._redo_stack):
            pipeline.redo()
            st.rerun()
        if columns[2].button("Clear all"):
            pipeline.clear()
            st.rerun()

        st.divider()
        st.subheader("Add a step")
        category = st.selectbox("Category", STEPS.categories())
        available = STEPS.by_category(category)
        step_key = st.selectbox("Operation", [s.key for s in available],
                                format_func=lambda k: STEPS.get(k).name)
        spec = STEPS.get(step_key)
        st.caption(spec.description)
        if spec.when_to_use:
            st.markdown(f"**When to use it:** {spec.when_to_use}")
        for caveat in spec.caveats:
            st.warning(caveat, icon="⚠️")

        applicable = (
            [n for n, c in profile.columns.items() if not spec.applies_to or c.semantic_type in spec.applies_to]
            if spec.scope == "column" else list(profile.columns)
        )
        target_columns = st.multiselect("Apply to which columns?", applicable, applicable[:0])
        params = {}
        for param in spec.params:
            if param.kind == "int":
                params[param.name] = st.number_input(
                    param.name, value=int(param.default or 0),
                    min_value=int(param.low) if param.low is not None else None,
                    max_value=int(param.high) if param.high is not None else None,
                    help=param.description or None, key=f"p_{step_key}_{param.name}",
                )
            elif param.kind == "float":
                params[param.name] = st.number_input(
                    param.name, value=float(param.default or 0.0),
                    min_value=float(param.low) if param.low is not None else None,
                    max_value=float(param.high) if param.high is not None else None,
                    help=param.description or None, key=f"p_{step_key}_{param.name}",
                )
            elif param.kind == "bool":
                params[param.name] = st.checkbox(param.name, bool(param.default),
                                                 help=param.description or None,
                                                 key=f"p_{step_key}_{param.name}")
            elif param.choices:
                params[param.name] = st.selectbox(param.name, param.choices,
                                                  help=param.description or None,
                                                  key=f"p_{step_key}_{param.name}")
        if st.button("Add step", type="primary"):
            pipeline.add(step_key, columns=target_columns or None, by="user", **params)
            state.notify("success", f"Added {spec.name}.")
            st.rerun()

with preview_tab:
    if pipeline is None or not pipeline.active_steps:
        st.info("Build a pipeline first.")
    else:
        st.caption(
            "Preview only — fitted on a sample so you can see the effect. The real fit happens "
            "inside cross-validation."
        )
        target = frame[objective.target] if objective.target in frame.columns else None
        features = frame.drop(columns=[objective.target], errors="ignore")
        preview = pipeline.preview(features, target)
        for stage in preview["stages"]:
            if stage.get("status") == "failed":
                st.error(f"{stage['step']} — {stage.get('error', '')}")
            else:
                before = stage.get("columns_before")
                after = stage.get("columns_after")
                delta = f"  ({before} → {after} columns)" if before is not None and before != after else ""
                st.markdown(f"✓ {stage['step']}{delta}")
        st.markdown(f"**Result:** {preview['final_shape'][0]:,} rows × {preview['final_shape'][1]} columns")
        dataframe(preview["sample"].head(15).round(4))

        st.subheader("Leakage control")
        report = pipeline.leakage_report()
        st.info(report["explanation"], icon="🔒")
        columns = st.columns(2)
        columns[0].markdown("**Fitted inside each fold**")
        for name in report["fitted_inside_cross_validation"] or ["(none)"]:
            columns[0].markdown(f"- {name}")
        columns[1].markdown("**Applied before splitting**")
        for name in report["applied_before_split"] or ["(none)"]:
            columns[1].markdown(f"- {name}")

with saved_tab:
    from dsai.repro.project import Project

    if not state.project_path:
        st.info("Save a project on the **Projects** page to store and reuse pipelines.")
    else:
        project = Project.open(state.project_path)
        columns = st.columns([2, 1])
        name = columns[0].text_input("Pipeline name", value=pipeline.name if pipeline else "pipeline")
        columns[1].write("")
        if columns[1].button("Save this pipeline", disabled=pipeline is None, use_container_width=True):
            project.save_pipeline(pipeline, name)
            state.notify("success", f"Saved pipeline '{name}'.")
            st.rerun()
        saved = project.list_pipelines()
        if saved:
            chosen = st.selectbox("Load a saved pipeline", saved)
            if st.button("Load"):
                state.pipeline = project.load_pipeline(chosen)
                state.notify("success", f"Loaded pipeline '{chosen}'.")
                st.rerun()

if pipeline is not None and pipeline.history:
    with st.expander("Pipeline history"):
        for entry in reversed(pipeline.history[-30:]):
            st.markdown(f"- `{entry.timestamp}` **{entry.action}** — {entry.detail}  *(by {entry.by})*")
