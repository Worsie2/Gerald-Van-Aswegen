"""Preprocessing page: review, edit or build the pipeline."""

from __future__ import annotations

import streamlit as st

import pandas as pd

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, dataframe, decision_panel, empty_state, inference,
    metric_row, override_notice, page_header, pipeline_graph, require_data, show_notices,
    sidebar_chrome, workflow_nav,
)
from dsai.viz import plots
from dsai.app.state import workspace
from dsai.core.schema import Objective, TaskType
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.preprocessing.recommender import recommend_pipeline
from dsai.preprocessing.steps import STEPS
from dsai.registry.base import REGISTRY

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Preprocessing",
    "Preprocessing is not a fixed recipe — what a dataset needs depends on the model that will "
    "consume it. Steps that learn from the data are fitted inside each cross-validation fold, never "
    "on the full dataset.",
    "Step 3 of 7",
)
workflow_nav("Preprocessing", state)

if not require_data(state):
    st.stop()

profile = state.profile
frame = state.typed_frame if state.typed_frame is not None else state.frame
# ---------------------------------------------------------------------------
# What the pipeline is for
#
# Preprocessing depends on the question, so the question has to be answerable
# here. Defaulting silently to the top-ranked objective made every pipeline a
# regression pipeline and left no way to say otherwise without leaving the page.
# ---------------------------------------------------------------------------
detected = state.objectives or []
options = [o.label() for o in detected] + ["Something else…"]

current = state.objective
index = 0
if current is not None:
    for position, candidate in enumerate(detected):
        if (candidate.task_type, candidate.target) == (current.task_type, current.target):
            index = position
            break
    else:
        index = len(options) - 1

purpose, note = st.columns([2, 3])
selected = purpose.selectbox(
    "What is this pipeline for?", options, index=index,
    help="Preprocessing is not one recipe. A classifier, a clustering run and a forecast "
         "each need different treatment, so the pipeline is built for the question you pick here.",
)

if selected == "Something else…":
    manual = st.columns([2, 2, 1])
    task = manual[0].selectbox(
        "Analysis type", list(TaskType),
        format_func=lambda t: t.value.replace("_", " "),
        index=list(TaskType).index(current.task_type) if current is not None else 0,
        key="_prep_task",
    )
    target_options = ["— none —"] + list(profile.columns)
    target_index = 0
    if current is not None and current.target in target_options:
        target_index = target_options.index(current.target)
    target = manual[1].selectbox(
        "Target variable", target_options, index=target_index,
        disabled=not task.is_supervised, key="_prep_target",
        help="Held out of the pipeline so no step can learn from the answer.",
    )
    objective = Objective(
        task_type=task,
        target=None if target.startswith("—") or not task.is_supervised else target,
        rationale="You chose this on the Preprocessing page.",
        source="user_override",
        priority=1.0,
    )
else:
    objective = detected[options.index(selected)]

if objective.task_type.is_supervised and not objective.target:
    caveat(
        f"{objective.task_type.value.replace('_', ' ').capitalize()} needs a target variable. "
        "Pick one above, or the pipeline will be built as if nothing were being predicted — "
        "which means no column is protected from leaking into the features."
    )

# Keep the rest of the workflow in step with what was chosen here.
state.objective = objective
note.write("")
note.caption(
    f"Building for **{objective.label().replace('_', ' ')}**"
    + (f" — '{objective.target}' is held out of every step." if objective.target else "")
    + "  This choice carries through to the Analysis page."
)

st.divider()

# A pipeline switcher above the tabs: several can be held at once and compared.
switch, new, delete = st.columns([3, 1, 1])
names = state.pipeline_names
if names:
    chosen = switch.selectbox(
        "Working on", names,
        index=names.index(state.active_pipeline) if state.active_pipeline in names else 0,
        help="Several pipelines can be held at once. The active one is used by the Analysis page.",
    )
    if chosen != state.active_pipeline:
        state.active_pipeline = chosen
        st.rerun()
else:
    switch.caption("No pipeline yet — build one below.")

new.write("")
if new.button("Duplicate", disabled=state.pipeline is None, use_container_width=True,
              help="Copy the active pipeline so you can try a variation without losing this one."):
    state.add_pipeline(f"{state.active_pipeline} copy", state.pipeline.copy())
    state.notify("success", f"Duplicated as '{state.active_pipeline}'.")
    st.rerun()
delete.write("")
if delete.button("Delete", disabled=len(names) < 2, use_container_width=True):
    state.remove_pipeline(state.active_pipeline)
    st.rerun()

build_tab, graph_tab, edit_tab, preview_tab, effect_tab, compare_tab, saved_tab = st.tabs(
    ["Build", "The pipeline", "Edit steps", "Preview effect", "What it changed",
     "Compare pipelines", "Saved pipelines"]
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
        label = "general purpose" if model_choice.startswith("—") else REGISTRY.get(model_choice).name
        name = state.add_pipeline(f"{label} · {aggressiveness}", pipeline)
        state.notify("success", f"Built '{name}' — {len(pipeline.active_steps)} step(s).")
        st.session_state["_preprocessing_decisions"] = decisions
        st.rerun()

    if st.button("Start from an empty pipeline"):
        state.add_pipeline("manual", PreprocessingPipeline(name="manual"))
        st.session_state["_preprocessing_decisions"] = []
        st.rerun()

    decisions = st.session_state.get("_preprocessing_decisions", [])
    if decisions:
        st.subheader("Why these steps")
        decision_panel(decisions)

pipeline = state.pipeline

with graph_tab:
    if pipeline is None:
        empty_state(
            "No pipeline yet",
            "Build one on the **Build** tab and it is drawn here as the chain it is — "
            "raw data at the top, the model at the bottom, every step between them named.",
        )
    else:
        active = pipeline.active_steps
        by_ai = sum(1 for s in active if s.added_by == "ai")
        ai_panel(
            f"This pipeline has **{len(active)} active step(s)**"
            + (f", **{by_ai}** of which I added." if by_ai else ", all of them yours.")
            + " Steps that learn anything from the data are fitted inside each cross-validation "
            "fold, never on the full dataset, so no step can see the rows it will be scored on.",
            heading="AI generated pipeline",
            why="Row-scope operations run before the split because they change which rows exist. "
                "Column-scope operations are compiled into an unfitted scikit-learn pipeline and "
                "refitted per fold. That separation is what makes leakage structurally impossible "
                "rather than merely avoided.",
        )
        nodes: list[dict] = [{
            "name": "Raw data", "kind": "source",
            "detail": f"{profile.n_rows:,} rows × {profile.n_columns} columns",
        }]
        for position, step in enumerate(active, start=1):
            nodes.append({
                "name": step.spec.name,
                "kind": "step",
                "scope": "before the split" if step.spec.scope == "row" else "per fold",
                "detail": step.describe() + ("" if step.spec.leakage_safe
                                             else " · fitted inside each fold so it cannot leak"),
            })
        nodes.append({
            "name": f"Model ({objective.task_type.value.replace('_', ' ')})",
            "kind": "model",
            "detail": (f"predicting '{objective.target}'" if objective.target
                       else "unsupervised — no target held out"),
        })
        pipeline_graph(nodes)
        st.caption(
            "Reordering and configuring steps is on the **Edit steps** tab. Dragging nodes needs a "
            "custom front-end component Streamlit does not provide, so the graph is the picture "
            "and the controls are beside it."
        )
        disabled = [s for s in pipeline.steps if not s.enabled]
        if disabled:
            caveat(f"{len(disabled)} step(s) are switched off and are not drawn above.")

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
        for note in spec.caveats:
            caveat(note)

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
        inference(report["explanation"], label="Leakage control")
        columns = st.columns(2)
        columns[0].markdown("**Fitted inside each fold**")
        for name in report["fitted_inside_cross_validation"] or ["(none)"]:
            columns[0].markdown(f"- {name}")
        columns[1].markdown("**Applied before splitting**")
        for name in report["applied_before_split"] or ["(none)"]:
            columns[1].markdown(f"- {name}")

with effect_tab:
    if pipeline is None or not pipeline.active_steps:
        empty_state("Nothing to compare yet", "Build a pipeline first.")
    else:
        st.caption(
            "The same rows before and after the pipeline. Reducing collinearity is usually why a "
            "pipeline exists, so this is where you check whether it actually did."
        )
        target_series = frame[objective.target] if objective.target in frame.columns else None
        features = frame.drop(columns=[objective.target], errors="ignore")
        try:
            rows, _ = pipeline.apply_row_steps(features)
            aligned = target_series.loc[rows.index] if target_series is not None else None
            transformed = pipeline.build_sklearn_pipeline().fit_transform(rows, aligned)
        except Exception as exc:
            transformed = None
            st.error(f"The pipeline could not be applied: {exc}")

        if transformed is not None:
            before_numeric = rows.select_dtypes(include="number")
            metric_row([
                ("Rows", f"{len(rows):,}", f"was {len(features):,}"),
                ("Columns", f"{transformed.shape[1]}", f"was {features.shape[1]}"),
                ("Numeric columns", f"{transformed.select_dtypes(include='number').shape[1]}",
                 f"was {before_numeric.shape[1]}"),
                ("Missing cells", f"{int(transformed.isna().sum().sum()):,}",
                 f"was {int(features.isna().sum().sum()):,}"),
            ])

            figure = plots.correlation_comparison(rows, transformed, mode=state.theme)
            if figure is not None:
                chart(figure, key="prep_corr")
                caveat(
                    "Correlation is computed on the columns as they stand at each point. After "
                    "one-hot encoding the 'after' matrix has more, narrower columns, so compare "
                    "the pattern rather than counting cells.",
                )
            else:
                caveat("Too few numeric columns on one side to compare correlation.")

            before_vif = {k: v for k, v in profile.multicollinearity.items()
                          if k in before_numeric.columns}
            if before_vif:
                from dsai.statistics.descriptive import variance_inflation_factors

                after_table = variance_inflation_factors(transformed)
                after_vif = dict(zip(after_table["variable"], after_table["vif"])) \
                    if not after_table.empty else {}
                figure = plots.vif_comparison(before_vif, after_vif, mode=state.theme)
                if figure is not None:
                    chart(figure, key="prep_vif")

            st.markdown("**One variable, before and after**")
            shared = [c for c in before_numeric.columns if c in transformed.columns]
            if shared:
                column = st.selectbox("Variable", shared, key="prep_dist_column")
                figure = plots.distribution_comparison(
                    rows[column], transformed[column], mode=state.theme,
                )
                if figure is not None:
                    chart(figure, key="prep_dist")
            else:
                caveat(
                    "No column survives the pipeline under its original name — every one was "
                    "encoded, renamed or replaced, so there is no like-for-like comparison to draw."
                )

            preview = pipeline.preview(features, target_series)
            figure = plots.pipeline_shape(preview["stages"], mode=state.theme)
            if figure is not None:
                chart(figure, key="prep_shape")

with compare_tab:
    if len(state.pipeline_names) < 2:
        empty_state(
            "Only one pipeline",
            "Use Duplicate above to make a variation, change a step, then compare them here.",
        )
    else:
        st.caption(
            "Each pipeline applied to the same data. Shape and collinearity are cheap to compare; "
            "which one actually predicts better is a question for the Analysis page."
        )
        target_series = frame[objective.target] if objective.target in frame.columns else None
        features = frame.drop(columns=[objective.target], errors="ignore")
        rows_out = []
        for name in state.pipeline_names:
            candidate = state.pipelines[name]
            try:
                kept, _ = candidate.apply_row_steps(features)
                aligned = target_series.loc[kept.index] if target_series is not None else None
                built = candidate.build_sklearn_pipeline()
                result = built.fit_transform(kept, aligned) if built is not None else kept
                numeric = result.select_dtypes(include="number")
                worst_vif = "—"
                if numeric.shape[1] >= 3:
                    from dsai.statistics.descriptive import variance_inflation_factors

                    table = variance_inflation_factors(numeric)
                    if not table.empty:
                        worst_vif = f"{table.iloc[0]['vif']:.1f}"
                rows_out.append({
                    "Pipeline": name + ("  (active)" if name == state.active_pipeline else ""),
                    "Steps": len(candidate.active_steps),
                    "Rows": f"{len(result):,}",
                    "Columns": result.shape[1],
                    "Worst VIF": worst_vif,
                    "Fitted in fold": len(candidate.leakage_report()["fitted_inside_cross_validation"]),
                })
            except Exception as exc:
                rows_out.append({
                    "Pipeline": name, "Steps": len(candidate.active_steps),
                    "Rows": "failed", "Columns": "—", "Worst VIF": "—",
                    "Fitted in fold": "—",
                })
                st.error(f"'{name}' could not be applied: {exc}")
        dataframe(pd.DataFrame(rows_out))
        caveat(
            "A lower column count or VIF is not automatically better — dropping information can "
            "cost accuracy. Run the Analysis page with each pipeline to find out which wins."
        )

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
                state.add_pipeline(chosen, project.load_pipeline(chosen))
                state.notify("success", f"Loaded pipeline '{chosen}'.")
                st.rerun()

if pipeline is not None and pipeline.history:
    with st.expander("Pipeline history"):
        for entry in reversed(pipeline.history[-30:]):
            st.markdown(f"- `{entry.timestamp}` **{entry.action}** — {entry.detail}  *(by {entry.by})*")
