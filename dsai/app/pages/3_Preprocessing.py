"""Preprocessing page: review, edit or build the pipeline."""

from __future__ import annotations

import streamlit as st

import pandas as pd

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, dataframe, decision_panel, empty_state, inference,
    metric_row, override_notice, page_header, pipeline_graph, require_data, show_notices,
    sidebar_chrome, workflow_nav,
)
from dsai.app.imputation import (
    STRATEGIES as IMPUTATION_STRATEGIES, Choice, preview_fill, steps_for_choices, strategies_for,
)
from dsai.app.state import workspace
from dsai.viz import plots
from dsai.core.schema import Objective, SemanticType, TaskType
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.preprocessing.recommender import recommend_pipeline
from dsai.preprocessing.steps import STEPS
from dsai.registry.base import REGISTRY



def _suggested_strategy(column, options: list[str]) -> str:
    """What the platform would pick, offered as the default rather than imposed.

    Mirrors the reasoning in the preprocessing recommender: a column that is
    mostly absent is worth removing, a skewed one wants the median, a symmetric
    one can take the mean, and a categorical one takes its commonest value.
    """
    if column.missing_pct > 60 and "Drop the column" in options:
        return "Drop the column"
    if "Median of the column" in options:
        skew = abs(column.skewness) if column.skewness is not None else 0.0
        if skew < 0.5 and "Mean of the column" in options:
            return "Mean of the column"
        return "Median of the column"
    if "Most frequent value" in options:
        return "Most frequent value"
    return options[0]


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
if new.button("Duplicate", disabled=state.pipeline is None, width='stretch',
              help="Copy the active pipeline so you can try a variation without losing this one."):
    state.add_pipeline(f"{state.active_pipeline} copy", state.pipeline.copy())
    state.notify("success", f"Duplicated as '{state.active_pipeline}'.")
    st.rerun()
delete.write("")
if delete.button("Delete", disabled=len(names) < 2, width='stretch'):
    state.remove_pipeline(state.active_pipeline)
    st.rerun()

(build_tab, missing_tab, graph_tab, edit_tab, preview_tab, effect_tab, compare_tab,
 saved_tab) = st.tabs(
    ["Build", "Missing values", "The pipeline", "Edit steps", "Preview effect", "What it changed",
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
    if columns[2].button("Build recommended pipeline", type="primary", width='stretch'):
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

with missing_tab:
    # One decision per column, rather than one strategy applied to a multiselect.
    # Which column has gaps, how many, and what filling them would actually put
    # there are all different questions, and a reader needs all three to choose.
    gappy = [(name, column) for name, column in profile.columns.items() if column.n_missing]
    if not gappy:
        empty_state(
            "No column has a missing value",
            "Nothing to decide here. Imputation steps can still be added by hand on the "
            "**Build** tab if you expect gaps in data you will score later.",
        )
    else:
        ai_panel(
            f"**{len(gappy)} column(s)** have gaps. Below, each one gets its own decision — the "
            "column median, the mean *within a group* like region or tier, a fixed value, or "
            "dropping it. The number each choice would insert is shown beside it, so you can "
            "judge the choice rather than take it on trust.",
            heading="AI Analyst · missing values",
            why="A single strategy across every column is almost never right. A skewed column and "
                "a symmetric one want different statistics, and a column that is mostly absent "
                "usually wants removing rather than inventing.",
        )
        caveat(
            "The values previewed here are computed on the whole dataset so they can be shown. "
            "The pipeline itself refits its imputers inside each cross-validation fold, so the "
            "numbers it actually uses come from training rows only and will differ slightly. "
            "That is the point — it is what stops a test row informing its own imputation."
        )

        grouping_columns = [
            name for name, column in profile.columns.items()
            if column.semantic_type in (SemanticType.CATEGORICAL_NOMINAL,
                                        SemanticType.CATEGORICAL_ORDINAL,
                                        SemanticType.BINARY)
            and 1 < column.n_unique <= 50
        ]

        choices: list[Choice] = []
        for name, column in sorted(gappy, key=lambda item: -item[1].missing_pct):
            with st.container(border=True):
                head, control = st.columns([2, 3])
                with head:
                    st.markdown(
                        f'<div class="dsai-colcard-name">{name}</div>'
                        f'<div class="dsai-meta">{column.semantic_type.value.replace("_", " ")}</div>'
                        f'<div style="font-size:.8125rem;color:var(--ink-2)">'
                        f"<strong>{column.n_missing:,}</strong> missing "
                        f"({column.missing_pct:.1f}% of rows)</div>",
                        unsafe_allow_html=True,
                    )
                    if column.missing_pct > 40:
                        caveat("More than 40% absent. Imputing this invents most of the column.")

                with control:
                    options = strategies_for(column.semantic_type)
                    default = _suggested_strategy(column, options)
                    strategy = st.selectbox(
                        "Fill the gaps with", options, index=options.index(default),
                        key=f"imp_{name}",
                        help=IMPUTATION_STRATEGIES[default]["note"],
                    )
                    spec = IMPUTATION_STRATEGIES[strategy]
                    st.caption(spec["note"])

                    group_by = None
                    fill_value = ""
                    if spec.get("needs_group"):
                        if grouping_columns:
                            group_by = st.selectbox(
                                "Group by", grouping_columns, key=f"impg_{name}",
                                help="The statistic is computed inside each of this column's groups.",
                            )
                        else:
                            caveat("No categorical column with a usable number of groups, so there "
                                   "is nothing to group by.")
                    if spec.get("needs_value"):
                        fill_value = st.text_input(
                            "Value to insert", "__missing__", key=f"impv_{name}",
                            help="Kept as its own level, so a model can learn from the absence.",
                        )

                choice = Choice(name, strategy, group_by, fill_value)
                choices.append(choice)

                summary, table = preview_fill(frame, choice)
                st.markdown(
                    f'<div class="dsai-caveat"><span class="dsai-caveat-label">Would insert</span>'
                    f"{summary}</div>",
                    unsafe_allow_html=True,
                )
                if table is not None and not table.empty:
                    with st.expander(f"The {len(table)} group value(s) this would use"):
                        dataframe(table)

        st.divider()
        planned = steps_for_choices(choices)
        acting = [c for c in choices if c.is_action]
        left, right = st.columns([3, 2])
        left.markdown(
            f"**{len(acting)} of {len(choices)} column(s)** would be changed, as "
            f"**{len(planned)} pipeline step(s)** — identical choices are merged into one step so "
            "the pipeline stays readable."
        )
        if right.button("Add these to the pipeline", type="primary", disabled=not planned,
                        width='stretch'):
            if pipeline is None:
                state.add_pipeline("missing values", PreprocessingPipeline(name="missing values"))
                pipeline = state.pipeline
            for step in planned:
                pipeline.add(step["step_key"], columns=step["columns"], by="user",
                             note="chosen per column on the Missing values tab", **step["params"])
            state.notify("success", f"Added {len(planned)} imputation step(s) to "
                                    f"'{state.active_pipeline}'.")
            st.rerun()

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
            elif param.kind == "categorical":
                # A categorical parameter with no fixed choices names a column,
                # so the dataset's own columns are the choices.
                picked = st.selectbox(
                    param.name, ["— none —"] + list(profile.columns),
                    help=param.description or None, key=f"p_{step_key}_{param.name}",
                )
                params[param.name] = None if picked.startswith("—") else picked
            elif param.kind == "text":
                params[param.name] = st.text_input(
                    param.name, str(param.default or ""),
                    help=param.description or None, key=f"p_{step_key}_{param.name}",
                )
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

        # The cleaned data is a deliverable in its own right — a lot of the value
        # of this page is the tidied dataset, and until now there was no way to
        # get it out of the app.
        st.divider()
        st.markdown("**Take the cleaned data with you**")
        st.caption(
            "The dataset with this pipeline applied, fitted on all the rows. Useful for handing "
            "on, or for checking the transformations against your own knowledge of the data. "
            "Note the difference from what the analysis does: for modelling, these steps are "
            "refitted inside each cross-validation fold, so the numbers used there come from "
            "training rows only."
        )
        export = st.columns([1, 1, 2])
        try:
            cleaned = pipeline.fit_transform_frame(features, target)
        except Exception as exc:
            cleaned = None
            caveat(f"The pipeline could not be applied to the whole dataset: {exc}")
        if cleaned is not None:
            import io as _io

            buffer = _io.StringIO()
            cleaned.to_csv(buffer, index=False)
            export[0].download_button(
                f"Cleaned data (.csv)", buffer.getvalue(),
                file_name=f"{state.dataset_name}_cleaned.csv", mime="text/csv",
                width='stretch',
            )
            try:
                parquet = _io.BytesIO()
                cleaned.to_parquet(parquet, index=False)
                export[1].download_button(
                    "Cleaned data (.parquet)", parquet.getvalue(),
                    file_name=f"{state.dataset_name}_cleaned.parquet",
                    mime="application/octet-stream", width='stretch',
                )
            except Exception:
                export[1].caption("Parquet needs `pyarrow`.")
            export[2].caption(
                f"{len(cleaned):,} rows × {cleaned.shape[1]} columns "
                f"(from {len(features):,} × {features.shape[1]})."
            )

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
        if columns[1].button("Save this pipeline", disabled=pipeline is None, width='stretch'):
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
