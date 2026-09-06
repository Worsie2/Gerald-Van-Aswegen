"""Models page: the tournament, the selected model, its explanation and diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import (
    apply_theme, caveat, sidebar_chrome, inference, dataframe, decision_panel, metric_row, page_header, require_run, show_notices, workflow_nav,
)
from dsai.app.state import scientist, workspace
from dsai.core.schema import TaskType
from dsai.engines import metrics as M
from dsai.viz import plots

st.set_page_config(page_title="Models · DSAI", page_icon="🏁", layout="wide")
state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Model tournament",
    "Every candidate, scored the same way. The highest raw score does not win automatically.",
    "Step 5 of 7",
)
workflow_nav("Models", state)

if not require_run(state):
    st.stop()

run = state.run
tournament = run.tournament
theme = state.theme

if tournament is None or not tournament.ranked:
    st.warning("No model produced a usable result. Check the analysis trace and data-quality issues.")
    for warning in run.warnings:
        st.warning(warning)
    st.stop()

# --------------------------------------------------------------------------
# the four named choices
# --------------------------------------------------------------------------
st.subheader("The platform's four answers")
st.caption(
    "Not one verdict but four, because 'best' depends on what you need: accuracy, explainability, "
    "or simply not being beaten by doing nothing."
)
columns = st.columns(4)
choices = [
    ("Recommended", tournament.recommended, "Best overall once generalisation, stability, interpretability and cost are weighed"),
    ("Highest score", tournament.highest_performance, "Best on the headline metric alone"),
    ("Simplest acceptable", tournament.simplest_acceptable, "Within 5% of the best, and far easier to explain"),
    ("Baseline", tournament.baseline, "Ignores every predictor — the score any real model must beat"),
]
for column, (label, model, help_text) in zip(columns, choices):
    with column:
        if model is None:
            st.metric(label, "—", help=help_text)
        else:
            st.metric(
                label,
                M.format_metric(tournament.primary_metric, model.primary_score),
                help=help_text,
            )
            st.caption(model.model_name)

for warning in tournament.warnings:
    caveat(warning)

# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------
st.divider()
st.subheader("Full comparison")
table = pd.DataFrame(tournament.table)
dataframe(table)
st.caption(
    f"Composite weights — performance {tournament.weights['performance']:.0%}, "
    f"generalisation {tournament.weights['generalisation']:.0%}, "
    f"stability {tournament.weights['stability']:.0%}, "
    f"interpretability {tournament.weights['interpretability']:.0%}, "
    f"training cost {tournament.weights['efficiency']:.0%}."
)

figure = plots.model_comparison(tournament, mode=theme)
if figure is not None:
    st.plotly_chart(figure, use_container_width=True, key="tournament_chart")

# --------------------------------------------------------------------------
# selected model detail
# --------------------------------------------------------------------------
st.divider()
best = run.best
if best is None:
    st.stop()

st.subheader(f"Selected model — {best.model_name}")
metrics_to_show = M.default_metrics(best.task_type)[:4]
metric_row([
    (M.METRIC_LABELS.get(m, m), M.format_metric(m, best.primary(m)), M.explain_metric(m))
    for m in metrics_to_show
])

detail_tabs = st.tabs(
    ["Performance", "Explanation", "Diagnostics", "Hyper-parameters", "Decision log", "Charts"]
)

with detail_tabs[0]:
    rows = []
    for metric in M.default_metrics(best.task_type):
        rows.append({
            "Metric": M.METRIC_LABELS.get(metric, metric),
            "Cross-validated": M.format_metric(metric, best.validation.mean_scores.get(metric)),
            "± std": M.format_metric(metric, best.validation.std_scores.get(metric)),
            "Hold-out test": M.format_metric(metric, best.test_scores.get(metric)),
            "Training": M.format_metric(metric, best.train_scores.get(metric)),
            "What it means": M.explain_metric(metric),
        })
    dataframe(pd.DataFrame(rows))
    st.caption(
        f"Trained on {best.n_train:,} rows, tested on {best.n_test:,} held out. "
        f"{best.validation.n_splits} validation fold(s), seed {best.random_seed}."
    )
    if best.overfitting_gap is not None:
        st.metric("Train-to-held-out gap", f"{best.overfitting_gap:.4f}",
                  help="Large gaps mean the training figure overstates real-world performance.")
    for warning in best.warnings:
        caveat(warning)

with detail_tabs[1]:
    if run.explanation is None:
        st.info("No explanation was generated for this model type.")
    else:
        st.markdown(f"**Method:** {run.explanation.method}")
        st.caption(run.explanation.method_note)
        figure = plots.feature_importance(run.explanation, mode=theme)
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True, key="explain_importance")
        for line in run.explanation.plain_english:
            st.markdown(f"- {line}")
        for note in run.explanation.caveats:
            caveat(note)

        if run.explanation.coefficients:
            with st.expander("Coefficients"):
                dataframe(pd.DataFrame(run.explanation.coefficients).round(6))

        st.divider()
        st.markdown("**How the prediction changes with one variable**")
        features = [f.feature for f in run.explanation.importances[:10]]
        chosen = st.selectbox("Variable", features)
        model = scientist().fitted_model(run)
        if model is not None and chosen:
            from dsai.explain.importance import partial_dependence_curve

            engine = scientist().engine_for(run)
            X, y, _ = engine.prepare(run.pipeline)
            curve = partial_dependence_curve(model, X, chosen) if chosen in X.columns else \
                {"supported": False, "reason": f"'{chosen}' is created during preprocessing, so it cannot be swept directly."}
            if curve.get("supported"):
                st.line_chart(pd.DataFrame({chosen: curve["grid"], "predicted": curve["predictions"]})
                              .set_index(chosen))
                inference(curve["interpretation"], label="Partial dependence")
            else:
                st.info(curve.get("reason", "Not available for this variable."))

with detail_tabs[2]:
    if not run.diagnostics or not run.diagnostics.get("usable", True):
        st.info("No diagnostics available for this model type.")
    else:
        st.markdown(run.diagnostics.get("interpretation", ""))
        for issue in run.diagnostics.get("issues", []):
            caveat(issue)
        if best.task_type is TaskType.REGRESSION:
            actual = best.extras.get("holdout_actual")
            predicted = best.extras.get("holdout_predicted")
            if actual and predicted:
                columns = st.columns(2)
                columns[0].plotly_chart(plots.predicted_vs_actual(actual, predicted, mode=theme),
                                        use_container_width=True, key="diag_pva")
                columns[1].plotly_chart(plots.residual_plot(actual, predicted, mode=theme),
                                        use_container_width=True, key="diag_resid")
        elif best.task_type.is_classification:
            if run.diagnostics.get("confusion"):
                st.plotly_chart(plots.confusion_matrix(run.diagnostics["confusion"], mode=theme),
                                use_container_width=True, key="diag_confusion")
            if run.diagnostics.get("per_class"):
                dataframe(pd.DataFrame(run.diagnostics["per_class"]))
            scores = best.extras.get("holdout_score")
            positive = best.extras.get("holdout_positive_label")
            if scores and positive is not None:
                binary = [1 if str(a) == positive else 0 for a in best.extras["holdout_actual"]]
                columns = st.columns(2)
                columns[0].plotly_chart(plots.roc_curve(binary, scores, mode=theme),
                                        use_container_width=True, key="diag_roc")
                columns[1].plotly_chart(plots.precision_recall_curve(binary, scores, mode=theme),
                                        use_container_width=True, key="diag_pr")
            if run.diagnostics.get("calibration", {}).get("supported"):
                calibration = run.diagnostics["calibration"]
                st.markdown("**Probability calibration**")
                inference(calibration["interpretation"], label="Calibration")
                dataframe(pd.DataFrame(calibration["bins"]))

with detail_tabs[3]:
    if best.hyperparameters:
        dataframe(pd.DataFrame([
            {"Parameter": k, "Value": str(v)} for k, v in best.hyperparameters.items()
        ]))
    else:
        st.info("This model has no tunable parameters, or defaults were used.")
    st.markdown("**Preprocessing applied**")
    for i, step in enumerate(best.preprocessing, 1):
        st.markdown(f"{i}. {step}")

with detail_tabs[4]:
    decision_panel(run.decisions, stage="model_recommendation")
    decision_panel(run.decisions, stage="model_selection")

with detail_tabs[5]:
    from dsai.viz.recommender import explain_chart_choice, recommend_charts

    frame = state.typed_frame if state.typed_frame is not None else state.frame
    specs = recommend_charts(run.profile, run.objective, run)
    st.caption(explain_chart_choice(specs))
    for index, spec in enumerate(specs):
        figure = plots.render(spec, frame, run, mode=theme)
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True, key=f"chart_{index}_{spec.kind}")
            st.caption(spec.reason)
