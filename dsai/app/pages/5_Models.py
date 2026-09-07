"""Models page: the tournament, the selected model, its explanation and diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, dataframe, decision_panel, error_state, inference,
    leaderboard, metric_row, page_header, rank_item, require_run, show_notices, sidebar_chrome,
    workflow_nav,
)
from dsai.app.state import scientist, workspace
from dsai.reporting.model_card import build_model_card
from dsai.core.schema import TaskType
from dsai.engines import metrics as M
from dsai.viz import plots

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
st.subheader("Leaderboard")
ai_panel(
    f"**{len(tournament.table)} model(s)** trained against the same held-out rows and the same "
    f"validation split, ranked on **{tournament.primary_metric}**. "
    + (f"**{run.best.model_name}** came out in front — best-performing *on this dataset under "
       "this validation strategy*, which is not the same as best in general."
       if run.best else "No model finished successfully."),
    heading="AI Analyst · model comparison",
    why="The ranking is composite, not a single score. A model that wins on the headline metric "
        "but generalises badly is ranked below one that is marginally worse and stable — because "
        "the first one will not hold up.",
)

table = pd.DataFrame(tournament.table)
_numeric = {c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])}
leaderboard(
    [
        {("#" if k == "rank" else k): (f"{v:,.4g}" if isinstance(v, float) else v)
         for k, v in row.items()}
        for row in tournament.table
    ],
    columns=[("#" if c == "rank" else c) for c in table.columns],
    lead_index=0,
    numeric={("#" if c == "rank" else c) for c in _numeric},
)
with st.expander("As a sortable grid"):
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
    chart(figure, key="tournament_chart")

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
    ["Performance", "Model card", "Explanation", "Diagnostics", "Why this row?", "More data?",
     "Hyper-parameters", "Decision log", "Charts"]
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
    # A trained model outlives the conversation that produced it. The card is
    # what lets someone decide six months later whether to use it.
    card = build_model_card(run, best, frame=state.typed_frame)
    st.caption(
        "Generated from the run itself, so it cannot drift away from what the model actually is. "
        "The section worth reading twice is *inappropriate uses* — every card lists what a model "
        "is good at; models fail where nobody thought to look."
    )
    st.markdown(card.to_markdown())
    st.download_button(
        "Model card (.md)", card.to_markdown(),
        file_name=f"{run.dataset_name}_{best.model_key}_model_card.md",
        mime="text/markdown",
    )

with detail_tabs[2]:
    if run.explanation is None:
        st.info("No explanation was generated for this model type.")
    else:
        st.markdown(f"**Method:** {run.explanation.method}")
        st.caption(run.explanation.method_note)
        figure = plots.feature_importance(run.explanation, mode=theme)
        if figure is not None:
            chart(figure, key="explain_importance")
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

with detail_tabs[3]:
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
                chart(plots.predicted_vs_actual(actual, predicted, mode=theme),
                      key="diag_pva", container=columns[0])
                chart(plots.residual_plot(actual, predicted, mode=theme),
                      key="diag_resid", container=columns[1])
        elif best.task_type.is_classification:
            if run.diagnostics.get("confusion"):
                chart(plots.confusion_matrix(run.diagnostics["confusion"], mode=theme),
                      key="diag_confusion")
            if run.diagnostics.get("per_class"):
                dataframe(pd.DataFrame(run.diagnostics["per_class"]))
            scores = best.extras.get("holdout_score")
            positive = best.extras.get("holdout_positive_label")
            if scores and positive is not None:
                binary = [1 if str(a) == positive else 0 for a in best.extras["holdout_actual"]]
                columns = st.columns(2)
                chart(plots.roc_curve(binary, scores, mode=theme),
                      key="diag_roc", container=columns[0])
                chart(plots.precision_recall_curve(binary, scores, mode=theme),
                      key="diag_pr", container=columns[1])
            if run.diagnostics.get("calibration", {}).get("supported"):
                calibration = run.diagnostics["calibration"]
                st.markdown("**Probability calibration**")
                inference(calibration["interpretation"], label="Calibration")
                dataframe(pd.DataFrame(calibration["bins"]))

with detail_tabs[4]:
    st.caption(
        "Why did the model give one particular row the answer it did? Useful when someone "
        "disputes a prediction, and the fastest way to catch a model relying on something absurd."
    )
    engine = scientist().engine_for(run)
    model = scientist().fitted_model(run)
    if engine is None or model is None:
        st.info("The fitted model is no longer in memory. Re-run the analysis to inspect rows.")
    else:
        X, y, _ = engine.prepare(run.pipeline)
        if len(X) == 0:
            st.info("No rows available.")
        else:
            position = st.number_input(
                "Row number", 0, len(X) - 1, 0,
                help=f"0 to {len(X) - 1}, in the order the model saw them.",
            )
            if st.button("Explain this row"):
                from dsai.explain.importance import explain_prediction

                with st.spinner("Working out what drove this prediction…"):
                    result = explain_prediction(model, X, int(position))
                if not result.get("supported"):
                    st.warning(result.get("reason", "This model does not support row explanation."))
                else:
                    metric_row([
                        ("Prediction", f"{result['prediction']:,.4g}", "For this row"),
                        ("Typical prediction", f"{result.get('baseline', 0):,.4g}",
                         "Average across the dataset"),
                        ("Method", result["method"], ""),
                    ])
                    st.markdown(result["narrative"])
                    contributions = result.get("contributions", [])
                    if contributions:
                        dataframe(pd.DataFrame(contributions).round(5))
                    st.caption(result.get("method_note", ""))
                    st.markdown("**The row itself**")
                    dataframe(X.iloc[[int(position)]])

with detail_tabs[5]:
    st.caption(
        "Whether collecting more of the same data would help, or whether the limit is the "
        "information in the features. These need different responses and are easy to confuse."
    )
    engine = scientist().engine_for(run)
    model = scientist().fitted_model(run)
    if engine is None or model is None or not best.task_type.is_supervised:
        st.info("Available after a supervised run, while the fitted model is still in memory.")
    elif st.button("Compute the learning curve", help="Refits the model on progressively larger samples."):
        from dsai.explain.diagnostics import learning_curve_data

        X, y, _ = engine.prepare(run.pipeline)
        scoring = "r2" if best.task_type is TaskType.REGRESSION else "balanced_accuracy"
        with st.spinner("Refitting on progressively larger samples…"):
            curve = learning_curve_data(model, X, y, scoring=scoring, cv=3)
        if not curve.get("supported"):
            st.warning(curve.get("reason", "Could not compute a learning curve."))
        else:
            st.line_chart(
                pd.DataFrame(
                    {"training rows": curve["train_sizes"],
                     "training score": curve["train_scores"],
                     "held-out score": curve["test_scores"]}
                ).set_index("training rows")
            )
            inference(curve["verdict"], label="What this means")
            metric_row([
                ("Final gap", f"{curve['final_gap']:.4f}", "Training minus held-out"),
                ("Gain from more data", f"{curve['total_improvement']:+.4f}",
                 "Held-out score, smallest to largest sample"),
                ("Scored on", curve["scoring"], ""),
            ])

with detail_tabs[6]:
    if best.hyperparameters:
        dataframe(pd.DataFrame([
            {"Parameter": k, "Value": str(v)} for k, v in best.hyperparameters.items()
        ]))
    else:
        st.info("This model has no tunable parameters, or defaults were used.")
    st.markdown("**Preprocessing applied**")
    for i, step in enumerate(best.preprocessing, 1):
        st.markdown(f"{i}. {step}")

with detail_tabs[7]:
    decision_panel(run.decisions, stage="model_recommendation")
    decision_panel(run.decisions, stage="model_selection")

with detail_tabs[8]:
    from dsai.viz.recommender import explain_chart_choice, recommend_charts

    frame = state.typed_frame if state.typed_frame is not None else state.frame
    specs = recommend_charts(run.profile, run.objective, run)
    st.caption(explain_chart_choice(specs))
    for index, spec in enumerate(specs):
        figure = plots.render(spec, frame, run, mode=theme)
        if figure is not None:
            chart(figure, key=f"chart_{index}_{spec.kind}")
            st.caption(spec.reason)
