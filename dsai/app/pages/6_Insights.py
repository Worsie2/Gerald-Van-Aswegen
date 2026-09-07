"""Insights page: findings, grouped by what kind of evidence backs them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, dataframe, finding_card, inference, known_unknowns,
    lineage_view, metric_row, page_header, rank_item, require_run, show_notices, sidebar_chrome,
    trust_panel, workflow_nav,
)
from dsai.app.state import workspace
from dsai.core.schema import EvidenceKind
from dsai.engines.insight import group_by_evidence
from dsai.viz import plots

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Findings",
    "Grouped by the kind of evidence behind each one. What was measured, what a test established, "
    "what a model inferred and what you assumed are kept apart on purpose.",
    "Step 6 of 7",
)
workflow_nav("Insights", state)

if not require_run(state):
    st.stop()

run = state.run
frame = state.typed_frame if state.typed_frame is not None else state.frame

if not run.findings:
    st.info("No findings were generated.")
    st.stop()

grouped = group_by_evidence(run.findings)
order = [
    "Observed in the data",
    "Established by a statistical test",
    "Derived from a model",
    "Platform interpretation",
    "Your assumptions (not verified)",
]

if run.trust is not None:
    st.markdown("### Why trust these findings?")
    trust_panel(run.trust)
    known_unknowns(run.trust)
    st.divider()

ai_panel(
    f"I found **{len(run.findings)} finding(s)** in this analysis. They are ordered by how strong "
    "the evidence behind each one is — what was measured first, what a statistical test "
    "established next, then what a model inferred, then interpretation, then anything you told me "
    "that I did not check.",
    heading="AI Analyst · key insights",
    why="Keeping those apart is the point. A number measured in your data and a conclusion drawn "
        "from a model are both true statements, but they are not the same kind of true, and acting "
        "on them carries different risk.",
)

_EVIDENCE_NOTE = {
    "Observed in the data": "Counted or computed directly. No model, no test, no inference.",
    "Established by a statistical test": "A test was run, its assumptions checked, and the effect "
                                         "size reported alongside the p-value.",
    "Derived from a model": "What a fitted model relies on. This is association, not cause.",
    "Platform interpretation": "The platform's reading of the above. Judgement, marked as such.",
    "Your assumptions (not verified)": "These came from you and were not checked against the data. "
                                        "Everything above that leans on them inherits them.",
}

_position = 0
read_tab, cards_tab = st.tabs(["As a report", "As cards"])
with read_tab:
    for heading in order:
        findings = grouped.get(heading)
        if not findings:
            continue
        st.markdown(f"### {heading}")
        st.caption(_EVIDENCE_NOTE.get(heading, ""))
        for finding in findings:
            _position += 1
            marks = [(finding.confidence.value, "accent" if finding.confidence.value == "high"
                      else "neutral")]
            if finding.caveats:
                marks.append(("has caveats", "warning"))
            if finding.id:
                marks.append((finding.id, "neutral"))
            rank_item(
                _position, finding.title,
                body=finding.detail
                     + ("".join(f"  \n**Evidence.** {e}" for e in finding.evidence[:2]))
                     + ("".join(f"  \n**Limit.** {c}" for c in finding.caveats[:2])),
                badges=marks,
                lead=_position == 1,
            )
            if run.ledger is not None and finding.id:
                supporting = run.ledger.evidence_for(finding.id)
                if supporting:
                    with st.expander(f"Where {finding.id} comes from"):
                        st.caption(
                            "Every number behind this claim, with its kind, and the run, model, "
                            "pipeline and dataset it came from. The identifiers are derived from "
                            "the content, so they stay the same across runs and exports."
                        )
                        lineage_view(
                            [{"level": "Finding", "id": finding.id, "label": finding.title,
                              "kind": ""}]
                            + [{"level": "Evidence", "id": e.id, "label": e.statement,
                                "kind": e.kind.value.replace("_", " ")} for e in supporting]
                            + [{"level": "Run", "id": run.id, "label": run.ledger.fingerprint,
                                "kind": ""},
                               {"level": "Model", "id": "", "label": run.ledger.model, "kind": ""},
                               {"level": "Pipeline", "id": "", "label": run.ledger.pipeline,
                                "kind": ""},
                               {"level": "Dataset", "id": "", "label": run.ledger.dataset,
                                "kind": ""}]
                        )

with cards_tab:
    for heading in order:
        findings = grouped.get(heading)
        if not findings:
            continue
        st.subheader(heading)
        if heading.startswith("Your assumptions"):
            st.caption("These came from you and were not checked against the data. Everything "
                       "below inherits them.")
        for index, finding in enumerate(findings):
            finding_card(finding, expanded=index == 0 and heading != "Your assumptions (not verified)")

st.divider()

# --------------------------------------------------------------------------
# segments, series, associations
# --------------------------------------------------------------------------
if run.segmentation is not None and getattr(run.segmentation, "clusters", None):
    st.subheader("Segments")
    for cluster in run.segmentation.clusters:
        with st.container(border=True):
            st.markdown(f"**{cluster.name}** — {cluster.size:,} rows ({cluster.share:.1%})")
            st.write(cluster.description)
            if cluster.defining_features:
                dataframe(pd.DataFrame(cluster.defining_features))
    figure = plots.cluster_profile(run.segmentation, mode=state.theme)
    if figure is not None:
        chart(figure, key="insight_cluster_profile")
    if run.segmentation.separating_features:
        with st.expander("Which variables separate the segments"):
            dataframe(pd.DataFrame(run.segmentation.separating_features))

if run.series_analysis:
    st.subheader("Time-series structure")
    analysis = run.series_analysis
    columns = st.columns(3)
    trend = analysis.get("trend", {})
    columns[0].metric("Trend", trend.get("direction", "—"),
                      help=trend.get("interpretation", ""))
    seasonality = analysis.get("seasonality", {})
    columns[1].metric("Seasonality",
                      "detected" if seasonality.get("detected") else "none found",
                      help=seasonality.get("interpretation", seasonality.get("reason", "")))
    stationarity = analysis.get("stationarity", {})
    columns[2].metric("Stationarity", stationarity.get("verdict", "—"),
                      help=stationarity.get("recommended_action", ""))
    for finding in analysis.get("findings", []):
        st.markdown(f"- {finding}")
    for issue in analysis.get("issues", []):
        caveat(issue)
    if run.best is not None:
        figure = plots.forecast_plot(run.best, mode=state.theme,
                                     target_name=run.objective.target or "value")
        if figure is not None:
            chart(figure, key="insight_forecast")
    decomposition = analysis.get("decomposition", {})
    if decomposition.get("supported"):
        figure = plots.decomposition_plot(decomposition, mode=state.theme)
        if figure is not None:
            chart(figure, key="insight_decomposition")

if run.associations is not None and run.associations.rules:
    st.subheader("Association rules")
    st.caption(
        f"{run.associations.shape.n_transactions:,} transactions, "
        f"{run.associations.shape.n_items} distinct items, average basket "
        f"{run.associations.shape.mean_basket_size}."
    )
    for rule in run.associations.rules[:15]:
        st.markdown(f"- {rule['statement']}")
    dataframe(pd.DataFrame([
        {
            "If": " + ".join(r["if"]), "Then": " + ".join(r["then"]),
            "Support": r["support"], "Confidence": r["confidence"], "Lift": r["lift"],
            "Transactions": r["n_transactions"],
        }
        for r in run.associations.rules
    ]))

# --------------------------------------------------------------------------
# anomalies, cluster sweep, hypothesis tests, exploration
# --------------------------------------------------------------------------
if run.anomalies:
    st.subheader("Rows that look unlike the rest")
    st.caption(
        f"{len(run.anomalies)} row(s) flagged, ordered by how unusual they are. Each carries the "
        "reason it stood out. Unusual is not the same as wrong — these are rows worth a human "
        "looking at, not rows to delete."
    )
    dataframe(pd.DataFrame(run.anomalies))
    caveat(
        "There are no labels here, so 'anomalous' means 'unlike the rest of this data'. Whether "
        "a flagged row is a problem, a data-entry error or your best customer is your judgement."
    )
    st.divider()

if run.cluster_sweep and run.cluster_sweep.get("supported"):
    st.subheader("How many segments?")
    sweep = run.cluster_sweep
    metric_row([
        ("Recommended", str(sweep["recommended_k"]), sweep["agreement"] + " agree"),
        ("Elbow", str(sweep["elbow_k"] or "—"), "Where added segments stop paying"),
        ("Best silhouette", str(sweep["best_silhouette_k"]), "Separation between segments"),
        ("Best Davies-Bouldin", str(sweep["best_davies_bouldin_k"]), "Lower is better"),
    ])
    inference(sweep["interpretation"], label="Cluster count")
    dataframe(pd.DataFrame(sweep["table"]).round(4))
    st.divider()

if run.tests:
    st.subheader("Statistical comparisons")
    st.caption(
        "Each comparison checked its own assumptions, was cross-checked against the "
        "non-parametric equivalent, and the p-values are adjusted for the number of tests run."
    )
    dataframe(pd.DataFrame([
        {
            "Grouped by": t["group"],
            "Test": t["primary"].test,
            "p-value": round(t["primary"].p_value, 6),
            "Adjusted p": round(t["adjusted_p"], 6),
            "Significant": "yes" if t["significant_after_correction"] else "no",
            "Effect size": round(t["primary"].effect_size or 0, 4),
            "Effect": t["primary"].effect_interpretation,
            "Cross-check agrees": "yes" if t["primary"].significant == t["secondary"].significant else "no",
        }
        for t in run.tests
    ]))
    caveat(
        "A significant p-value says a difference is unlikely to be chance. The effect size says "
        "whether it is big enough to act on. They are different questions and both are shown."
    )
    st.divider()

if run.exploration.get("outliers"):
    st.subheader("Distribution and outliers")
    dataframe(pd.DataFrame(run.exploration["outliers"]).round(3))
    st.divider()

if run.exploration.get("correlations"):
    st.subheader("Relationships between variables")
    dataframe(pd.DataFrame(run.exploration["correlations"]).round(4))
    figure = plots.correlation_heatmap(frame, run.profile.numeric_columns, mode=state.theme)
    if figure is not None:
        chart(figure, key="insight_corr")
    caveat("Correlation is not causation.")
    st.divider()

st.divider()

# --------------------------------------------------------------------------
# validation checks
# --------------------------------------------------------------------------
st.subheader("Validation checks")
st.caption(
    "The questions the platform asks itself before presenting anything. A failed check does not "
    "invalidate a finding — it limits what can be claimed from it."
)
if run.self_check is None:
    st.info("No validation checks were run.")
else:
    for check in run.self_check.checks:
        icon = "✓" if check.passed else ("▲" if check.severity == "blocking" else "△")
        with st.expander(f"{icon}  {check.question}", expanded=not check.passed and check.severity == "blocking"):
            st.write(check.detail or "Passed.")
            if not check.passed and check.downgrade_steps:
                st.caption(f"Confidence in the conclusions was lowered by {check.downgrade_steps} level(s) as a result.")

st.divider()
st.markdown("Continue to **Recommendations** for what to do about all this.")
