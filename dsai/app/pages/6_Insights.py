"""Insights page: findings, grouped by what kind of evidence backs them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import (
    apply_theme, caveat, sidebar_chrome, dataframe, finding_card, page_header, require_run, show_notices, workflow_nav,
)
from dsai.app.state import workspace
from dsai.core.schema import EvidenceKind
from dsai.engines.insight import group_by_evidence
from dsai.viz import plots

st.set_page_config(page_title="Insights · DSAI", page_icon="🔎", layout="wide")
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
for heading in order:
    findings = grouped.get(heading)
    if not findings:
        continue
    st.subheader(heading)
    if heading.startswith("Your assumptions"):
        st.caption("These came from you and were not checked against the data. Everything below inherits them.")
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
        st.plotly_chart(figure, use_container_width=True, key="insight_cluster_profile")
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
            st.plotly_chart(figure, use_container_width=True, key="insight_forecast")
    decomposition = analysis.get("decomposition", {})
    if decomposition.get("supported"):
        figure = plots.decomposition_plot(decomposition, mode=state.theme)
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True, key="insight_decomposition")

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
