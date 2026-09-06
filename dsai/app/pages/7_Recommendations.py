"""Recommendations page: what to do, with the evidence behind it."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    apply_theme, caveat, sidebar_chrome, page_header, recommendation_card, require_run, show_notices, workflow_nav,
)
from dsai.app.state import workspace
from dsai.engines.recommend import group_by_category

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Recommendations",
    "Each one traces back to a specific analytical output. Nothing here is an opinion the platform "
    "formed without evidence.",
    "Step 7 of 7",
)
workflow_nav("Recommendations", state)

if not require_run(state):
    st.stop()

run = state.run

if not run.recommendations:
    st.info("No recommendations were generated.")
    st.stop()

if run.self_check and run.self_check.blocking:
    st.error(
        "**These recommendations did not pass validation.** "
        + " ".join(run.self_check.blocking),
    )

grouped = group_by_category(run.recommendations)
order = ["Risks to address first", "Data quality", "Actions to take", "What to investigate next"]
tabs = st.tabs([f"{h} ({len(grouped.get(h, []))})" for h in order if grouped.get(h)])
present = [h for h in order if grouped.get(h)]

for tab, heading in zip(tabs, present):
    with tab:
        if heading == "Risks to address first":
            st.caption("Deal with these before acting on anything else.")
        elif heading == "What to investigate next":
            st.caption("Where the next analysis would add the most.")
        for index, recommendation in enumerate(grouped[heading]):
            recommendation_card(recommendation, index)

st.divider()
st.subheader("Limitations that apply to all of this")
caveat(
    "This is observational data. Every relationship found is an association — a third variable, "
    "reverse causation or selection into the sample would each produce the same pattern. Acting on "
    "a recommendation assumes the relationship survives intervention, which only a controlled test "
    "can establish. Test on a small group before rolling anything out.",
    label="Applies to all of the above",
)
if run.context.assumptions:
    st.markdown("**Your assumptions, which every recommendation inherits:**")
    for assumption in run.context.assumptions:
        st.markdown(f"- {assumption}")
if run.context.limitations:
    st.markdown("**Limitations you flagged:**")
    for limitation in run.context.limitations:
        st.markdown(f"- {limitation}")

st.divider()
st.markdown("Export the full report from the **Report** page, or ask a follow-up question in **Ask**.")
