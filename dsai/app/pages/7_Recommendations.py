"""Recommendations page: what to do, with the evidence behind it."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, error_state, lineage_view, page_header, rank_item,
    recommendation_card, require_run, show_notices, sidebar_chrome, workflow_nav,
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
    error_state(
        "These recommendations did not pass validation",
        " ".join(run.self_check.blocking),
        "Treat everything below as provisional. The checks that failed are listed in full on the "
        "**Models** page under *Decision log*, and the **Report** carries them into every export.",
    )

_impact = {"high": ("high impact", "accent"), "medium": ("medium impact", "neutral"),
           "low": ("low impact", "neutral")}

ai_panel(
    f"I have **{len(run.recommendations)} recommendation(s)**, each traced back to a specific "
    "analytical output. Risks come first because acting on the rest before clearing them is how "
    "an analysis does damage.",
    heading="AI Analyst · recommended actions",
    why="Nothing here is an opinion the platform formed without evidence. Every action below "
        "names the finding it rests on; where the evidence is weak, the confidence says so rather "
        "than the wording hiding it.",
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
        for index, recommendation in enumerate(grouped[heading], start=1):
            marks = [(recommendation.confidence.value,
                      "accent" if recommendation.confidence.value == "high" else "neutral")]
            impact = str(getattr(recommendation, "expected_impact", "") or "").lower()
            for level, mark in _impact.items():
                if level in impact:
                    marks.append(mark)
                    break
            if recommendation.id:
                marks.append((recommendation.id, "neutral"))
            rank_item(
                index, recommendation.action,
                body="**Why.** " + recommendation.reason
                     + ("".join(f"  \n**Evidence.** {e}" for e in recommendation.evidence[:2]))
                     + (f"  \n**Expected impact.** {recommendation.expected_impact}"
                        if recommendation.expected_impact else "")
                     + ("".join(f"  \n**Limit.** {c}" for c in recommendation.caveats[:2])
                        if getattr(recommendation, "caveats", None) else ""),
                badges=marks,
                lead=index == 1 and heading == present[0],
            )
        if run.ledger is not None:
            with st.expander("Where these come from"):
                st.caption(
                    "Each action traced back through the findings it rests on, to the evidence, "
                    "the run, the model, the pipeline and the dataset."
                )
                for recommendation in grouped[heading]:
                    if not recommendation.id:
                        continue
                    st.markdown(f"**{recommendation.id}** — {recommendation.action}")
                    lineage_view(run.ledger.chain(recommendation.id))

        with st.expander("The same recommendations as cards"):
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
