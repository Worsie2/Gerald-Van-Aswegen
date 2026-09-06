"""Overview: what the platform does and how to work with it."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import apply_theme, caveat, metric_row, page_header, sidebar_chrome
from dsai.app.state import workspace
from dsai.preprocessing.steps import STEPS
from dsai.registry.base import REGISTRY, load_builtin_models

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
load_builtin_models()

page_header(
    "An AI data scientist, not a chatbot about one",
    "Upload data and the platform profiles it, decides what analysis it supports, builds "
    "preprocessing suited to each candidate model, trains and cross-validates a field of them, "
    "compares them on more than the headline score, explains the winner, checks its own "
    "conclusions, and tells you what to do — with the evidence attached to every claim.",
    "Workspace",
)

stats = REGISTRY.stats()
metric_row([
    ("Algorithms", f"{stats['total']}", f"{stats['available']} available in this environment"),
    ("Preprocessing steps", f"{len(STEPS)}", "Leakage-controlled by construction"),
    ("Statistical tests", "15+", "Each with assumptions and effect size"),
    ("Export formats", "6", "Including runnable Python"),
])

left, right = st.columns(2, gap="large")

with left:
    st.markdown("## The workflow")
    st.markdown(
        """
1. **Data** — load it; every variable is profiled and quality problems flagged
2. **Context** — tell it what the data means; kept separate from what was measured
3. **Preprocessing** — review, edit or build the pipeline; leakage prevented structurally
4. **Analysis** — choose the objective, see the plan, run it
5. **Models** — the tournament, the selected model, its explanation and diagnostics
6. **Insights** — findings, grouped by the kind of evidence behind them
7. **Recommendations** — what to do, traced back to the analysis
        """
    )
    st.caption("Also: Ask for plain-language questions · Report to export · Projects to save work.")

with right:
    st.markdown("## Three ways to work")
    st.markdown(
        """
**AI automatic** — *"Do the analysis for me."* It runs the whole workflow and reports what
it found and what it decided.

**AI assisted** — *"Recommend what I should do, then let me approve it."* It plans, you
adjust, then it runs.

**Manual / expert** — *"I know what I want."* Full control of the objective, variables,
pipeline, algorithms, hyper-parameters, validation strategy and metrics.
        """
    )
    st.caption("All three use the same engine. Only the amount you decide changes.")

st.markdown("## What it will not do")
for limit in [
    "Claim a model is best in general. It reports the best-performing model *for this dataset "
    "under the validation strategy used*, and names the simplest acceptable alternative alongside it.",
    "Present a correlation as a cause. Every model-derived finding is labelled as an association, "
    "with the caveat attached rather than buried.",
    "Hide a weak result. If nothing beats a model that ignores every predictor, it says so "
    "plainly instead of presenting the least-bad option.",
    "Let preprocessing leak. Anything that learns from the data is fitted inside each "
    "cross-validation fold, never on the full dataset.",
    "Mix your assumptions with its measurements. What you told it is reported separately from "
    "what it found.",
]:
    caveat(limit, label="Never")

if not state.has_data:
    st.divider()
    st.markdown(
        '<div class="dsai-decide"><span class="dsai-decide-label">Start here</span>'
        "Open the <strong>Data</strong> page in the sidebar. There are six sample datasets with "
        "deliberately known structure if you would rather look around before using your own — you "
        "already know the right answer, so you can judge whether to trust the platform elsewhere."
        "</div>",
        unsafe_allow_html=True,
    )
