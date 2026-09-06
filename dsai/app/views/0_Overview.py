"""Overview: what the platform does and how to work with it."""

from __future__ import annotations

import streamlit as st

import datetime as _dt

from dsai.app.components import (
    ai_panel, apply_theme, caveat, metric_row, page_header, quality_bars, sidebar_chrome,
)
from dsai.app.quality import quality_breakdown
from dsai.app.state import workspace
from dsai.preprocessing.steps import STEPS
from dsai.registry.base import REGISTRY, load_builtin_models

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
load_builtin_models()

# --------------------------------------------------------------------------
# With a project open this page is a dashboard: the state of the work, and the
# one thing worth doing next. Without one it is the front door. The same page
# serving both keeps the app from having a landing screen that goes stale the
# moment anything is loaded.
# --------------------------------------------------------------------------
if state.has_data:
    profile = state.profile
    run = state.run
    hour = _dt.datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    page_header(state.dataset_name or "Untitled project", "", greeting)

    figures: list[tuple[str, str, str]] = [
        ("Rows", f"{profile.n_rows:,}", f"{profile.n_columns} variables"),
        ("Data quality", f"{profile.quality_score:.0f}/100",
         f"{len(profile.quality_issues)} issue(s) flagged" if profile.quality_issues
         else "nothing flagged"),
    ]
    if run is not None and run.tournament is not None:
        figures.append(("Models tested", str(len(run.results)),
                        f"ranked on {run.plan.primary_metric}" if run.plan else ""))
        if run.best is not None:
            figures.append(("Best on this dataset", run.best.model_name,
                            "under the validation strategy used"))
    else:
        figures.append(("Models tested", "—", "no analysis run yet"))
        figures.append(("Questions found", str(len(state.objectives)),
                        "detected from the data"))
    metric_row(figures)

    # What the platform makes of where things stand, and the single next move.
    if run is None:
        lead = state.objectives[0] if state.objectives else None
        ai_panel(
            f"**{state.dataset_name}** is loaded and profiled. "
            + (f"The strongest question it can answer is **{lead.label().replace('_', ' ')}**. "
               if lead else "")
            + "Nothing has been analysed yet — the plan is shown in full before anything runs.",
            why=(lead.rationale if lead else ""),
            evidence=[
                f"{profile.n_rows:,} rows × {profile.n_columns} columns",
                f"Data quality {profile.quality_score:.0f}/100",
            ] + ([f"{len(profile.leakage_suspects)} column(s) may leak the answer"]
                 if getattr(profile, "leakage_suspects", None) else []),
            confidence=(lead.confidence if lead else None),
        )
        st.page_link("pages/4_Analysis.py", label="Set up the analysis →")
    else:
        top = run.findings[0] if run.findings else None
        action = run.recommendations[0] if run.recommendations else None
        ai_panel(
            f"I ran **{len(run.results)} model(s)** and kept **{run.best.model_name}** as the "
            f"best-performing on this dataset under the validation strategy used"
            + (f". I found **{len(run.findings)} finding(s)** and **{len(run.recommendations)} "
               f"recommendation(s)**." if run.findings else ".")
            if run.best else "The run finished without a usable model.",
            why=(f"**{top.title}** — {top.detail}" if top else ""),
            evidence=([action.action] if action else []),
            confidence=(top.confidence if top else None),
        )
        links = st.columns([1, 1, 1, 1, 2])
        links[0].page_link("pages/5_Models.py", label="The comparison →")
        links[1].page_link("pages/6_Insights.py", label="Findings →")
        links[2].page_link("pages/7_Recommendations.py", label="What to do →")
        links[3].page_link("pages/9_Report.py", label="Report →")

    # Quality, as bars, because a score with no breakdown is not actionable.
    st.divider()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Data quality")
        quality_bars(quality_breakdown(profile))
        if profile.quality_issues:
            st.caption(f"{len(profile.quality_issues)} issue(s) — see the **Data** page to work through them.")
    with right:
        st.markdown("### Recent activity")
        if state.runs:
            for entry in reversed(state.runs[-5:]):
                st.markdown(
                    f'<div style="font-size:.8125rem;color:var(--ink-2);padding:.3rem 0;'
                    f'border-bottom:1px solid var(--border)">'
                    f'<strong style="color:var(--ink)">{entry.objective.task_type.value.replace("_", " ") if entry.objective else "run"}</strong>'
                    f' · {len(entry.results)} model(s) · {entry.duration_s:.0f}s</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Nothing has run yet. Activity appears here once it has.")
    st.stop()


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
