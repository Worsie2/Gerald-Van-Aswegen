"""Overview: what the platform does and how to work with it."""

from __future__ import annotations

import streamlit as st

import datetime as _dt

from dsai.app.components import (
    ai_panel, apply_theme, caveat, error_state, known_unknowns, metric_row, page_header,
    quality_bars, sidebar_chrome, status_bar, trust_panel,
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
    status_bar(state)

    # ---------------------------------------------------------------------
    # Nothing analysed yet: the page is about getting there.
    # ---------------------------------------------------------------------
    if run is None:
        lead = state.objectives[0] if state.objectives else None
        metric_row([
            ("Rows", f"{profile.n_rows:,}", f"{profile.n_columns} variables"),
            ("Data quality", f"{profile.quality_score:.0f}/100",
             f"{len(profile.quality_issues)} issue(s) flagged" if profile.quality_issues
             else "nothing flagged"),
            ("Questions found", str(len(state.objectives)), "detected from the data"),
            ("Analysed", "not yet", "the plan is shown before anything runs"),
        ])
        ai_panel(
            f"**{state.dataset_name}** is loaded and profiled. "
            + (f"The strongest question it can answer is **{lead.label().replace('_', ' ')}**. "
               if lead else "")
            + "Nothing has been analysed yet.",
            why=(lead.rationale if lead else ""),
            evidence=[
                f"{profile.n_rows:,} rows × {profile.n_columns} columns",
                f"Data quality {profile.quality_score:.0f}/100",
            ] + ([f"{len(profile.leakage_suspects)} column(s) may leak the answer"]
                 if getattr(profile, "leakage_suspects", None) else []),
            confidence=(lead.confidence if lead else None),
        )
        st.page_link("pages/4_Analysis.py", label="Set up the analysis →")

        st.divider()
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("### Data quality")
            quality_bars(quality_breakdown(profile))
            if profile.quality_issues:
                st.caption(f"{len(profile.quality_issues)} issue(s) — the **Data** page works "
                           "through them.")
        with right:
            st.markdown("### What this data could answer")
            for objective in state.objectives[:5]:
                st.markdown(
                    f'<div style="font-size:.8125rem;padding:.3rem 0;'
                    f'border-bottom:1px solid var(--border)">'
                    f'<strong style="color:var(--ink)">{objective.label().replace("_", " ")}</strong>'
                    f'<br><span style="color:var(--ink-3)">{objective.rationale[:110]}</span></div>',
                    unsafe_allow_html=True,
                )
        st.stop()

    # ---------------------------------------------------------------------
    # An analysis exists: lead with the answer, then whether to believe it.
    # ---------------------------------------------------------------------
    headline = run.findings[0] if run.findings else None
    action = run.recommendations[0] if run.recommendations else None

    st.markdown("### The answer")
    if headline is not None:
        st.markdown(
            f'<div class="dsai-rank-title" style="font-size:1.35rem;line-height:1.35;'
            f'max-width:60ch">{headline.title}</div>'
            f'<div style="color:var(--ink-2);font-size:.9375rem;line-height:1.6;'
            f'max-width:66ch;margin-top:.5rem">{headline.detail}</div>',
            unsafe_allow_html=True,
        )
    elif run.best is not None:
        st.markdown(
            f"**{run.best.model_name}** came out in front, but the run produced no finding worth "
            "leading with. The **Models** page has the comparison."
        )
    else:
        error_state(
            "This run produced no usable model",
            "Every candidate failed to train. The Models page lists what happened to each.",
            "Check the Data page for unresolved quality problems, then try again with a "
            "preprocessing pipeline built on the Preprocessing page.",
        )

    st.divider()

    # Whether to believe it, before anything else about it.
    if run.trust is not None:
        st.markdown("### Why trust this?")
        trust_panel(run.trust)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### What drives the result")
        if run.explanation is not None and run.explanation.importances:
            for position, item in enumerate(run.explanation.top(5), start=1):
                st.markdown(
                    f'<div style="display:flex;gap:.7rem;align-items:baseline;padding:.28rem 0;'
                    f'border-bottom:1px solid var(--border);font-size:.8125rem">'
                    f'<span style="font-family:var(--mono);color:var(--ink-3)">{position}</span>'
                    f'<span style="color:var(--ink);font-weight:600">{item.feature}</span>'
                    f'<span style="margin-left:auto;color:var(--ink-3)">'
                    f'{item.direction or ""}</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Measured by {run.explanation.method}. What the model relies on — which "
                       "is not the same as what causes the outcome.")
        else:
            st.caption("No explanation was produced for this model.")

    with right:
        st.markdown("### What should happen next")
        for position, recommendation in enumerate(run.recommendations[:3], start=1):
            st.markdown(
                f'<div style="padding:.35rem 0;border-bottom:1px solid var(--border);'
                f'font-size:.8125rem;line-height:1.5">'
                f'<span style="font-family:var(--mono);color:var(--ink-3)">{position}</span> '
                f'<strong style="color:var(--ink)">{recommendation.action}</strong>'
                f'<br><span style="color:var(--ink-3)">{recommendation.reason[:120]}</span></div>',
                unsafe_allow_html=True,
            )
        if not run.recommendations:
            st.caption("No recommendations were generated.")

    if run.trust is not None:
        st.divider()
        st.markdown("### What could invalidate this")
        known_unknowns(run.trust)
        if run.trust.debt_items:
            with st.expander(f"Assumption debt — {run.trust.assumption_debt}/100 "
                             f"({run.trust.debt_band})"):
                st.caption(
                    "What would have to be true for the conclusion to hold, and has not been "
                    "shown. Not a probability — a running list of what is still owed."
                )
                for item in run.trust.debt_items:
                    st.markdown(f"- {item}")

    st.divider()
    links = st.columns([1, 1, 1, 1, 1, 1])
    links[0].page_link("pages/6_Insights.py", label="Explore evidence →")
    links[1].page_link("pages/13_Score.py", label="Score new data →")
    links[2].page_link("pages/9_Report.py", label="Generate report →")
    links[3].page_link("pages/5_Models.py", label="Review analysis →")
    links[4].page_link("pages/10_Projects.py", label="Runs →")

    st.divider()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Data quality")
        quality_bars(quality_breakdown(profile))
    with right:
        st.markdown("### Recent activity")
        for entry in reversed(state.runs[-5:]):
            question = entry.objective.task_type.value.replace("_", " ") if entry.objective else "run"
            model = entry.best.model_name if entry.best else "no model"
            st.markdown(
                f'<div style="font-size:.8125rem;color:var(--ink-2);padding:.3rem 0;'
                f'border-bottom:1px solid var(--border)">'
                f'<strong style="color:var(--ink)">{question}</strong> · {model} · '
                f"{len(entry.results)} model(s) · {entry.duration_s:.0f}s</div>",
                unsafe_allow_html=True,
            )
        if len(state.runs) >= 2:
            st.page_link("pages/10_Projects.py", label="Compare two runs →")
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
