"""The workspace entry point.

Run with::

    streamlit run dsai/app/main.py

or::

    dsai app
"""

from __future__ import annotations

import streamlit as st

from dsai.app.state import workspace
from dsai.registry.base import REGISTRY, load_builtin_models
from dsai.preprocessing.steps import STEPS

st.set_page_config(page_title="DSAI — AI data science workspace", page_icon="🔬", layout="wide")

state = workspace()
load_builtin_models()

with st.sidebar:
    st.markdown("### DSAI")
    st.caption("An AI data scientist that runs the analysis, not just describes it.")
    st.divider()
    if state.has_data:
        st.markdown(f"**Dataset:** {state.dataset_name}")
        st.caption(f"{len(state.frame):,} rows × {state.frame.shape[1]} columns")
        if state.profile:
            st.caption(f"Quality score {state.profile.quality_score}/100")
    else:
        st.caption("No dataset loaded.")
    if state.run is not None:
        st.divider()
        st.markdown("**Last run**")
        st.caption(state.run.summary())
    st.divider()
    state.theme = st.radio("Chart theme", ["light", "dark"],
                           index=0 if state.theme == "light" else 1, horizontal=True)

st.title("An AI data scientist, not a chatbot about one")
st.markdown(
    """
Upload data and the platform profiles it, decides what analysis the data actually supports,
builds a preprocessing pipeline suited to each candidate model, trains and cross-validates a
field of them, compares them on more than the headline score, explains the winner, checks its
own conclusions, and tells you what to do — with the evidence attached to every claim.

Every automated decision is a recommendation you can override, and every one is recorded.
    """
)

columns = st.columns(4)
stats = REGISTRY.stats()
columns[0].metric("Algorithms", stats["total"], help=f"{stats['available']} available in this environment")
columns[1].metric("Preprocessing steps", len(STEPS))
columns[2].metric("Statistical tests", "15+")
columns[3].metric("Export formats", "6")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("The workflow")
    st.markdown(
        """
1. **Data** — load it; the platform profiles every variable and flags quality problems
2. **Context** — tell it what the data means; kept separate from what was measured
3. **Preprocessing** — review, edit or build the pipeline; leakage is prevented structurally
4. **Analysis** — choose the objective, see the plan, run it
5. **Models** — the tournament, the selected model, its explanation and diagnostics
6. **Insights** — findings, grouped by the kind of evidence behind them
7. **Recommendations** — what to do, traced back to the analysis
        """
    )
    st.caption("Also: **Ask** for plain-language questions, **Report** to export, **Projects** to save.")

with right:
    st.subheader("Three ways to work")
    st.markdown(
        """
**AI automatic** — *"Do the analysis for me."* The platform runs the whole workflow and
reports what it found and what it decided.

**AI assisted** — *"Recommend what I should do, then let me approve it."* It plans, you
adjust, then it runs.

**Manual / expert** — *"I know what I want."* Full control of the objective, variables,
preprocessing pipeline, algorithms, hyper-parameters, validation strategy and metrics.

All three use the same engine. Only the amount you decide changes.
        """
    )

st.divider()
st.subheader("What it will not do")
st.markdown(
    """
- **Claim a model is best in general.** It reports the best-performing model *for this dataset
  under the validation strategy used*, and names the simplest acceptable alternative alongside it.
- **Present a correlation as a cause.** Every model-derived finding is labelled as an
  association, with the caveat attached rather than buried.
- **Hide a weak result.** If nothing beats a model that ignores every predictor, it says so
  plainly instead of presenting the least-bad option.
- **Let preprocessing leak.** Anything that learns from the data is fitted inside each
  cross-validation fold, never on the full dataset.
- **Mix your assumptions with its measurements.** What you told it is reported separately from
  what it found.
    """
)

if not state.has_data:
    st.divider()
    st.info("Start on the **Data** page in the sidebar — there are sample datasets if you just want to look around.", icon="📄")
