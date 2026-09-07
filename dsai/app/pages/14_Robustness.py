"""Robustness page: does the conclusion survive reasonable changes to the method?"""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, dataframe, empty_state, error_state, metric_row,
    page_header, require_run, show_notices, sidebar_chrome, status_bar,
)
from dsai.app.state import workspace
from dsai.core.schema import TaskType
from dsai.engines.sensitivity import build_specifications, run_sensitivity
from dsai.engines.subgroups import MIN_GROUP_ROWS, discover_subgroups
from dsai.viz import plots

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Robustness",
    "An analysis embeds choices nobody argued about — which imputation, whether to clip outliers, "
    "which model family. This asks whether the conclusion depended on them.",
)

if not require_run(state):
    st.stop()

run = state.run
frame = state.typed_frame if state.typed_frame is not None else state.frame

if run.best is None:
    empty_state(
        "This run produced no model to test",
        "Robustness testing re-runs the analysis under alternative specifications. There has to "
        "be something to re-run first.",
        actions=[("Back to the analysis", "pages/4_Analysis.py")],
    )
    st.stop()

status_bar(state)

sensitivity_tab, subgroup_tab = st.tabs(
    ["What could change this conclusion?", "Where does the model work poorly?"]
)

# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------
with sensitivity_tab:
    specs = build_specifications(run)
    ai_panel(
        f"I can re-run this analysis **{len(specs)} ways** — every one a choice a competent "
        "analyst might have made instead — and tell you whether the answer holds. A conclusion "
        "that survives all of them is worth acting on. One that flips when the outlier treatment "
        "changes is a finding about the outlier treatment.",
        heading="AI Analyst · sensitivity",
        why="Only reasonable specifications are tried. Searching over unreasonable ones and "
            "reporting that the result survived would be theatre.",
    )

    with st.expander(f"The {len(specs)} specifications, and why each one is defensible"):
        dataframe(
            __import__("pandas").DataFrame([
                {"Varies": s.dimension, "Specification": s.label, "Why it is reasonable": s.rationale}
                for s in specs
            ])
        )

    estimate = max(1, round(run.duration_s * len(specs) / max(len(run.results), 1) / 60))
    caveat(
        f"This trains {len(specs)} models. On this dataset that is likely to take a few minutes "
        f"— roughly {estimate} minute(s) at the speed the original run managed. The interface "
        "will not respond while it works."
    )

    if st.button("Run the sensitivity analysis", type="primary", width='stretch'):
        placeholder = st.empty()
        with st.status("Running specifications…", expanded=True) as status:
            def progress(index: int, total: int, label: str) -> None:
                status.update(label=f"[{index + 1}/{total}] {label}")

            try:
                report = run_sensitivity(run, frame, specs, progress=progress)
                st.session_state["_sensitivity"] = report
                status.update(label=f"{report.n_ran} of {len(specs)} specification(s) completed",
                              state="complete")
            except Exception as exc:
                status.update(label="Failed", state="error")
                error_state(
                    "The sensitivity analysis could not finish",
                    f"{type(exc).__name__}: {exc}",
                    "The original analysis is unaffected — nothing here modifies it. Try again "
                    "with a shorter time budget on the **Analysis** page if the run is timing out.",
                )
        st.rerun()

    report = st.session_state.get("_sensitivity")
    if report is not None:
        low, high = report.spread()
        metric_row([
            ("Specifications run", f"{report.n_ran}", f"of {len(report.results)} attempted"),
            ("Agreement", f"{report.agreement:.0%}",
             "share naming the same strongest driver"),
            ("Verdict", "stable" if report.stable else "sensitive",
             "whether the conclusion survived"),
            (f"{report.metric} range",
             f"{low:,.4g} – {high:,.4g}" if low is not None else "—",
             "across every specification"),
        ])

        if report.stable:
            ai_panel(report.verdict, heading="AI Analyst · sensitivity result")
        else:
            error_state(
                "The conclusion depends on how the analysis was run",
                report.verdict,
                "Decide which specification is right and say why, or report the result as "
                "conditional on that choice. Both are honest; presenting one specification as "
                "the answer is not.",
            )
        for note in report.notes:
            caveat(note)

        st.subheader("Every specification")
        dataframe(report.table())
        st.caption(
            "The Top driver column is what a reader acts on, so it is what agreement is measured "
            "on — not the third decimal place of the metric. A categorical driver named before "
            "and after encoding counts as the same variable."
        )

        succeeded = [r for r in report.succeeded if r.score is not None]
        if len(succeeded) > 2:
            chart(
                plots.bar(
                    [r.spec.label for r in succeeded], [r.score for r in succeeded],
                    title=f"{report.metric} across specifications", mode=state.theme,
                    orientation="h",
                ),
                caption="How far the headline number moved. A tight cluster means the metric did "
                        "not depend on these choices; that is a weaker claim than the direction "
                        "holding, and both are reported.",
                key="sens_bar",
            )

# --------------------------------------------------------------------------
# subgroups
# --------------------------------------------------------------------------
with subgroup_tab:
    actual = run.best.extras.get("holdout_actual")
    predicted = (run.best.extras.get("holdout_predicted")
                 or run.best.extras.get("holdout_predicted_labels"))
    index = run.best.extras.get("holdout_index")

    if not actual or not predicted or index is None:
        empty_state(
            "No held-out predictions to break down",
            "This needs the model's predictions on rows it did not train on, which this run did "
            "not keep.",
        )
    else:
        try:
            held = frame.loc[list(index)]
        except Exception:
            held = None

        if held is None or len(held) != len(actual):
            empty_state("The held-out rows could not be matched back to the data", "")
        else:
            ai_panel(
                "An average error hides its own distribution. A model at 6% almost everywhere and "
                "40% on one region has the same average as one that is mediocre throughout — and "
                "only one of those is a problem you can do something about.",
                heading="AI Analyst · subgroup performance",
                why=f"Only groups of at least {MIN_GROUP_ROWS} rows are reported, and only "
                    "differences beyond three standard errors are flagged. On any dataset, "
                    "searching enough subgroups will find one that looks bad by chance.",
            )
            candidates = [c for c in held.columns if c != run.best.target]
            chosen = st.multiselect(
                "Break the results down by", candidates,
                default=candidates[:5],
                help="Categorical columns are split by value; numeric ones into thirds.",
            )
            depth = st.radio(
                "Combinations", [1, 2], index=1, horizontal=True,
                format_func=lambda d: "single conditions" if d == 1 else "pairs of conditions",
                help="Pairs find weaknesses that neither condition shows alone, at the cost of "
                     "searching many more groups.",
            )

            if chosen:
                with st.spinner("Comparing groups…"):
                    subgroups = discover_subgroups(
                        held, actual, predicted, run.best.task_type,
                        candidate_columns=chosen, max_depth=int(depth),
                    )
                if not subgroups.usable:
                    st.info(subgroups.note or subgroups.verdict)
                else:
                    metric_row([
                        ("Overall", f"{subgroups.overall_error:,.4g}", subgroups.metric),
                        ("Groups compared", f"{len(subgroups.groups)}",
                         f"{subgroups.n_too_small} were too small"),
                        ("Standing out", f"{len(subgroups.notable)}",
                         "beyond three standard errors"),
                        ("Searched", f"{subgroups.n_considered}", "candidate groups"),
                    ])
                    if subgroups.notable:
                        error_state(
                            "The model does materially worse on some groups",
                            subgroups.verdict,
                            "Look at the worst group in the data itself. Fewer training rows, a "
                            "variable that means something different within it, or genuine "
                            "heterogeneity would each produce this.",
                        )
                    else:
                        ai_panel(subgroups.verdict, heading="AI Analyst · subgroup result")
                    dataframe(subgroups.table())
                    caveat(
                        "These are measured differences in prediction error. Calling one bias or "
                        "a fairness failure would be a conclusion this evidence cannot support — "
                        "and would let the reader skip the investigation that matters."
                    )
