"""Projects page: save a workspace and come back to it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, dataframe, empty_state, error_state, page_header,
    show_notices, sidebar_chrome,
)
from dsai.app.state import workspace
from dsai.engines.compare import compare_runs
from dsai.repro.project import Project, list_projects

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Projects",
    "A project stores the dataset, your context, saved pipelines and every run — as plain files, "
    "so it can be inspected and version-controlled.",
)

root = st.text_input("Projects folder", value=str(Path.home() / "dsai_projects"))
root_path = Path(root)

create_tab, open_tab, runs_tab, compare_tab = st.tabs(
    ["Save current work", "Open a project", "Run history", "Compare two runs"]
)

with create_tab:
    if not state.has_data:
        st.info("Load a dataset first.")
    else:
        name = st.text_input("Project name", value=state.dataset_name or "untitled")
        description = st.text_area("Description", height=70)
        if st.button("Save project", type="primary"):
            path = root_path / name.replace(" ", "_")
            try:
                if Project.exists(path):
                    project = Project.open(path)
                    project.set_data(state.frame)
                    project.set_context(state.context)
                else:
                    project = Project.create(path, state.frame, name, description, state.context)
                for name, saved in state.pipelines.items():
                    project.save_pipeline(saved, name)
                for run in state.runs:
                    project.save_run(run)
                    model = state.scientist.fitted_model(run) if state.scientist else None
                    if model is not None:
                        project.save_model(model, run.id)
                state.project_path = str(path)
                state.notify("success", f"Saved to {path}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save: {exc}")

with open_tab:
    projects = list_projects(root_path)
    if not projects:
        st.info(f"No projects found in {root_path}.")
    else:
        dataframe(pd.DataFrame(projects))
        chosen = st.selectbox("Project", [p["path"] for p in projects],
                              format_func=lambda p: Path(p).name)
        if st.button("Open", type="primary"):
            try:
                project = Project.open(chosen)
                # Opening a project is loading a dataset: everything derived
                # from whatever was open before has to go with it.
                state.set_dataset(project.data(), None, project.meta.name)
                state.context = project.context()
                state.project_path = chosen
                state.notify("success", f"Opened '{project.meta.name}'.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not open: {exc}")

with runs_tab:
    if not state.project_path:
        st.info("Open or save a project to see its run history.")
    else:
        project = Project.open(state.project_path)
        runs = project.list_runs()
        if not runs:
            st.info("No runs saved in this project yet.")
        else:
            dataframe(pd.DataFrame(runs))
            chosen = st.selectbox("Inspect a run", [r["run_id"] for r in runs])
            if chosen:
                payload = project.load_run(chosen)
                st.markdown(f"**{payload.get('summary', '')}**")
                columns = st.columns(3)
                columns[0].metric("Findings", len(payload.get("findings", [])))
                columns[1].metric("Recommendations", len(payload.get("recommendations", [])))
                columns[2].metric("Decisions", len(payload.get("decisions", [])))
                with st.expander("Manifest"):
                    st.json(payload.get("manifest", {}), expanded=False)
                with st.expander("Findings"):
                    for finding in payload.get("findings", []):
                        st.markdown(f"- **{finding['title']}** — {finding['detail']}")
                with st.expander("Recommendations"):
                    for recommendation in payload.get("recommendations", []):
                        st.markdown(f"- **{recommendation['action']}** — {recommendation['reason']}")

st.divider()
st.subheader("Current session")
columns = st.columns(4)
columns[0].metric("Dataset", state.dataset_name or "—")
columns[1].metric("Rows", f"{len(state.frame):,}" if state.has_data else "—")
columns[2].metric("Runs this session", len(state.runs))
with compare_tab:
    # Changing one preprocessing choice and re-running tells you nothing unless
    # you can see what it did. This is that.
    if len(state.runs) < 2:
        empty_state(
            "Two runs are needed to compare anything",
            f"This session has **{len(state.runs)}**. Run the analysis again with something "
            "changed — a different objective, another pipeline, a longer time budget — and both "
            "will appear here.",
            actions=[("Back to the analysis", "pages/4_Analysis.py")],
        )
    else:
        def _label(index: int) -> str:
            run = state.runs[index]
            question = run.objective.label().replace("_", " ") if run.objective else "no objective"
            model = run.best.model_name if run.best else "no model"
            return f"{index + 1}. {question} · {model} · {run.duration_s:.0f}s"

        picker = st.columns(2)
        a_index = picker[0].selectbox("Run A (the earlier one)", range(len(state.runs)),
                                      index=len(state.runs) - 2, format_func=_label)
        b_index = picker[1].selectbox("Run B (what you changed to)", range(len(state.runs)),
                                      index=len(state.runs) - 1, format_func=_label)
        if a_index == b_index:
            caveat("Both selections are the same run. Pick two different ones.")
        else:
            diff = compare_runs(state.runs[a_index], state.runs[b_index])

            if diff.comparable:
                ai_panel(diff.verdict, heading="AI Analyst · what changed",
                         why="A difference smaller than the fold-to-fold spread within a single "
                             "run is not a difference — re-running either one with a different "
                             "seed could reverse it. That comparison is made before any verdict "
                             "is given.")
            else:
                error_state("These two runs cannot be compared on score", diff.reason,
                            "The setup and preprocessing below can still be compared — they just "
                            "do not add up to one being better than the other.")
            for note in diff.caveats:
                caveat(note)

            setup_view, score_view, steps_view, findings_view = st.tabs(
                ["Setup", "Scores", "Preprocessing", "Findings"]
            )
            with setup_view:
                st.caption("Everything that defined each run. The Changed column is the answer to "
                           "'what did I actually alter?'")
                dataframe(diff.setup)
            with score_view:
                if diff.scores.empty:
                    st.info("No comparable scores between these two runs.")
                else:
                    dataframe(diff.scores)
                    st.caption(
                        "Difference is B minus A in the metric's own units. Whether that counts as "
                        "better depends on the metric — the last column says which."
                    )
            with steps_view:
                if diff.preprocessing.empty:
                    st.info("Neither run recorded a pipeline.")
                else:
                    dataframe(diff.preprocessing)
            with findings_view:
                if diff.findings.empty:
                    st.info("Neither run produced findings.")
                else:
                    st.caption(
                        "A finding that appears in one run and not the other is worth more "
                        "attention than one that survives both — it is the part of the conclusion "
                        "that depends on what you changed."
                    )
                    dataframe(diff.findings)

columns[3].metric("Project", Path(state.project_path).name if state.project_path else "unsaved")
