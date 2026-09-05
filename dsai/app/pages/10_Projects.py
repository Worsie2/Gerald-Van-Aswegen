"""Projects page: save a workspace and come back to it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dsai.app.components import dataframe, page_header, show_notices
from dsai.app.state import workspace
from dsai.repro.project import Project, list_projects

st.set_page_config(page_title="Projects · DSAI", page_icon="🗂", layout="wide")
state = workspace()
show_notices(state)

page_header(
    "Projects",
    "A project stores the dataset, your context, saved pipelines and every run — as plain files, "
    "so it can be inspected and version-controlled.",
)

root = st.text_input("Projects folder", value=str(Path.home() / "dsai_projects"))
root_path = Path(root)

create_tab, open_tab, runs_tab = st.tabs(["Save current work", "Open a project", "Run history"])

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
                if state.pipeline is not None:
                    project.save_pipeline(state.pipeline)
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
                state.frame = project.data()
                state.context = project.context()
                state.dataset_name = project.meta.name
                state.project_path = chosen
                state.profile = None
                state.reset_analysis()
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
columns[3].metric("Project", Path(state.project_path).name if state.project_path else "unsaved")
