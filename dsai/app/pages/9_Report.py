"""Report page: the finished report, and every export format."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st

from dsai.app.components import decision_panel, page_header, require_run, show_notices
from dsai.app.state import workspace
from dsai.reporting.builder import build_report
from dsai.reporting.exporters import to_excel, to_html, to_json, to_markdown, to_python
from dsai.repro.provenance import build_manifest

st.set_page_config(page_title="Report · DSAI", page_icon="📑", layout="wide")
state = workspace()
show_notices(state)

page_header(
    "Report",
    "The same analysis written for two audiences, plus everything needed to reproduce it.",
)

if not require_run(state):
    st.stop()

run = state.run
audience = st.radio(
    "Written for",
    ["Both audiences", "Management", "Technical"],
    horizontal=True,
    captions=[
        "Everything, business narrative first.",
        "What was found and what to do, without the methodology.",
        "Method, validation and diagnostics in full.",
    ],
)
audience_key = {"Both audiences": "both", "Management": "business", "Technical": "technical"}[audience]

report = build_report(run, audience_key, state.context.currency)
markdown = report.to_markdown()

read_tab, decisions_tab, manifest_tab, export_tab = st.tabs(
    ["Read", "Decision log", "Reproducibility", "Export"]
)

with read_tab:
    st.markdown(markdown)

with decisions_tab:
    st.caption(
        "Every decision the platform made, with its reason and what it set aside. This is a record "
        "of conclusions, not of internal reasoning."
    )
    stages = sorted({d.stage for d in run.decisions})
    stage = st.selectbox("Stage", ["all"] + stages)
    decision_panel(run.decisions, None if stage == "all" else stage)

with manifest_tab:
    manifest = build_manifest(run)
    st.code(manifest.render(), language=None)
    st.caption(
        f"Fingerprint `{manifest.fingerprint()}` is a hash over the dataset, objective, "
        "preprocessing, validation strategy, seed and model. Two runs with the same fingerprint "
        "should produce the same numbers."
    )

with export_tab:
    stem = f"{run.dataset_name}_{run.id}"
    columns = st.columns(3)

    columns[0].download_button(
        "Markdown report", markdown, file_name=f"{stem}.md", mime="text/markdown",
        use_container_width=True,
    )
    columns[1].download_button(
        "HTML report", to_html(run, audience_key), file_name=f"{stem}.html", mime="text/html",
        use_container_width=True,
    )
    columns[2].download_button(
        "JSON (full results)", to_json(run), file_name=f"{stem}.json", mime="application/json",
        use_container_width=True,
    )

    columns = st.columns(3)
    data_hint = (run.source.path if run.source and run.source.path else "your_data.csv")
    columns[0].download_button(
        "Python script", to_python(run, data_path=data_hint),
        file_name=f"{stem}_reproduce.py", mime="text/x-python", use_container_width=True,
        help="Standalone pandas + scikit-learn. Reproduces this analysis without the platform.",
    )

    with tempfile.TemporaryDirectory() as directory:
        try:
            path = to_excel(run, Path(directory) / f"{stem}.xlsx")
            columns[1].download_button(
                "Excel workbook", path.read_bytes(), file_name=f"{stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Summary, model comparison, findings, recommendations, variable profile and decision log.",
            )
        except Exception as exc:
            columns[1].button("Excel unavailable", disabled=True, use_container_width=True,
                              help=str(exc))

    columns[2].download_button(
        "Business summary", to_markdown(run, "business"),
        file_name=f"{stem}_business.md", mime="text/markdown", use_container_width=True,
    )

    st.divider()
    st.caption(
        "PDF: use the HTML export and print to PDF from your browser — it produces a better result "
        "than a dedicated renderer here would. If `reportlab` is installed, `dsai.reporting."
        "exporters.to_pdf()` writes one directly."
    )

    st.subheader("Preview of the generated Python")
    st.code(to_python(run, data_path=data_hint)[:3000], language="python")
