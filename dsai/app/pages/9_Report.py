"""Report page: the finished report, and every export format."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from dsai.app.components import (
    apply_theme, caveat, chart, dataframe, decision_panel, page_header, require_run, show_notices,
    sidebar_chrome,
)
from dsai.app.workings import learned_parameters
from dsai.app.state import workspace
from dsai.reporting.builder import build_report
from dsai.reporting.exporters import (
    to_excel, to_html, to_json, to_markdown, to_pdf, to_python, write_charts,
)
from dsai.repro.provenance import build_manifest

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
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

methodology = st.checkbox(
    "Include the methodology and the calculations",
    value=False,
    help="Adds three sections: how the analysis was run, the arithmetic behind every figure it "
         "reports — fold by fold — and whether anything went wrong producing it.",
)

frame = state.typed_frame if state.typed_frame is not None else state.frame
report = build_report(run, audience_key, state.context.currency,
                      frame=frame, mode=state.theme, include_methodology=methodology)
markdown = report.to_markdown()

read_tab, charts_tab, method_tab, decisions_tab, manifest_tab, export_tab = st.tabs(
    ["Read", "Charts", "Methodology & workings", "Decision log", "Reproducibility", "Export"]
)

with read_tab:
    # The report reads in order, charts placed in the section they illustrate —
    # the same document that comes out of the HTML export.
    st.markdown(f"# {report.title}")
    st.caption(f"Generated {report.generated_at}")
    if report.executive_summary:
        st.markdown("## Executive summary")
        st.markdown(report.executive_summary)
    for heading, body in report.sections:
        st.markdown(f"## {heading}")
        st.markdown(body)
        for figure in report.figures_for(heading):
            st.markdown(f"**{figure.title}**")
            chart(figure.figure, caption=figure.caption, key=f"report_read_{figure.key}",
                  table=figure.table)

with charts_tab:
    if not report.figures:
        st.info(
            "This run produced nothing to chart. Charts appear once there is a model comparison, "
            "an explanation, segments or a forecast to draw."
        )
    else:
        st.caption(
            f"{len(report.figures)} chart(s), each travelling with the report into every export. "
            "The caption states what the chart shows in words, so the point survives a reader who "
            "cannot see it."
        )
        for figure in report.figures:
            st.subheader(figure.title)
            chart(figure.figure, caption=figure.caption, key=f"report_chart_{figure.key}",
                  table=figure.table)

with method_tab:
    # Always available here regardless of the export toggle: a reader checking
    # whether the analysis is sound should not have to change an export setting
    # to see the workings.
    st.caption(
        "How it was run, the arithmetic behind every reported figure, and whether anything went "
        "wrong on the way. Tick the box above to carry these into the exported report as well."
    )
    full = build_report(run, "technical", state.context.currency, frame=frame,
                        mode=state.theme, include_charts=False, include_methodology=True)
    wanted = ["Methodology", "The calculations", "Did it run cleanly"]
    for heading in wanted:
        body = dict(full.sections).get(heading, "")
        if not body.strip():
            continue
        st.markdown(f"### {heading}")
        st.markdown(body)
        st.divider()

    if state.pipeline is not None:
        learned = learned_parameters(state.pipeline, frame)
        if learned:
            st.markdown("### What the preprocessing actually learned")
            st.caption(
                "The values each fitted step would use, computed here on the whole dataset so they "
                "can be shown. The pipeline refits them inside every cross-validation fold, so the "
                "numbers it uses in scoring come from training rows only and will differ slightly."
            )
            for title, table in learned:
                st.markdown(f"**{title}**")
                dataframe(table)

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
        "HTML report", to_html(run, audience_key, frame=frame, methodology=methodology),
        file_name=f"{stem}.html",
        mime="text/html", use_container_width=True,
        help="Self-contained: text, tables and the charts, interactive, with no internet needed.",
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
            path = to_excel(run, Path(directory) / f"{stem}.xlsx", frame=frame)
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
        "Business summary", to_markdown(run, "business", frame=frame),
        file_name=f"{stem}_business.md", mime="text/markdown", use_container_width=True,
    )

    st.divider()
    st.subheader("Charts and PDF")
    if report.figures:
        with tempfile.TemporaryDirectory() as directory:
            files = write_charts(run, directory, frame=frame)
            if files:
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for key, relative in files.items():
                        bundle.write(Path(directory) / relative, arcname=relative)
                st.download_button(
                    f"Charts as PNG ({len(files)} files, .zip)", archive.getvalue(),
                    file_name=f"{stem}_charts.zip", mime="application/zip",
                    use_container_width=True,
                )
            else:
                caveat(
                    "PNG chart export needs a headless browser, which this machine does not have. "
                    "Install one with `plotly_get_chrome`, or use the HTML report — its charts are "
                    "interactive and need nothing extra."
                )

        with tempfile.TemporaryDirectory() as directory:
            try:
                pdf = to_pdf(run, Path(directory) / f"{stem}.pdf", audience_key, frame=frame,
                             methodology=methodology)
                st.download_button(
                    "PDF report", Path(pdf).read_bytes(), file_name=f"{stem}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            except Exception as exc:
                st.caption(
                    f"PDF not written here ({exc}). The HTML export printed to PDF from your "
                    "browser gives a better result anyway, and keeps the charts."
                )

    st.subheader("Preview of the generated Python")
    st.code(to_python(run, data_path=data_hint)[:3000], language="python")
