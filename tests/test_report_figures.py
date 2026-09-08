"""Charts travelling with the report, and what happens when they cannot render.

The rule these enforce: a chart is never the only way to read something. Every
figure carries a written caption and, where it has one, the numbers behind it —
so a reader with no images still gets the point.
"""

from __future__ import annotations

import re

import pytest

from dsai.reporting.builder import build_report
from dsai.reporting.exporters import to_html, to_markdown, to_pdf, to_python
from dsai.reporting.figures import ReportFigure, build_figures, to_html_div, to_png


def test_figures_are_built_for_a_regression_run(regression_run):
    run, frame = regression_run
    figures = build_figures(run, frame=frame)
    keys = {f.key for f in figures}
    assert "model_comparison" in keys
    assert "importance" in keys


def test_every_figure_carries_a_caption_and_alt(regression_run):
    run, frame = regression_run
    for figure in build_figures(run, frame=frame):
        assert figure.caption.strip(), f"{figure.key} has no caption"
        assert figure.alt.strip(), f"{figure.key} has no alt text"
        # The caption must say something about the data, not describe a chart.
        assert len(figure.caption) > 40


def test_figures_attach_to_sections_that_exist(regression_run):
    run, frame = regression_run
    report = build_report(run, frame=frame)
    headings = {heading for heading, _ in report.sections}
    for figure in report.figures:
        assert figure.after_section in headings


def test_html_report_embeds_charts_and_their_tables(regression_run):
    run, frame = regression_run
    html = to_html(run, frame=frame)
    assert "plotly" in html.lower()
    assert "<figure>" in html
    assert "The numbers behind this chart" in html
    # Self-contained: the library is inlined, not fetched from a CDN. The bundle
    # names a CDN in its own config defaults, so the check is for a script tag
    # that would go and fetch something, not for the string appearing anywhere.
    assert not re.search(r'<script[^>]+src\s*=\s*"https?://', html)
    assert "Plotly.newPlot" in html


def test_html_charts_have_an_accessible_label(regression_run):
    run, frame = regression_run
    html = to_html(run, frame=frame)
    assert 'role="img"' in html
    assert 'aria-label="' in html


def test_markdown_carries_captions_without_images(regression_run):
    """With no chart files written, the caption and the table must stand alone."""
    run, frame = regression_run
    markdown = to_markdown(run, frame=frame)
    assert "![" not in markdown
    report = build_report(run, frame=frame)
    for figure in report.figures:
        assert figure.caption[:40] in markdown


def test_markdown_links_images_when_they_exist(regression_run):
    run, frame = regression_run
    report = build_report(run, frame=frame)
    files = {f.key: f"charts/{f.key}.png" for f in report.figures}
    markdown = report.to_markdown(chart_files=files)
    for figure in report.figures:
        assert f"![{figure.alt}](charts/{figure.key}.png)" in markdown


def test_report_survives_a_figure_that_cannot_be_drawn(regression_run, monkeypatch):
    """A broken plotting layer costs the reader pictures, never the report."""
    import dsai.viz.plots as plots

    monkeypatch.setattr(plots, "model_comparison",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    figures = build_figures(regression_run[0], frame=regression_run[1])
    assert "model_comparison" not in {f.key for f in figures}
    assert figures  # the others still came through


def test_png_and_html_helpers_never_raise_on_none():
    assert to_png(None) is None
    assert to_html_div(None) is None


def test_figure_without_a_table_is_still_valid():
    figure = ReportFigure(key="k", title="t", caption="c" * 50, alt="a",
                          after_section="Key findings")
    assert figure.has_table is False


def test_correlation_falls_back_without_the_frame(regression_run):
    """No data to hand still leaves the profile's measured pairs to draw."""
    run, _ = regression_run
    figures = build_figures(run, frame=None)
    correlation = [f for f in figures if f.key == "correlation"]
    if correlation:
        assert correlation[0].has_table
        assert "association" in correlation[0].caption.lower()


# ---------------------------------------------------------------------------
# to_pdf, to_python — the two export formats with no prior coverage at all
# ---------------------------------------------------------------------------

def test_pdf_export_writes_a_real_pdf(regression_run, tmp_path):
    """reportlab is installed in this environment, so this must take the real
    PDF path, not the HTML-with-a-different-extension fallback -- and produce
    a file structurally recognisable as a PDF, not just a non-empty file."""
    run, frame = regression_run
    path = to_pdf(run, tmp_path / "report.pdf", frame=frame)
    assert path is not None
    written = path.read_bytes()
    assert written.startswith(b"%PDF-"), "does not start with a PDF header"
    assert b"%%EOF" in written[-2048:], "no PDF trailer near the end of the file"
    assert len(written) > 2_000, "suspiciously small for a multi-section report"


def test_pdf_export_falls_back_to_html_without_reportlab(regression_run, tmp_path, monkeypatch):
    """The documented fallback: no silent failure -- an honest HTML file is
    written AND the caller is told loudly, by a raised exception, that what
    it asked for was not a PDF. The Report page's only call site wraps this
    in a try/except for exactly that reason; this test locks in the contract
    that call site depends on."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "reportlab" or name.startswith("reportlab."):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    run, frame = regression_run
    target = tmp_path / "report.pdf"
    with pytest.raises(RuntimeError, match="reportlab"):
        to_pdf(run, target, frame=frame)

    fallback = tmp_path / "report_print.html"
    assert fallback.exists(), "no HTML fallback written alongside the failed PDF"
    assert b"<html" in fallback.read_bytes()[:500].lower()
    assert not target.exists(), "must not leave a .pdf file behind that is not actually a PDF"


def test_python_export_is_the_same_script_the_repro_command_generates(regression_run, tmp_path):
    """to_python is a thin wrapper over dsai.repro.codegen -- confirm the
    wrapper actually writes what it returns, to the path it is given."""
    run, _ = regression_run
    path = tmp_path / "reproduce.py"
    script = to_python(run, path, data_path="water_customers.csv")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == script
    assert "water_customers.csv" in script
    compile(script, str(path), "exec")  # syntactically valid Python, at minimum
