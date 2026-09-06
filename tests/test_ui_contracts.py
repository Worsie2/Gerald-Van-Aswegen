"""Contracts the interface must hold, checked without a browser.

These exist because of bugs that actually happened. A stray ``*/`` inside a CSS
comment once silently deleted a third of the stylesheet — the app still ran, the
tests still passed, and the whole premium layer simply had no styling. A test
that reads the generated CSS catches that class of failure in a second.
"""

from __future__ import annotations

import re

import pytest

from dsai.app import components as C
from dsai.app.state import Accessibility


def _rules(mode: str) -> str:
    """The stylesheet with comments removed, i.e. what a browser would honour."""
    css = C._stylesheet(mode)
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_stylesheet_has_no_unbalanced_braces(mode):
    body = _rules(mode)
    assert body.count("{") == body.count("}")


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_no_comment_closes_early(mode):
    """A ``*/`` inside a comment body ends it early and eats the rules after it."""
    css = C._stylesheet(mode)
    for match in re.finditer(r"/\*(.*?)\*/", css, flags=re.S):
        assert "/*" not in match.group(1), f"nested comment near: {match.group(1)[:60]}"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_component_class_is_styled(mode):
    """Each ``dsai-`` class the Python emits must have a rule the browser will see.

    Catches both a class renamed in one place only, and a whole block of rules
    silently dropped by a malformed comment.
    """
    source = open(C.__file__, encoding="utf-8").read()
    emitted = set(re.findall(r'class="(dsai-[a-z-]+)"', source))
    body = _rules(mode)
    missing = sorted(name for name in emitted if f".{name}" not in body)
    assert not missing, f"emitted but never styled: {missing}"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_no_unsubstituted_placeholders(mode):
    assert not re.findall(r"\{[a-z_]+\}", C._stylesheet(mode))


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_accessibility_layer_is_valid_css(mode):
    for access in [
        Accessibility(),
        Accessibility(text_scale=1.5, high_contrast=True, reduce_motion=True,
                      always_show_tables=True, underline_links=True),
    ]:
        css = C._accessibility_css(mode, access)
        body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        assert body.count("{") == body.count("}")
        assert "--focus:" in body


def test_high_contrast_actually_changes_the_ink():
    plain = C._accessibility_css("dark", Accessibility())
    strong = C._accessibility_css("dark", Accessibility(high_contrast=True))
    assert "--ink:#ffffff" in strong.replace(" ", "")
    assert "--ink:#ffffff" not in plain.replace(" ", "")


def test_text_scale_multiplies_the_type_scale():
    scaled = C._accessibility_css("dark", Accessibility(text_scale=1.5))
    assert "--t-body:" in scaled
    assert C._scaled("0.9375rem", 1.5) == "1.406rem"
    assert C._scaled("14px", 2.0) == "28.000px"
    assert C._scaled("auto", 2.0) == "auto"


def test_rich_keeps_line_breaks_apart():
    """Two evidence lines must not run together into one sentence."""
    assert C._rich("first  \nsecond") == "first<br>second"
    assert "<strong>" in C._rich("**bold**")
    assert "&lt;script&gt;" in C._rich("<script>")


def test_confidence_glyph_is_hidden_from_screen_readers():
    from dsai.core.schema import Confidence

    html = C._confidence_html(Confidence.HIGH)
    assert 'aria-hidden="true"' in html
    assert "high" in html


def test_figure_to_frame_reads_a_bar_chart():
    plotly = pytest.importorskip("plotly.graph_objects")
    figure = plotly.Figure(plotly.Bar(x=["a", "b"], y=[1, 2]))
    frame = C.figure_to_frame(figure)
    assert list(frame["Y"]) == [1, 2]
    assert C.figure_to_frame(None) is None
    assert C.figure_to_frame(plotly.Figure()) is None


def test_figure_to_frame_reads_a_heatmap():
    plotly = pytest.importorskip("plotly.graph_objects")
    figure = plotly.Figure(plotly.Heatmap(z=[[1, 2], [3, 4]], x=["p", "q"], y=["r", "s"]))
    frame = C.figure_to_frame(figure)
    assert frame.shape == (2, 3)


def test_stage_progress_names_every_state():
    html = C.stage_progress([("Profiling", "done", ""), ("Training", "running", "3 of 8"),
                             ("Nothing", "todo", "")])
    assert 'data-state="running"' in html
    assert "3 of 8" in html
    assert 'aria-label="Analysis progress"' in html


def test_quality_breakdown_covers_every_dimension():
    from dsai.app.quality import QUALITY_DIMENSIONS, quality_breakdown
    from dsai.app.samples import build_sample
    from dsai.core.profiler import profile_dataset

    frame, _ = build_sample("messy_survey")
    profile, _ = profile_dataset(frame, name="messy")
    rows = quality_breakdown(profile)
    assert len(rows) == len(QUALITY_DIMENSIONS)
    assert all(0 <= score <= 100 for _, score, _ in rows)
    assert all(note for _, _, note in rows)


def test_doctor_reports_without_raising():
    from dsai.doctor import render, run_checks

    checks = run_checks()
    assert any(c.name == "Python version" for c in checks)
    text = render(checks)
    assert "dsai doctor" in text
    # Every failure must carry a fix; a diagnosis with no remedy is not one.
    assert all(c.fix for c in checks if not c.ok)
