"""Chart palette and layout defaults.

The categorical order below is validated: it clears the colour-vision-deficiency
and normal-vision separation gates on adjacent pairs in both light and dark mode.
The ordering is the safety mechanism, not decoration — do not re-order or cycle
it. A ninth series folds into "Other" rather than getting a generated hue.
"""

from __future__ import annotations

from typing import Any

#: Fixed categorical order. Assigned by position, never cycled.
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

#: Forms that put every pair of series on screen at once (scatter, bubble)
#: cannot use more than three of these hues and stay separable.
ALL_PAIRS_SERIES_CAP = 3

#: Single-hue ramp for magnitude. Light to dark, never a rainbow.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
]
#: Two poles with a neutral midpoint, for values that are meaningfully +/-.
DIVERGING_LIGHT = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a0a0", "#e34948", "#8f2020"]
DIVERGING_DARK = ["#0d366b", "#256abf", "#86b6ef", "#383835", "#f0a0a0", "#e66767", "#8f2020"]

STATUS = {
    "good": "#008300",
    "warning": "#eda100",
    "serious": "#eb6834",
    "critical": "#e34948",
}

SURFACES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#7a7873",
        "grid": "#e6e5e1",
        "axis": "#c9c7c1",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#8c8b83",
        "grid": "#2f2f2c",
        "axis": "#484743",
    },
}

FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)


def palette(mode: str = "light", n: int | None = None, all_pairs: bool = False) -> list[str]:
    """Categorical colours in fixed order.

    ``all_pairs=True`` is for chart forms where every series is visible against
    every other (scatter, bubble); those are capped at three, because past that
    the pairs stop being separable for colour-blind readers.
    """
    colours = CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT
    limit = ALL_PAIRS_SERIES_CAP if all_pairs else len(colours)
    if n is not None:
        limit = min(limit, n)
    return colours[:limit]


def sequential(mode: str = "light", reverse: bool = False) -> list[str]:
    ramp = list(SEQUENTIAL_BLUE)
    return ramp[::-1] if reverse else ramp


def diverging(mode: str = "light") -> list[str]:
    return DIVERGING_DARK if mode == "dark" else DIVERGING_LIGHT


def ink(mode: str = "light") -> dict[str, str]:
    return SURFACES["dark" if mode == "dark" else "light"]


def layout(title: str = "", mode: str = "light", height: int = 380, **overrides: Any) -> dict[str, Any]:
    """Plotly layout defaults: recessive grid, ink-coloured text, room to breathe."""
    colours = ink(mode)
    base: dict[str, Any] = {
        "title": {
            "text": title,
            "font": {"size": 15, "color": colours["text_primary"], "family": FONT_FAMILY},
            "x": 0, "xanchor": "left", "pad": {"b": 12},
        },
        "height": height,
        "paper_bgcolor": colours["surface"],
        "plot_bgcolor": colours["surface"],
        "font": {"family": FONT_FAMILY, "size": 12, "color": colours["text_secondary"]},
        "margin": {"l": 60, "r": 24, "t": 52, "b": 52},
        "hoverlabel": {"font": {"family": FONT_FAMILY, "size": 12}, "bgcolor": colours["surface"]},
        "xaxis": {
            "gridcolor": colours["grid"], "zerolinecolor": colours["axis"],
            "linecolor": colours["axis"], "tickfont": {"color": colours["text_muted"], "size": 11},
            "title": {"font": {"color": colours["text_secondary"], "size": 12}},
        },
        "yaxis": {
            "gridcolor": colours["grid"], "zerolinecolor": colours["axis"],
            "linecolor": colours["axis"], "tickfont": {"color": colours["text_muted"], "size": 11},
            "title": {"font": {"color": colours["text_secondary"], "size": 12}},
        },
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1,
            "font": {"color": colours["text_secondary"], "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "showlegend": False,
    }
    base.update(overrides)
    return base
