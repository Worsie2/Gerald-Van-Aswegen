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


# --------------------------------------------------------------------------
# Interface tokens
# --------------------------------------------------------------------------
# The chart palette above and the interface below deliberately share one
# source of truth. Slot 1 of the categorical palette is the single accent;
# everything else in the chrome is ink, surface or rule. An interface that
# competes with its own charts for attention makes both harder to read.

UI = {
    "light": {
        # Slot 1 of the categorical palette is the single accent, and the only
        # hue in the chrome. An interface that competes with its own charts for
        # attention makes both harder to read.
        "accent": CATEGORICAL_LIGHT[0],
        "accent_soft": "#eaf2fd",
        "accent_ink": "#1c5cab",
        "accent_deep": "#184f95",
        # The elevation ladder. Five steps, each a real surface with a job:
        # the plane the app sits on, the rail, the sheet content sits on, the
        # thing lifted off it, and the well things sit inside.
        "plane": "#f9f9f7",
        "rail": "#f2f1ee",
        "surface": SURFACES["light"]["surface"],
        "surface_raised": "#ffffff",
        "surface_sunken": "#f1f0ed",
        "border": "#e6e5e1",
        "border_strong": "#d3d1cb",
        "border_faint": "rgba(11,11,11,.06)",
        "text_primary": SURFACES["light"]["text_primary"],
        "text_secondary": SURFACES["light"]["text_secondary"],
        "text_muted": SURFACES["light"]["text_muted"],
        # Fine shadows, not drop shadows. Depth is a hint here, not an effect.
        "shadow": "0 1px 2px rgba(11,11,11,.045)",
        "shadow_raised": "0 1px 3px rgba(11,11,11,.06), 0 6px 16px -6px rgba(11,11,11,.08)",
        "shadow_overlay": "0 12px 32px -8px rgba(11,11,11,.16), 0 2px 6px rgba(11,11,11,.06)",
        "hover": "rgba(11,11,11,.035)",
        "selected": "#eaf2fd",
    },
    "dark": {
        "accent": CATEGORICAL_DARK[0],
        "accent_soft": "#14243a",
        "accent_ink": "#9ec5f4",
        "accent_deep": "#256abf",
        "plane": "#0d0d0d",
        "rail": "#151514",
        "surface": SURFACES["dark"]["surface"],
        "surface_raised": "#212120",
        "surface_sunken": "#151514",
        "border": "#2c2c2a",
        "border_strong": "#3d3d39",
        "border_faint": "rgba(255,255,255,.07)",
        "text_primary": SURFACES["dark"]["text_primary"],
        "text_secondary": SURFACES["dark"]["text_secondary"],
        "text_muted": SURFACES["dark"]["text_muted"],
        "shadow": "0 1px 2px rgba(0,0,0,.4)",
        "shadow_raised": "0 1px 3px rgba(0,0,0,.5), 0 6px 16px -6px rgba(0,0,0,.55)",
        "shadow_overlay": "0 12px 32px -8px rgba(0,0,0,.7), 0 2px 6px rgba(0,0,0,.4)",
        "hover": "rgba(255,255,255,.045)",
        "selected": "#14243a",
    },
}

#: Spacing. A four-step geometric scale — enough to build rhythm, few enough
#: that two things separated by different amounts always means something.
SPACE = {"xs": ".25rem", "sm": ".5rem", "md": ".875rem", "lg": "1.5rem", "xl": "2.5rem"}

#: Corner radius. Small and consistent; the spec's warning about "excessive
#: rounded rectangles" is a warning about radius used as decoration.
RADIUS = {"sm": "4px", "md": "6px", "lg": "10px", "pill": "999px"}

#: Motion. Everything in the 150-300ms band the spec calls for, on an ease-out
#: curve so movement decelerates into place rather than stopping dead. Every
#: rule that uses these is suppressed by the reduce-motion setting.
MOTION = {
    "fast": "140ms",
    "base": "200ms",
    "slow": "280ms",
    "ease": "cubic-bezier(.22,.61,.36,1)",
}

#: There is no ninth hue available for an "AI" colour: every candidate violet
#: measures under 4 ΔE from categorical slot 7, which means a reader cannot
#: separate AI chrome from a data series. So AI identity is carried by a mark
#: (✦), by the deep end of the accent's own ramp, and by motion — none of which
#: can be mistaken for a series.
AI_MARK = "✦"

#: Type scale. Deliberately narrow — five sizes is enough to build a hierarchy,
#: and more than that starts reading as decoration.
TYPE_SCALE = {
    "title": "1.75rem",
    "section": "1.0625rem",
    "body": "0.9375rem",
    "caption": "0.8125rem",
    "eyebrow": "0.6875rem",
}

MONO_FAMILY = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'


def ui(mode: str = "light") -> dict[str, str]:
    """Interface tokens for the given mode."""
    return UI["dark" if mode == "dark" else "light"]
