"""Shared interface pieces, and the single stylesheet the whole app is built on.

The visual language exists to answer one question at a glance: *how much weight
does this statement carry?* Four kinds of thing appear throughout the app and
they must not look alike.

============  ===================================================  ==============
Kind          Treatment                                            Why
============  ===================================================  ==============
Measured      No decoration at all. Plain text, tabular numerals.   A fact needs
                                                                   no framing.
Inference     A thin accent rule on the left, and a small label     Marked as a
              naming the method.                                    conclusion,
                                                                   calmly.
Your call     The only filled, raised surface in the app.           It wants an
                                                                   action, so it
                                                                   pulls the eye.
Limit         A muted rule, secondary ink, one size down.           A margin note
                                                                   by a careful
                                                                   author.
============  ===================================================  ==============

Caveats appear on nearly every page — that honesty is the product. Rendered as
warning boxes they read as *something is broken*; rendered as margin notes they
read as *someone has thought about this*. Filled alert styling is reserved for
things that are genuinely wrong: a blocking validation failure, a critical data
problem, a model that did not run.
"""

from __future__ import annotations

import html as _html
from typing import Any

import pandas as pd

from dsai.core.schema import Confidence, Decision, EvidenceKind, Finding, Recommendation
from dsai.viz.theme import MONO_FAMILY, TYPE_SCALE, ui

# Confidence is never conveyed by colour alone: each level carries a distinct
# glyph and its own word.
CONFIDENCE_MARK = {
    Confidence.HIGH: ("●●●", "high"),
    Confidence.MODERATE: ("●●○", "moderate"),
    Confidence.LOW: ("●○○", "low"),
    Confidence.SPECULATIVE: ("○○○", "speculative"),
}
EVIDENCE_LABEL = {
    EvidenceKind.OBSERVED: "Measured in the data",
    EvidenceKind.STATISTICAL: "Statistical test",
    EvidenceKind.MODEL: "Model-derived",
    EvidenceKind.INTERPRETATION: "Platform interpretation",
    EvidenceKind.USER_ASSUMPTION: "Your assumption — not verified",
}
SEVERITY_MARK = {"critical": "▲", "warning": "▲", "info": "■"}

WORKFLOW_STAGES = [
    ("Data", "has_data"),
    ("Context", "has_context"),
    ("Preprocessing", "has_pipeline"),
    ("Analysis", "has_run"),
    ("Models", "has_run"),
    ("Insights", "has_run"),
    ("Recommendations", "has_run"),
]


# --------------------------------------------------------------------------
# the stylesheet
# --------------------------------------------------------------------------

_DARK_RULES = """
[data-testid="stSidebar"], [data-testid="stHeader"] { color-scheme:dark; }
[data-testid="stAppViewContainer"], .stApp { color-scheme:dark; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
  background:var(--sunken) !important; color:var(--ink) !important;
  border-color:var(--border) !important;
}
[data-baseweb="select"] > div, [data-baseweb="popover"] li {
  background:var(--sunken) !important; color:var(--ink) !important;
  border-color:var(--border) !important;
}
[data-testid="stDataFrame"], [data-testid="stDataFrame"] canvas { background:var(--surface); }
[data-testid="stExpander"] { background:var(--sunken); }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] p,
[data-testid="stExpander"] li { color:var(--ink) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--raised) !important; }
[data-testid="stAlert"] { filter:saturate(.75) brightness(.92); }
[data-testid="stJson"] { background:var(--sunken); }
.stButton button, .stDownloadButton button {
  background:var(--sunken); color:var(--ink); border-color:var(--border-strong);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--accent); }
"""


def _stylesheet(mode: str) -> str:
    t = ui(mode)
    dark_rules = _DARK_RULES if mode == "dark" else ""
    return f"""
<style>
:root {{
  --accent:{t['accent']}; --accent-soft:{t['accent_soft']}; --accent-ink:{t['accent_ink']};
  --surface:{t['surface']}; --raised:{t['surface_raised']}; --sunken:{t['surface_sunken']};
  --border:{t['border']}; --border-strong:{t['border_strong']};
  --ink:{t['text_primary']}; --ink-2:{t['text_secondary']}; --ink-3:{t['text_muted']};
  --shadow:{t['shadow']};
  --mono:{MONO_FAMILY};
  --t-title:{TYPE_SCALE['title']}; --t-section:{TYPE_SCALE['section']};
  --t-body:{TYPE_SCALE['body']}; --t-caption:{TYPE_SCALE['caption']};
  --t-eyebrow:{TYPE_SCALE['eyebrow']};
}}

/* ---- ground ---- */
.stApp, [data-testid="stAppViewContainer"] {{ background:var(--surface); }}
[data-testid="stAppViewContainer"] .main .block-container {{
  padding-top:2.75rem; padding-bottom:6rem; max-width:74rem;
}}
html, body, [data-testid="stAppViewContainer"] {{ color:var(--ink); font-size:var(--t-body); }}
[data-testid="stHeader"] {{ background:transparent; }}

/* ---- type scale: Streamlit's defaults are far too loud for a reading interface ---- */
h1, [data-testid="stMarkdownContainer"] h1 {{
  font-size:var(--t-title) !important; font-weight:600 !important;
  letter-spacing:-.021em; line-height:1.2; margin:0 0 .3rem; color:var(--ink);
}}
h2, [data-testid="stMarkdownContainer"] h2 {{
  font-size:1.15rem !important; font-weight:600 !important;
  letter-spacing:-.012em; margin:2.25rem 0 .6rem; color:var(--ink);
}}
h3, [data-testid="stMarkdownContainer"] h3 {{
  font-size:var(--t-section) !important; font-weight:600 !important;
  margin:1.5rem 0 .4rem; color:var(--ink);
}}
p, li, [data-testid="stMarkdownContainer"] {{ color:var(--ink); line-height:1.62; }}
[data-testid="stCaptionContainer"], .stCaption, small {{
  color:var(--ink-3) !important; font-size:var(--t-caption) !important; line-height:1.5;
}}
code, [data-testid="stMarkdownContainer"] code {{
  font-family:var(--mono); font-size:.86em; background:var(--sunken);
  color:var(--ink-2); padding:.08em .34em; border-radius:3px;
  border:1px solid var(--border);
}}
hr {{ border-color:var(--border); margin:2rem 0; }}

/* ---- sidebar: a table of contents, not a control panel ---- */
[data-testid="stSidebar"] {{
  background:var(--sunken); border-right:1px solid var(--border);
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ padding-top:.4rem; }}
[data-testid="stSidebarNav"] a {{
  border-radius:5px; margin:1px 6px; padding:.3rem .55rem !important;
}}
[data-testid="stSidebarNav"] a span {{
  font-size:var(--t-caption); color:var(--ink-2); letter-spacing:.005em;
}}
[data-testid="stSidebarNav"] a:hover {{ background:var(--border); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background:var(--accent-soft); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{
  color:var(--accent-ink); font-weight:600;
}}
/* our own navigation, rendered with st.page_link */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
  border-radius:5px; padding:.26rem .5rem; margin:0; min-height:0;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a p {{
  font-size:var(--t-caption) !important; color:var(--ink-2); margin:0;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background:var(--border); }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{
  background:var(--accent-soft);
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] p {{
  color:var(--accent-ink); font-weight:600;
}}

/* ---- the eyebrow: a small orienting label above a heading ---- */
.dsai-eyebrow {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.085em;
  font-weight:600; color:var(--ink-3); margin-bottom:.45rem;
}}
.dsai-lede {{
  color:var(--ink-2); font-size:.9375rem; line-height:1.6;
  max-width:56ch; margin:0 0 .35rem;
}}

/* ---- workflow stepper ----
   A grid of equal columns rather than a flex row of steps and separator
   elements. Streamlit styles <li> inside its markdown containers, which was
   overriding the separator widths and pushing the row past its container;
   equal grid tracks cannot overflow, and the connector is drawn as a
   pseudo-element so there is nothing extra for those styles to touch. */
.dsai-steps {{
  display:grid; grid-template-columns:repeat(7, 1fr); align-items:start;
  margin:1.4rem 0 2.1rem; padding:0; list-style:none; width:100%;
}}
.dsai-step {{
  position:relative; display:flex; flex-direction:column; align-items:center;
  gap:.36rem; min-width:0; padding:0; margin:0; list-style:none;
}}
.dsai-step::marker {{ content:""; }}
.dsai-step + .dsai-step::before {{
  content:""; position:absolute; top:8px; height:1.5px; border-radius:1px;
  right:calc(50% + 14px); left:calc(-50% + 14px); background:var(--border-strong);
}}
/* A connector is filled only where the path either side of it has actually
   been travelled: the stage before it is done, and the stage after it has been
   reached. Filling on "reached" alone drew a line into work not yet started. */
.dsai-step[data-complete="yes"] + .dsai-step[data-complete="yes"]::before {{
  background:var(--accent);
}}
.dsai-dot {{
  width:17px; height:17px; border-radius:50%; display:grid; place-items:center;
  font-size:9px; line-height:1; font-weight:700; flex:0 0 auto; position:relative;
  border:1.5px solid var(--border-strong); background:var(--surface); color:transparent;
}}
.dsai-step[data-state="done"] .dsai-dot {{
  background:var(--accent); border-color:var(--accent); color:#fff;
}}
.dsai-step[data-state="current"] .dsai-dot {{
  border-color:var(--accent); border-width:2px; background:var(--surface);
  box-shadow:0 0 0 3px var(--accent-soft);
}}
.dsai-step[data-state="current"] .dsai-dot::after {{
  content:""; width:6px; height:6px; border-radius:50%; background:var(--accent);
}}
.dsai-step-label {{
  font-size:.625rem; letter-spacing:.02em; text-align:center; width:100%;
  color:var(--ink-3); line-height:1.25; overflow-wrap:break-word; hyphens:auto;
}}
.dsai-step[data-state="done"] .dsai-step-label {{ color:var(--ink-2); }}
.dsai-step[data-state="current"] .dsai-step-label {{ color:var(--ink); font-weight:700; }}

/* ---- statistics: hierarchy from type, not from boxes ---- */
.dsai-stats {{
  display:flex; gap:1.75rem; flex-wrap:wrap; margin:.3rem 0 1.5rem;
  padding:1.05rem 0 1.15rem; border-top:1px solid var(--border);
  border-bottom:1px solid var(--border);
}}
.dsai-stats > div {{ flex:1 1 8rem; min-width:7rem; max-width:15rem; }}
.dsai-stat-label {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.075em;
  color:var(--ink-3); font-weight:600; margin-bottom:.3rem;
}}
.dsai-stat-value {{
  font-size:1.5rem; font-weight:600; color:var(--ink); line-height:1.1;
  font-variant-numeric:tabular-nums; letter-spacing:-.018em;
}}
.dsai-stat-note {{
  font-size:.75rem; color:var(--ink-3); margin-top:.22rem; line-height:1.4;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}}
.dsai-stat-value--muted {{ color:var(--ink-3); font-weight:500; }}

/* ---- the four kinds of statement ---- */

/* 1. Inference — the platform concluded this. */
.dsai-inference {{
  border-left:2px solid var(--accent); padding:.1rem 0 .1rem .85rem;
  margin:.7rem 0 1rem;
}}
.dsai-inference-label {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; color:var(--accent-ink); display:block; margin-bottom:.28rem;
}}
.dsai-inference-body {{ color:var(--ink); font-size:var(--t-body); line-height:1.6; }}

/* 2. Limit — honest caveat. Recessive on purpose: present, never alarming. */
.dsai-caveat {{
  border-left:2px solid var(--border-strong); padding:.05rem 0 .05rem .85rem;
  margin:.55rem 0; color:var(--ink-2); font-size:var(--t-caption); line-height:1.55;
  max-width:64ch;
}}
.dsai-caveat-label {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; color:var(--ink-3); margin-right:.5rem;
}}

/* 3. Your call — the only filled, raised surface. */
.dsai-decide {{
  background:var(--raised); border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:6px;
  padding:.95rem 1.1rem; margin:.85rem 0; box-shadow:var(--shadow);
}}
.dsai-decide-label {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; color:var(--accent-ink); display:block; margin-bottom:.35rem;
}}

/* 4. Measured needs no class — plain text is the treatment. */

/* ---- cards ---- */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color:var(--border) !important; border-radius:7px !important;
  background:var(--raised); box-shadow:var(--shadow);
}}
.dsai-card-title {{
  font-size:1rem; font-weight:600; color:var(--ink); line-height:1.4;
  letter-spacing:-.008em; margin-bottom:.2rem;
}}
.dsai-meta {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.07em;
  color:var(--ink-3); font-weight:600; margin-bottom:.55rem;
}}
.dsai-conf {{ font-family:var(--mono); letter-spacing:.06em; font-size:.72rem; }}

/* ---- expanders: quiet until opened ---- */
[data-testid="stExpander"] {{
  border:1px solid var(--border); border-radius:6px; background:var(--surface);
  margin-bottom:.45rem;
}}
[data-testid="stExpander"] summary {{ font-size:.9rem; padding:.15rem 0; }}
[data-testid="stExpander"] summary p {{ font-size:.9rem !important; color:var(--ink); }}
[data-testid="stExpander"]:hover {{ border-color:var(--border-strong); }}

/* ---- tabs ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap:1.5rem; border-bottom:1px solid var(--border);
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
  padding:.35rem 0; font-size:var(--t-caption); letter-spacing:.012em; color:var(--ink-3);
}}
[data-testid="stTabs"] [aria-selected="true"] {{ color:var(--ink) !important; font-weight:600; }}

/* ---- alerts: reserved for what is genuinely wrong, and flatter than default ---- */
[data-testid="stAlert"] {{
  border-radius:6px; border:1px solid var(--border); box-shadow:none;
  font-size:var(--t-caption); padding:.6rem .8rem;
}}
[data-testid="stAlert"] svg {{ opacity:.7; }}
[data-testid="stAlert"] p {{ font-size:var(--t-caption) !important; line-height:1.55; }}

/* ---- tables ---- */
[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:6px; }}
[data-testid="stDataFrame"] * {{ font-variant-numeric:tabular-nums; }}
.dsai-table {{ width:100%; border-collapse:collapse; font-size:var(--t-caption); margin:.5rem 0 1rem; }}
.dsai-table th {{
  text-align:left; font-size:var(--t-eyebrow); text-transform:uppercase;
  letter-spacing:.07em; color:var(--ink-3); font-weight:600;
  padding:.4rem .7rem .4rem 0; border-bottom:1px solid var(--border-strong);
}}
.dsai-table td {{
  padding:.42rem .7rem .42rem 0; border-bottom:1px solid var(--border);
  color:var(--ink); font-variant-numeric:tabular-nums; vertical-align:top;
}}
.dsai-table tr:last-child td {{ border-bottom:none; }}

/* ---- buttons ---- */
.stButton button, .stDownloadButton button {{
  border-radius:6px; border:1px solid var(--border-strong); font-size:var(--t-caption);
  font-weight:500; padding:.36rem .85rem; transition:none;
}}
.stButton button:hover, .stDownloadButton button:hover {{
  border-color:var(--accent); color:var(--accent-ink);
}}
.stButton button[kind="primary"] {{
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600;
}}
.stButton button[kind="primary"]:hover {{
  background:var(--accent-ink); border-color:var(--accent-ink); color:#fff;
}}

/* ---- inputs ---- */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {{
  border-radius:6px !important; font-size:var(--t-caption) !important;
  background:var(--surface) !important;
}}
label, [data-testid="stWidgetLabel"] p {{
  font-size:var(--t-caption) !important; color:var(--ink-2) !important; font-weight:500;
}}

/* ---- trace ---- */
[data-testid="stCode"] {{ background:var(--sunken); border:1px solid var(--border); border-radius:6px; }}
[data-testid="stCode"] code {{
  font-size:.78rem; line-height:1.65; background:none; border:none; color:var(--ink-2);
}}

/* ---- empty state ---- */
.dsai-empty {{
  border:1px dashed var(--border-strong); border-radius:7px; padding:2rem 1.5rem;
  text-align:center; color:var(--ink-3); font-size:var(--t-caption); margin:1.5rem 0;
}}
.dsai-empty strong {{ color:var(--ink-2); display:block; margin-bottom:.28rem; font-size:.9rem; }}

/* ---- dark mode: config.toml sets a static base, so the widget internals
       Streamlit paints itself have to be reached here ---- */
{dark_rules}

@media (max-width:1280px) {{
  [data-testid="stAppViewContainer"] .main .block-container {{ padding-left:2rem; padding-right:2rem; }}
  .dsai-stats {{ gap:1.8rem; }}
}}
</style>
"""


def apply_theme(mode: str = "light") -> None:
    """Inject the stylesheet. Every page calls this once, at the top.

    All custom styling lives here rather than being scattered across pages, so
    there is exactly one place to change how the app looks.
    """
    import streamlit as st

    st.markdown(_stylesheet(mode), unsafe_allow_html=True)


def _navigation() -> None:
    """Grouped navigation, rendered by the app rather than by Streamlit.

    Streamlit's automatic sidebar nav hides everything past the tenth page
    behind a "view more" button, which put three pages out of sight. Rendering
    it here keeps every page visible and makes the grouping explicit.
    """
    import streamlit as st

    sections = st.session_state.get("_dsai_sections")
    if not sections:
        return
    for heading, pages in sections.items():
        if heading:
            st.markdown(
                f'<div class="dsai-eyebrow" style="margin:.9rem 0 .3rem">{_esc(heading)}</div>',
                unsafe_allow_html=True,
            )
        for page in pages:
            st.page_link(page, label=page.title)
    st.divider()


def sidebar_chrome(state: Any) -> None:
    """Shared sidebar: what is loaded, and the appearance toggle.

    Rendered on every page rather than only the landing page, because a
    multipage Streamlit app runs each page in isolation and the user needs the
    same orientation wherever they are.
    """
    import streamlit as st

    with st.sidebar:
        _navigation()
        st.markdown('<div class="dsai-eyebrow">Workspace</div>', unsafe_allow_html=True)
        if getattr(state, "has_data", False):
            st.markdown(
                f'<div style="font-size:.84rem;color:var(--ink);font-weight:600;'
                f'margin-bottom:.15rem">{_esc(state.dataset_name or "dataset")}</div>'
                f'<div style="font-size:.75rem;color:var(--ink-3)">'
                f"{len(state.frame):,} rows &middot; {state.frame.shape[1]} columns</div>",
                unsafe_allow_html=True,
            )
            if state.profile is not None:
                st.markdown(
                    f'<div style="font-size:.75rem;color:var(--ink-3);margin-top:.2rem">'
                    f"Quality {state.profile.quality_score}/100</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:.78rem;color:var(--ink-3)">No dataset loaded</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        chosen = st.radio(
            "Appearance", ["light", "dark"],
            index=0 if state.theme == "light" else 1,
            horizontal=True, key="dsai_theme_toggle",
            help="Applies to the interface and the charts together.",
        )
        if chosen != state.theme:
            state.theme = chosen
            st.rerun()


def _esc(text: Any) -> str:
    return _html.escape(str(text))


def _rich(text: Any) -> str:
    """Escape, then restore the small markdown subset the analysis text uses.

    Findings and caveats are written with *emphasis* and `code` because they are
    also rendered into Markdown reports. Escaping alone would print the asterisks.
    """
    import re

    out = _html.escape(str(text))
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"`([^`]+?)`", r"<code>\1</code>", out)
    return out


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "", step: str = "") -> None:
    import streamlit as st

    block = []
    if step:
        block.append(f'<div class="dsai-eyebrow">{_esc(step)}</div>')
    block.append(f"<h1>{_esc(title)}</h1>")
    if subtitle:
        block.append(f'<p class="dsai-lede">{_esc(subtitle)}</p>')
    st.markdown("".join(block), unsafe_allow_html=True)


def workflow_nav(current: str, state: Any = None) -> None:
    """The seven stages as a real progress indicator.

    Completion is read from the workspace, not assumed from position, so the
    stepper reflects what has actually happened rather than where the user
    happens to be standing.
    """
    import streamlit as st

    done: dict[str, bool] = {}
    if state is not None:
        context = getattr(state, "context", None)
        done = {
            "has_data": bool(getattr(state, "has_data", False)),
            "has_context": bool(context is not None and not context.is_empty),
            "has_pipeline": getattr(state, "pipeline", None) is not None,
            "has_run": bool(getattr(state, "has_run", False)),
        }

    parts = ['<ol class="dsai-steps">']
    for index, (label, flag) in enumerate(WORKFLOW_STAGES):
        is_current = label.lower() == current.lower()
        complete = done.get(flag, False)
        # The current stage reads as "current" even once complete: it is where
        # the user is, and that matters more than whether it is ticked.
        state_name = "current" if is_current else ("done" if complete else "todo")
        aria = ' aria-current="step"' if is_current else ""
        parts.append(
            f'<li class="dsai-step" data-state="{state_name}" '
            f'data-complete="{"yes" if complete else "no"}"{aria}>'
            f'<span class="dsai-dot">{"✓" if state_name == "done" else ""}</span>'
            f'<span class="dsai-step-label">{_esc(label)}</span>'
            f"</li>"
        )
    parts.append("</ol>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# the four kinds of statement
# --------------------------------------------------------------------------

def inference(body: str, label: str = "Model-derived") -> None:
    """Something the platform concluded, marked as a conclusion."""
    import streamlit as st

    st.markdown(
        f'<div class="dsai-inference"><span class="dsai-inference-label">{_esc(label)}</span>'
        f'<span class="dsai-inference-body">{_rich(body)}</span></div>',
        unsafe_allow_html=True,
    )


def caveat(text: str, label: str = "Limit") -> None:
    """An honest limit on what was just said.

    Deliberately not an alert. These appear on almost every page — that is the
    product working as intended — and a page of amber boxes reads as failure
    rather than as care.
    """
    import streamlit as st

    st.markdown(
        f'<div class="dsai-caveat"><span class="dsai-caveat-label">{_esc(label)}</span>{_rich(text)}</div>',
        unsafe_allow_html=True,
    )


def decision_needed(body: str, label: str = "Your call") -> None:
    """Something waiting on the user. The only raised surface in the app."""
    import streamlit as st

    st.markdown(
        f'<div class="dsai-decide"><span class="dsai-decide-label">{_esc(label)}</span>{_rich(body)}</div>',
        unsafe_allow_html=True,
    )


def confidence_badge(confidence: Confidence) -> str:
    mark, word = CONFIDENCE_MARK.get(confidence, ("○○○", "unknown"))
    return f"{mark} {word}"


def _confidence_html(confidence: Confidence) -> str:
    mark, word = CONFIDENCE_MARK.get(confidence, ("○○○", "unknown"))
    return f'<span class="dsai-conf">{mark}</span> {word}'


# --------------------------------------------------------------------------
# data display
# --------------------------------------------------------------------------

def metric_row(items: list[tuple[str, Any, str]]) -> None:
    """Figures in a row, separated by rules rather than boxed in cards."""
    import streamlit as st

    cells = []
    for label, value, note in items:
        muted = value in (None, "—", "")
        cells.append(
            f"<div>"
            f'<div class="dsai-stat-label">{_esc(label)}</div>'
            f'<div class="dsai-stat-value{" dsai-stat-value--muted" if muted else ""}">'
            f"{_esc(value if not muted else '—')}</div>"
            + (f'<div class="dsai-stat-note" title="{_esc(note)}">{_esc(note)}</div>' if note else "")
            + "</div>"
        )
    st.markdown(f'<div class="dsai-stats">{"".join(cells)}</div>', unsafe_allow_html=True)


def simple_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    """A small table styled as type, for short summaries where a grid is heavy."""
    import streamlit as st

    if not rows:
        return
    columns = columns or list(rows[0])
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_rich(row.get(c, ''))}</td>" for c in columns) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<table class="dsai-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


def dataframe(frame: pd.DataFrame, **kwargs: Any) -> None:
    import streamlit as st

    st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)


def empty_state(title: str, detail: str = "") -> None:
    import streamlit as st

    st.markdown(
        f'<div class="dsai-empty"><strong>{_esc(title)}</strong>{_esc(detail)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# domain cards
# --------------------------------------------------------------------------

def finding_card(finding: Finding, expanded: bool = False) -> None:
    import streamlit as st

    with st.expander(finding.title, expanded=expanded):
        st.markdown(
            f'<div class="dsai-meta">{_esc(EVIDENCE_LABEL.get(finding.kind, finding.kind.value))}'
            f' &nbsp;·&nbsp; confidence {_confidence_html(finding.confidence)}</div>',
            unsafe_allow_html=True,
        )
        st.write(finding.detail)
        if finding.evidence:
            simple_table([{"Evidence": item} for item in finding.evidence[:6]])
        for item in finding.caveats[:3]:
            caveat(item)


def recommendation_card(recommendation: Recommendation, index: int = 0) -> None:
    import streamlit as st

    with st.container(border=True):
        st.markdown(
            f'<div class="dsai-card-title">{_esc(recommendation.action)}</div>'
            f'<div class="dsai-meta">{_esc(recommendation.category.replace("_", " "))}'
            f' &nbsp;·&nbsp; confidence {_confidence_html(recommendation.confidence)}</div>',
            unsafe_allow_html=True,
        )
        st.write(recommendation.reason)
        if recommendation.expected_impact:
            st.markdown(
                f'<div class="dsai-caveat"><span class="dsai-caveat-label">Impact</span>'
                f"{_rich(recommendation.expected_impact)}</div>",
                unsafe_allow_html=True,
            )
        if recommendation.evidence:
            with st.expander("Evidence", expanded=False):
                simple_table([{"Evidence": item} for item in recommendation.evidence])
                if recommendation.traceable_to:
                    st.caption("Traces back to: " + ", ".join(recommendation.traceable_to))
        for item in recommendation.caveats[:2]:
            caveat(item)


def decision_panel(decisions: list[Decision], stage: str | None = None) -> None:
    """The decision log: what was chosen, why, and what was set aside."""
    import streamlit as st

    shown = [d for d in decisions if stage is None or d.stage == stage]
    if not shown:
        empty_state("No decisions recorded for this stage yet.")
        return
    for decision in shown:
        with st.expander(decision.decision, expanded=False):
            st.markdown(
                f'<div class="dsai-meta">{_esc(decision.stage.replace("_", " "))}'
                f' &nbsp;·&nbsp; confidence {_confidence_html(decision.confidence)}'
                + (" &nbsp;·&nbsp; overridden by you" if decision.overridden_by_user else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.write(decision.reason)
            if decision.evidence:
                simple_table([{"Evidence": item} for item in decision.evidence])
            if decision.rejected:
                st.markdown('<div class="dsai-meta">Considered and set aside</div>',
                            unsafe_allow_html=True)
                simple_table(
                    [{"Option": r["option"], "Why not": r["reason"]} for r in decision.rejected],
                    ["Option", "Why not"],
                )


def quality_issues(profile: Any) -> None:
    import streamlit as st

    if not profile.quality_issues:
        st.markdown(
            '<div class="dsai-caveat"><span class="dsai-caveat-label">Checked</span>'
            "No data-quality problems were detected.</div>",
            unsafe_allow_html=True,
        )
        return
    for issue in sorted(profile.quality_issues,
                        key=lambda i: {"critical": 0, "warning": 1, "info": 2}.get(i.severity, 3)):
        mark = SEVERITY_MARK.get(issue.severity, "■")
        with st.expander(f"{mark}  {issue.message}", expanded=issue.severity == "critical"):
            st.markdown(f'<div class="dsai-meta">{_esc(issue.severity)}</div>', unsafe_allow_html=True)
            if issue.columns:
                st.markdown("Columns: " + ", ".join(f"`{c}`" for c in issue.columns[:20]))
            if issue.suggested_action:
                inference(issue.suggested_action, label="Suggested action")
            if issue.detail:
                st.json(issue.detail, expanded=False)


def trace_view(trace: Any, limit: int = 60) -> None:
    import streamlit as st

    icons = {"done": "✓", "running": "…", "warning": "!", "failed": "✗", "skipped": "–"}
    lines = []
    for event in trace.events[-limit:]:
        timing = f"  ({event.elapsed_s:.2f}s)" if event.elapsed_s else ""
        detail = f" — {event.detail}" if event.detail else ""
        lines.append(f"{icons.get(event.status, '·')} {event.step}{detail}{timing}")
    st.code("\n".join(lines), language=None)


def override_notice(message: str = "") -> None:
    caveat(
        message or "Every choice on this page is a recommendation. Change anything you disagree "
                   "with — your override is recorded in the decision log.",
        label="You decide",
    )


# --------------------------------------------------------------------------
# guards and notices
# --------------------------------------------------------------------------

def require_data(state: Any) -> bool:
    if not state.has_data:
        empty_state("No dataset loaded", "Load one on the Data page to begin.")
        return False
    return True


def require_run(state: Any) -> bool:
    if not state.has_run:
        empty_state("No analysis yet", "Run one on the Analysis page to see results here.")
        return False
    return True


def show_notices(state: Any) -> None:
    """Transient feedback about what just happened. Genuinely an alert."""
    import streamlit as st

    for level, message in state.take_notices():
        {"success": st.success, "warning": st.warning, "error": st.error}.get(level, st.info)(message)
