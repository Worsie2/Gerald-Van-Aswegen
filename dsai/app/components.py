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
from dsai.viz.theme import (
    AI_MARK, MONO_FAMILY, MOTION, RADIUS, SPACE, STATUS, TYPE_SCALE, ui,
)

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

# Streamlit's own widget internals are painted from the static base in
# config.toml, which is light. Dark mode is therefore the override: these rules
# reach the widgets the stylesheet's tokens cannot. The light block below exists
# for the few places our own plane differs from Streamlit's default.
_LIGHT_RULES = """
[data-testid="stSidebar"], [data-testid="stHeader"] { color-scheme:light; }
[data-testid="stAppViewContainer"], .stApp { color-scheme:light; }
[data-testid="stHeader"] { background:var(--plane) !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
  background:var(--surface) !important; color:var(--ink) !important;
  border-color:var(--border) !important;
}
[data-baseweb="select"] > div, [data-baseweb="popover"] li,
[data-baseweb="popover"] ul {
  background:var(--surface) !important; color:var(--ink) !important;
  border-color:var(--border) !important;
}
[data-baseweb="popover"] li:hover { background:var(--hover) !important; }
[data-testid="stDataFrame"], [data-testid="stDataFrame"] canvas { background:var(--surface); }
[data-testid="stExpander"] { background:var(--surface); }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] p,
[data-testid="stExpander"] li { color:var(--ink) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface) !important; }
[data-testid="stJson"] { background:var(--sunken); }
.stButton button, .stDownloadButton button {
  background:var(--surface); color:var(--ink); border-color:var(--border-strong);
}
.stSlider [data-baseweb="slider"] div[role="slider"] { border-color:var(--border-strong); }
"""

# Reconstructing dark from a light base: every widget internal Streamlit paints
# for itself has to be reached by hand.
_DARK_RULES = """
[data-testid="stSidebar"], [data-testid="stHeader"] { color-scheme:dark; }
[data-testid="stAppViewContainer"], .stApp { color-scheme:dark; }
[data-testid="stHeader"] { background:var(--plane) !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
  background:var(--sunken) !important; color:var(--ink) !important;
  border-color:var(--border) !important;
}
[data-baseweb="select"] > div, [data-baseweb="popover"] li,
[data-baseweb="popover"] ul {
  background:var(--raised) !important; color:var(--ink) !important;
  border-color:var(--border) !important;
}
[data-baseweb="popover"] li:hover { background:var(--hover) !important; }
[data-testid="stDataFrame"], [data-testid="stDataFrame"] canvas { background:var(--surface); }
[data-testid="stExpander"] { background:var(--surface); }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface) !important; }
[data-testid="stAlert"] { filter:saturate(.8) brightness(.95); }
[data-testid="stJson"] { background:var(--sunken); }
.stButton button, .stDownloadButton button {
  background:var(--surface); color:var(--ink); border-color:var(--border-strong);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--accent); }
"""


def _stylesheet(mode: str) -> str:
    t = ui(mode)
    mode_rules = _LIGHT_RULES if mode == "light" else _DARK_RULES
    return f"""
<style>
:root {{
  --accent:{t['accent']}; --accent-soft:{t['accent_soft']}; --accent-ink:{t['accent_ink']};
  --accent-deep:{t['accent_deep']};
  --plane:{t['plane']}; --rail:{t['rail']};
  --surface:{t['surface']}; --raised:{t['surface_raised']}; --sunken:{t['surface_sunken']};
  --border:{t['border']}; --border-strong:{t['border_strong']}; --border-faint:{t['border_faint']};
  --ink:{t['text_primary']}; --ink-2:{t['text_secondary']}; --ink-3:{t['text_muted']};
  --shadow:{t['shadow']}; --shadow-raised:{t['shadow_raised']};
  --shadow-overlay:{t['shadow_overlay']};
  --hover:{t['hover']}; --selected:{t['selected']};
  --good:{STATUS['good']}; --warning:{STATUS['warning']};
  --serious:{STATUS['serious']}; --critical:{STATUS['critical']};
  --mono:{MONO_FAMILY};
  --t-title:{TYPE_SCALE['title']}; --t-section:{TYPE_SCALE['section']};
  --t-body:{TYPE_SCALE['body']}; --t-caption:{TYPE_SCALE['caption']};
  --t-eyebrow:{TYPE_SCALE['eyebrow']};
  --s-xs:{SPACE['xs']}; --s-sm:{SPACE['sm']}; --s-md:{SPACE['md']};
  --s-lg:{SPACE['lg']}; --s-xl:{SPACE['xl']};
  --r-sm:{RADIUS['sm']}; --r-md:{RADIUS['md']}; --r-lg:{RADIUS['lg']}; --r-pill:{RADIUS['pill']};
  --m-fast:{MOTION['fast']}; --m-base:{MOTION['base']}; --m-slow:{MOTION['slow']};
  --ease:{MOTION['ease']};
}}

/* ---- ground ----
   Five surfaces, each one step of elevation: the plane the app sits on, the
   rail beside it, the sheet content sits on, the thing lifted off that sheet,
   and the well things sit inside. Contrast between them is deliberately small —
   the hierarchy is legible without any of it shouting. */
.stApp, [data-testid="stAppViewContainer"] {{ background:var(--plane); }}
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
  background:var(--rail); border-right:1px solid var(--border);
}}
[data-testid="stSidebar"] > div {{ padding-top:.6rem; }}
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
  border-radius:var(--r-sm); padding:.28rem .55rem; margin:0; min-height:0;
  transition:background var(--m-fast) var(--ease), color var(--m-fast) var(--ease);
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a p {{
  font-size:var(--t-caption) !important; color:var(--ink-2); margin:0;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background:var(--hover); }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{
  background:var(--selected); box-shadow:inset 2px 0 0 var(--accent);
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
  border-color:var(--border) !important; border-radius:var(--r-md) !important;
  background:var(--surface); box-shadow:var(--shadow);
  transition:border-color var(--m-base) var(--ease), box-shadow var(--m-base) var(--ease);
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{ box-shadow:var(--shadow-raised); }}
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
  border:1px solid var(--border); border-radius:var(--r-md); background:var(--surface);
  margin-bottom:.45rem;
  transition:border-color var(--m-fast) var(--ease), background var(--m-fast) var(--ease);
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
  transition:color var(--m-fast) var(--ease);
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{ color:var(--ink-2); }}
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
  border-radius:var(--r-md); border:1px solid var(--border-strong); font-size:var(--t-caption);
  font-weight:500; padding:.36rem .85rem; background:var(--surface);
  transition:border-color var(--m-fast) var(--ease), background var(--m-fast) var(--ease),
             color var(--m-fast) var(--ease), transform var(--m-fast) var(--ease);
}}
.stButton button:hover, .stDownloadButton button:hover {{
  border-color:var(--accent); color:var(--accent-ink); background:var(--hover);
}}
.stButton button:active, .stDownloadButton button:active {{ transform:translateY(.5px); }}
.stButton button[kind="primary"] {{
  background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600;
}}
.stButton button[kind="primary"]:hover {{
  background:var(--accent-ink); border-color:var(--accent-ink); color:#fff;
}}

/* ---- widget surfaces ----
   Streamlit paints select, input and textarea backgrounds from
   secondaryBackgroundColor in config.toml — one static value that cannot follow
   a runtime theme change. Repainting them from our own tokens here means one
   rule serves both modes, because the token already differs per mode.

   Targeted by test id rather than by data-baseweb: the element Streamlit
   actually paints does not carry that attribute in every version, and a
   selector that silently stops matching is how a whole mode ends up rendering
   on the wrong ground. */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stTextArea"] > div > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stDateInput"] > div > div,
[data-testid="stTimeInput"] > div > div,
[data-baseweb="select"] > div {{
  background:var(--sunken) !important;
  border-color:var(--border) !important;
  color:var(--ink) !important;
}}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input,
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {{ color:var(--ink) !important; }}
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] {{
  background:var(--raised) !important; border:1px solid var(--border) !important;
  box-shadow:var(--shadow-overlay) !important;
}}
li[role="option"] {{ background:transparent !important; color:var(--ink) !important; }}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
  background:var(--hover) !important;
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

/* ======================================================================
   The premium layer: chrome, AI surfaces, and the pieces the workspace is
   assembled from. Every component below is built from the same tokens as
   everything above it: spacing from the --s tokens, radius from --r, motion
   from --m and --ease. One design language rather than a collection of
   one-off treatments.
   ====================================================================== */

/* ---- top strip: project, command bar, state ---- */
.dsai-top {{
  display:flex; align-items:center; gap:var(--s-md);
  padding:.55rem 0 .7rem; margin:-1.4rem 0 var(--s-lg);
  border-bottom:1px solid var(--border);
}}
.dsai-top-project {{ display:flex; align-items:baseline; gap:.55rem; min-width:0; }}
.dsai-top-brand {{
  font-size:var(--t-caption); font-weight:700; color:var(--ink);
  letter-spacing:-.01em; white-space:nowrap;
}}
.dsai-top-name {{
  font-size:var(--t-caption); color:var(--ink-3); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; max-width:22ch;
}}
.dsai-top-spacer {{ flex:1 1 auto; }}
.dsai-top-state {{
  display:inline-flex; align-items:center; gap:.4rem; font-size:var(--t-eyebrow);
  letter-spacing:.06em; text-transform:uppercase; font-weight:600; color:var(--ink-3);
  white-space:nowrap;
}}
.dsai-kbd {{
  font-family:var(--mono); font-size:.68rem; color:var(--ink-3);
  border:1px solid var(--border-strong); border-radius:var(--r-sm);
  padding:.05rem .3rem; background:var(--sunken); white-space:nowrap;
}}

/* ---- badges: state in a word and a shape, never in a colour alone ---- */
.dsai-badge {{
  display:inline-flex; align-items:center; gap:.3rem; font-size:var(--t-eyebrow);
  font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  padding:.12rem .45rem; border-radius:var(--r-sm); border:1px solid var(--border);
  color:var(--ink-2); background:var(--sunken); white-space:nowrap;
}}
.dsai-badge[data-tone="accent"] {{
  color:var(--accent-ink); border-color:var(--accent); background:var(--accent-soft);
}}
.dsai-badge[data-tone="good"] {{ color:var(--good); border-color:var(--good); background:transparent; }}
.dsai-badge[data-tone="warning"] {{ color:var(--warning); border-color:var(--warning); background:transparent; }}
.dsai-badge[data-tone="critical"] {{ color:var(--critical); border-color:var(--critical); background:transparent; }}

/* ---- the AI surfaces ----
   There is no ninth hue available for "AI" — every candidate sits too close to
   a categorical slot for a reader to separate them. So AI is marked by the ✦,
   by the deep end of the accent's own ramp, and by motion. None of those can be
   mistaken for a data series. */
.dsai-ai {{
  max-width:54rem;
  border:1px solid var(--border); border-left:2px solid var(--accent);
  border-radius:var(--r-md); background:var(--surface);
  padding:var(--s-md) 1.1rem; margin:var(--s-md) 0 var(--s-lg);
  box-shadow:var(--shadow);
}}
.dsai-ai-head {{
  display:flex; align-items:center; gap:.45rem; margin-bottom:.6rem;
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; color:var(--accent-ink);
}}
.dsai-ai-mark {{ font-size:.8rem; line-height:1; }}
.dsai-ai-body {{ color:var(--ink); font-size:var(--t-body); line-height:1.62; max-width:64ch; }}
.dsai-ai-body p {{ margin:0 0 .5rem; }}
.dsai-ai-rule {{ border:0; border-top:1px solid var(--border); margin:.85rem 0; }}
.dsai-ai-why {{
  font-size:var(--t-caption); color:var(--ink-2); line-height:1.55; max-width:64ch;
}}
.dsai-ai-why strong {{ color:var(--ink); font-weight:600; }}

/* AI at work: the mark breathes rather than a spinner going round. Suppressed
   entirely by reduce-motion, where the stage list alone carries the state. */
.dsai-ai[data-busy="yes"] .dsai-ai-mark {{ animation:dsai-pulse 1.9s var(--ease) infinite; }}
@keyframes dsai-pulse {{ 0%,100% {{ opacity:.35; }} 50% {{ opacity:1; }} }}

/* ---- staged progress: named work, not a spinner ---- */
.dsai-stages {{ list-style:none; margin:.6rem 0 .2rem; padding:0; }}
.dsai-stage {{
  display:flex; align-items:baseline; gap:.6rem; padding:.2rem 0;
  font-size:var(--t-caption); line-height:1.5; color:var(--ink-3);
}}
.dsai-stage-mark {{
  font-family:var(--mono); font-size:.72rem; width:1.1rem; flex:0 0 auto; text-align:center;
}}
.dsai-stage[data-state="done"] {{ color:var(--ink-2); }}
.dsai-stage[data-state="done"] .dsai-stage-mark {{ color:var(--good); }}
.dsai-stage[data-state="running"] {{ color:var(--ink); font-weight:600; }}
.dsai-stage[data-state="running"] .dsai-stage-mark {{
  color:var(--accent); animation:dsai-pulse 1.4s var(--ease) infinite;
}}
.dsai-stage[data-state="failed"] {{ color:var(--critical); }}
.dsai-stage[data-state="warning"] .dsai-stage-mark {{ color:var(--warning); }}
.dsai-stage-detail {{ color:var(--ink-3); font-weight:400; }}

/* ---- skeletons: never a blank screen ---- */
.dsai-skeleton {{ display:block; margin:.45rem 0; }}
.dsai-skeleton span {{
  display:block; height:.7rem; border-radius:var(--r-sm); margin-bottom:.45rem;
  background:linear-gradient(90deg, var(--sunken) 0%, var(--border) 50%, var(--sunken) 100%);
  background-size:200% 100%; animation:dsai-shimmer 1.4s linear infinite;
}}
@keyframes dsai-shimmer {{ 0% {{ background-position:200% 0; }} 100% {{ background-position:-200% 0; }} }}

/* ---- quality bars ---- */
.dsai-bars {{ margin:.6rem 0 1.2rem; }}
.dsai-bar-row {{
  display:grid; grid-template-columns:minmax(7rem,11rem) 1fr 3.2rem;
  align-items:center; gap:var(--s-md); padding:.32rem 0;
}}
.dsai-bar-label {{ font-size:var(--t-caption); color:var(--ink-2); }}
.dsai-bar-track {{
  height:6px; border-radius:var(--r-pill); background:var(--sunken);
  border:1px solid var(--border-faint); overflow:hidden;
}}
.dsai-bar-fill {{
  height:100%; border-radius:var(--r-pill); background:var(--accent);
  transition:width var(--m-slow) var(--ease);
}}
.dsai-bar-row[data-tone="good"] .dsai-bar-fill {{ background:var(--good); }}
.dsai-bar-row[data-tone="warning"] .dsai-bar-fill {{ background:var(--warning); }}
.dsai-bar-row[data-tone="critical"] .dsai-bar-fill {{ background:var(--critical); }}
.dsai-bar-value {{
  font-size:var(--t-caption); color:var(--ink); text-align:right;
  font-variant-numeric:tabular-nums; font-weight:600;
}}

/* ---- the hero figure ---- */
.dsai-hero {{ display:flex; align-items:baseline; gap:.7rem; margin:.2rem 0 .1rem; }}
.dsai-hero-value {{
  font-size:2.75rem; font-weight:600; letter-spacing:-.03em; line-height:1;
  color:var(--ink);
}}
.dsai-hero-unit {{ font-size:1rem; color:var(--ink-3); font-weight:500; }}

/* ---- pipeline graph: the chain, drawn ---- */
.dsai-pipe {{ margin:.6rem 0 1.3rem; }}
.dsai-pipe-node {{
  position:relative; border:1px solid var(--border); border-radius:var(--r-md);
  background:var(--surface); padding:.55rem .8rem; margin:0 0 1.35rem;
  transition:border-color var(--m-fast) var(--ease), box-shadow var(--m-fast) var(--ease);
}}
.dsai-pipe-node:hover {{ border-color:var(--border-strong); box-shadow:var(--shadow); }}
.dsai-pipe-node:not(:last-child)::after {{
  content:""; position:absolute; left:1.6rem; top:100%; width:1.5px; height:1.35rem;
  background:var(--border-strong);
}}
.dsai-pipe-node:not(:last-child)::before {{
  content:""; position:absolute; left:calc(1.6rem - 3px); top:calc(100% + 1rem);
  border-left:4px solid transparent; border-right:4px solid transparent;
  border-top:5px solid var(--border-strong);
}}
.dsai-pipe-node[data-kind="source"] {{ border-style:dashed; background:var(--sunken); }}
.dsai-pipe-node[data-kind="model"] {{ border-left:2px solid var(--accent); }}
.dsai-pipe-row {{ display:flex; align-items:baseline; gap:.6rem; }}
.dsai-pipe-index {{
  font-family:var(--mono); font-size:.7rem; color:var(--ink-3); width:1.5rem; flex:0 0 auto;
}}
.dsai-pipe-name {{ font-size:var(--t-caption); font-weight:600; color:var(--ink); }}
.dsai-pipe-scope {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.06em;
  color:var(--ink-3); font-weight:600;
}}
.dsai-pipe-detail {{
  font-size:var(--t-caption); color:var(--ink-3); margin:.18rem 0 0 2.1rem; line-height:1.5;
}}

/* ---- ranked list: models, insights, recommendations ---- */
.dsai-rank {{
  display:grid; grid-template-columns:2.4rem 1fr; gap:.2rem var(--s-md);
  padding:var(--s-md) 0; border-top:1px solid var(--border); align-items:start;
}}
.dsai-rank:last-child {{ border-bottom:1px solid var(--border); }}
.dsai-rank-index {{
  font-family:var(--mono); font-size:1.05rem; font-weight:600; color:var(--ink-3);
  line-height:1.3; font-variant-numeric:tabular-nums;
}}
.dsai-rank[data-lead="yes"] .dsai-rank-index {{ color:var(--accent); }}
.dsai-rank-head {{ display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }}
.dsai-rank-title {{
  font-size:1rem; font-weight:600; color:var(--ink); letter-spacing:-.008em; line-height:1.35;
}}
.dsai-rank-body {{ font-size:var(--t-caption); color:var(--ink-2); line-height:1.6; max-width:70ch; }}
.dsai-rank-metrics {{
  display:flex; gap:var(--s-lg); flex-wrap:wrap; margin:.45rem 0 .1rem;
}}
.dsai-rank-metric {{ min-width:5rem; }}
.dsai-rank-metric dt {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.07em;
  color:var(--ink-3); font-weight:600; margin:0 0 .1rem;
}}
.dsai-rank-metric dd {{
  margin:0; font-size:.9375rem; font-weight:600; color:var(--ink);
  font-variant-numeric:tabular-nums;
}}

/* ---- leaderboard ---- */
.dsai-board td[data-lead="yes"], .dsai-board tr[data-lead="yes"] td {{
  background:var(--selected);
}}
.dsai-board tr[data-lead="yes"] td:first-child {{ box-shadow:inset 2px 0 0 var(--accent); }}
.dsai-num {{ text-align:right; font-variant-numeric:tabular-nums; }}

/* ---- error state: what went wrong, and the way out ---- */
.dsai-error {{
  border:1px solid var(--critical); border-radius:var(--r-md); background:var(--surface);
  padding:1rem 1.1rem; margin:var(--s-md) 0 var(--s-lg);
}}
.dsai-error-title {{
  font-size:1rem; font-weight:600; color:var(--ink); margin-bottom:.3rem;
  display:flex; align-items:center; gap:.45rem;
}}
.dsai-error-body {{ font-size:var(--t-caption); color:var(--ink-2); line-height:1.6; max-width:64ch; }}
.dsai-error-fix {{
  margin-top:.7rem; padding-top:.7rem; border-top:1px solid var(--border);
  font-size:var(--t-caption); color:var(--ink); line-height:1.6;
}}
.dsai-error-fix strong {{
  display:block; font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.07em;
  color:var(--ink-3); margin-bottom:.25rem;
}}

/* ---- empty state, upgraded from a dashed box to an invitation ---- */
.dsai-empty {{
  border:1px dashed var(--border-strong); border-radius:var(--r-lg);
  padding:2.2rem 1.75rem; text-align:left; margin:var(--s-lg) 0;
  background:var(--surface);
}}
.dsai-empty strong {{
  color:var(--ink); display:block; margin-bottom:.4rem; font-size:1.05rem;
  font-weight:600; letter-spacing:-.01em;
}}
.dsai-empty-body {{
  color:var(--ink-2); font-size:var(--t-caption); line-height:1.6; max-width:52ch;
}}

/* ---- column card, for the dataset workspace ---- */
.dsai-colcard {{
  border:1px solid var(--border); border-radius:var(--r-md); background:var(--surface);
  padding:.85rem 1rem; margin-bottom:.5rem;
}}
.dsai-colcard-head {{ display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }}
.dsai-colcard-name {{ font-size:.9375rem; font-weight:600; color:var(--ink); font-family:var(--mono); }}
.dsai-colfacts {{
  display:flex; gap:var(--s-lg); flex-wrap:wrap; margin-top:.55rem;
}}
.dsai-colfact dt {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.07em;
  color:var(--ink-3); font-weight:600; margin:0 0 .1rem;
}}
.dsai-colfact dd {{
  margin:0; font-size:var(--t-caption); color:var(--ink); font-variant-numeric:tabular-nums;
}}

/* ---- trust panel ----
   A score is meaningless without the list it came from, so the two are one
   component and the list cannot be collapsed away from the number. */
.dsai-trust {{
  border:1px solid var(--border); border-radius:var(--r-md); background:var(--surface);
  padding:1.1rem 1.2rem; margin:var(--s-md) 0 var(--s-lg);
}}
.dsai-trust-head {{ display:flex; align-items:baseline; gap:1.1rem; flex-wrap:wrap; }}
.dsai-trust-score {{
  font-size:2.1rem; font-weight:600; letter-spacing:-.03em; line-height:1; color:var(--ink);
  font-variant-numeric:tabular-nums;
}}
.dsai-trust-of {{ font-size:.95rem; color:var(--ink-3); font-weight:500; }}
.dsai-trust-caption {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.08em;
  font-weight:600; color:var(--ink-3);
}}
.dsai-trust-verdict {{
  margin-top:.7rem; font-size:var(--t-body); line-height:1.6; color:var(--ink); max-width:66ch;
}}
.dsai-trust-cols {{ display:flex; gap:var(--s-xl); flex-wrap:wrap; margin-top:1rem; }}
.dsai-trust-col {{ flex:1 1 18rem; min-width:15rem; }}
.dsai-trust-col h4 {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.08em;
  font-weight:700; color:var(--ink-3); margin:0 0 .45rem;
}}
.dsai-trust-item {{
  display:flex; gap:.5rem; align-items:baseline; padding:.2rem 0;
  font-size:var(--t-caption); line-height:1.55; color:var(--ink-2);
}}
.dsai-trust-mark {{ font-family:var(--mono); font-size:.72rem; flex:0 0 .9rem; }}
.dsai-trust-item[data-kind="up"] .dsai-trust-mark {{ color:var(--good); }}
.dsai-trust-item[data-kind="down"] .dsai-trust-mark {{ color:var(--warning); }}

/* ---- analysis status bar ---- */
.dsai-status {{
  display:flex; gap:.35rem; flex-wrap:wrap; align-items:stretch;
  margin:.2rem 0 1.4rem;
}}
.dsai-status-cell {{
  flex:1 1 6rem; min-width:5.5rem; padding:.42rem .6rem;
  border:1px solid var(--border); border-radius:var(--r-sm); background:var(--surface);
}}
.dsai-status-name {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.07em;
  font-weight:700; color:var(--ink-3); display:block;
}}
.dsai-status-value {{
  font-size:var(--t-caption); color:var(--ink-2); display:flex; align-items:baseline; gap:.3rem;
}}
.dsai-status-cell[data-state="done"] {{ border-color:var(--good); }}
.dsai-status-cell[data-state="done"] .dsai-status-value {{ color:var(--ink); font-weight:600; }}
.dsai-status-cell[data-state="warn"] {{ border-color:var(--warning); }}
.dsai-status-cell[data-state="blocked"] {{ border-color:var(--critical); }}
.dsai-status-cell[data-state="todo"] {{ border-style:dashed; }}

/* ---- lineage ---- */
.dsai-lineage {{ margin:.5rem 0 1rem; }}
.dsai-lineage-row {{
  display:grid; grid-template-columns:7.5rem 5rem 1fr; gap:var(--s-md);
  align-items:baseline; padding:.35rem 0; border-bottom:1px solid var(--border);
  font-size:var(--t-caption);
}}
.dsai-lineage-row:last-child {{ border-bottom:none; }}
.dsai-lineage-level {{
  font-size:var(--t-eyebrow); text-transform:uppercase; letter-spacing:.07em;
  font-weight:700; color:var(--ink-3);
}}
.dsai-lineage-id {{ font-family:var(--mono); font-size:.72rem; color:var(--accent-ink); }}
.dsai-lineage-label {{ color:var(--ink); line-height:1.5; }}
.dsai-lineage-kind {{ color:var(--ink-3); }}

/* ---- structural accessibility: always present, not a preference ---- */
.dsai-skip {{
  position:absolute; left:-9999px; top:0; z-index:9999;
  background:var(--accent); color:#fff; padding:.6rem 1rem;
  border-radius:0 0 var(--r-md) 0;
  font-size:var(--t-caption); font-weight:600; text-decoration:none;
}}
.dsai-skip:focus {{ left:0; }}
/* Read by a screen reader, invisible on screen. Not display:none, which would
   take it out of the accessibility tree along with everything else. */
.dsai-sr {{
  position:absolute !important; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0;
}}

/* ---- tooltips on our own elements ---- */
[data-dsai-tip] {{ position:relative; border-bottom:1px dotted var(--border-strong); cursor:help; }}
[data-dsai-tip]:hover::after, [data-dsai-tip]:focus-visible::after {{
  content:attr(data-dsai-tip); position:absolute; bottom:calc(100% + 6px); left:0;
  z-index:50; width:max-content; max-width:26rem; padding:.5rem .65rem;
  background:var(--raised); color:var(--ink); border:1px solid var(--border-strong);
  border-radius:var(--r-md); box-shadow:var(--shadow-overlay);
  font-size:var(--t-caption); font-weight:400; line-height:1.5; white-space:normal;
  text-transform:none; letter-spacing:normal;
}}

/* ---- mode corrections: config.toml sets a static dark base, so the widget
       internals Streamlit paints itself have to be reached here ---- */
{mode_rules}

@media (max-width:1280px) {{
  [data-testid="stAppViewContainer"] .main .block-container {{ padding-left:2rem; padding-right:2rem; }}
  .dsai-stats {{ gap:1.8rem; }}
}}
</style>
"""


_ACCESS_RULES = """
/* ---- accessibility ----
   Focus is never invisible. Streamlit's default outline disappears against
   several of our surfaces, so it is replaced rather than relied on. */
:where(a, button, summary, input, select, textarea, [role="button"],
       [role="tab"], [data-baseweb="select"] > div, [tabindex]):focus-visible {
  outline:3px solid var(--focus) !important;
  outline-offset:2px !important;
  border-radius:4px;
}
"""

_HIGH_CONTRAST = {
    "light": """
:root {
  --ink:#000000; --ink-2:#1f1f1f; --ink-3:#2f2f2f;
  --border:#5a5a5a; --border-strong:#000000;
  --surface:#ffffff; --raised:#ffffff; --sunken:#f0f0f0;
  --accent:#0b4a9c; --accent-ink:#08376f; --accent-soft:#dbe8f8;
  --shadow:none;
}
[data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * { text-shadow:none !important; }
.dsai-card, .dsai-panel, [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"], [data-testid="stAlert"] { border:2px solid var(--border-strong) !important; }
.stButton button, .stDownloadButton button { border:2px solid var(--border-strong) !important; }
""",
    "dark": """
:root {
  --ink:#ffffff; --ink-2:#eeeeee; --ink-3:#d8d8d8;
  --border:#a8a8a8; --border-strong:#ffffff;
  --surface:#000000; --raised:#0d0d0d; --sunken:#141414;
  --accent:#7fb8ff; --accent-ink:#a9d0ff; --accent-soft:#12294a;
  --shadow:none;
}
.dsai-card, .dsai-panel, [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"], [data-testid="stAlert"] { border:2px solid var(--border-strong) !important; }
.stButton button, .stDownloadButton button { border:2px solid var(--border-strong) !important; }
""",
}

_REDUCE_MOTION = """
*, *::before, *::after {
  animation-duration:.001ms !important; animation-iteration-count:1 !important;
  transition-duration:.001ms !important; scroll-behavior:auto !important;
}
"""

# Honoured whether or not the setting is switched on, because the operating
# system has already been asked and answering it twice is not the user's job.
_REDUCE_MOTION_MEDIA = f"""
@media (prefers-reduced-motion: reduce) {{ {_REDUCE_MOTION} }}
"""

_UNDERLINE_LINKS = """
[data-testid="stAppViewContainer"] a:not(.dsai-skip),
[data-testid="stSidebar"] a:not([data-testid="stPageLink"] a) {
  text-decoration:underline !important; text-underline-offset:2px;
}
"""


def _accessibility_css(mode: str, access: Any) -> str:
    """Everything the accessibility settings add on top of the base stylesheet.

    Layered after the base sheet so it overrides it, and written as token
    redefinitions rather than per-component rules wherever possible — one
    change then reaches every component that already uses the token.
    """
    scale = getattr(access, "text_scale", 1.0) or 1.0
    rules = [_ACCESS_RULES, _REDUCE_MOTION_MEDIA]

    # The focus ring must clear 3:1 against both the surface and the component
    # it rings, so it is not the accent colour in either mode.
    rules.append(
        ":root { --focus:%s; }" % ("#ffd400" if mode == "dark" else "#0b4a9c")
    )

    if scale != 1.0:
        rules.append(f"""
:root {{
  --t-title:{_scaled(TYPE_SCALE['title'], scale)};
  --t-section:{_scaled(TYPE_SCALE['section'], scale)};
  --t-body:{_scaled(TYPE_SCALE['body'], scale)};
  --t-caption:{_scaled(TYPE_SCALE['caption'], scale)};
  --t-eyebrow:{_scaled(TYPE_SCALE['eyebrow'], scale)};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
  font-size:var(--t-body);
}}
/* Streamlit hard-codes several sizes; scale them from the same factor rather
   than letting the page fall out of proportion. */
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label, [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"], [data-testid="stMetricValue"],
.stButton button, .stDownloadButton button, [data-baseweb="select"] {{
  font-size:calc(1em * {scale}) !important;
}}
[data-testid="stAppViewContainer"] .main .block-container {{ max-width:{74 / scale:.1f}rem; }}
""")

    if getattr(access, "high_contrast", False):
        rules.append(_HIGH_CONTRAST["dark" if mode == "dark" else "light"])
    if getattr(access, "reduce_motion", False):
        rules.append(_REDUCE_MOTION)
    if getattr(access, "underline_links", False):
        rules.append(_UNDERLINE_LINKS)
    return "<style>" + "\n".join(rules) + "</style>"


def _scaled(size: str, factor: float) -> str:
    """Scale a CSS length like ``0.94rem`` by *factor*, keeping the unit."""
    for unit in ("rem", "em", "px"):
        if size.endswith(unit):
            try:
                return f"{float(size[: -len(unit)]) * factor:.3f}{unit}"
            except ValueError:
                return size
    return size


def apply_theme(mode: str = "dark", access: Any = None) -> None:
    """Inject the stylesheet. Every page calls this once, at the top.

    All custom styling lives here rather than being scattered across pages, so
    there is exactly one place to change how the app looks. The accessibility
    layer goes on last so it wins, and is read off the workspace when it is not
    passed in — the pages should not each have to remember to forward it.
    """
    import streamlit as st

    st.markdown(_stylesheet(mode), unsafe_allow_html=True)

    if access is None:
        workspace = st.session_state.get("dsai_workspace")
        access = getattr(workspace, "access", None)
    if access is not None:
        st.markdown(_accessibility_css(mode, access), unsafe_allow_html=True)


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
    notes = _nav_status(st.session_state.get("dsai_workspace"))
    for heading, pages in sections.items():
        if heading:
            st.markdown(
                f'<div class="dsai-eyebrow" style="margin:.9rem 0 .3rem">{_esc(heading)}</div>',
                unsafe_allow_html=True,
            )
        for page in pages:
            st.page_link(page, label=page.title)
            note = notes.get(page.title)
            if note:
                # The rail doubles as a status display: a page that needs
                # attention says so where the user is already looking, rather
                # than only once they open it.
                st.markdown(
                    f'<div style="font-size:.68rem;color:var(--ink-3);margin:-.35rem 0 .3rem '
                    f'1.05rem;line-height:1.3">{_esc(note)}</div>',
                    unsafe_allow_html=True,
                )
    st.divider()


def _nav_status(state: Any) -> dict[str, str]:
    """A short status line for the pages that have something to say.

    Deliberately sparse. A marker beside every item is wallpaper; a marker
    beside two is a signal.
    """
    if state is None:
        return {}
    notes: dict[str, str] = {}
    profile = getattr(state, "profile", None)
    run = getattr(state, "run", None)

    if profile is not None:
        critical = profile.issues_by_severity("critical")
        if critical:
            notes["1 · Data"] = f"{len(critical)} critical issue(s)"
        elif profile.leakage_suspects:
            notes["1 · Data"] = f"{len(profile.leakage_suspects)} leakage suspect(s)"

    context = getattr(state, "context", None)
    if context is not None and context.is_empty and getattr(state, "has_data", False):
        notes["2 · Context"] = "nothing told to the platform yet"

    if getattr(state, "has_data", False) and getattr(state, "pipeline", None) is None:
        notes["3 · Preprocessing"] = "no pipeline built"

    if getattr(state, "objective", None) is None and getattr(state, "has_data", False):
        notes["4 · Analysis"] = "objective not set"

    if run is None:
        if getattr(state, "has_data", False):
            notes.setdefault("5 · Models", "nothing run yet")
        return notes

    if run.best is None:
        notes["5 · Models"] = "no model completed"
    elif run.self_check is not None and run.self_check.blocking:
        notes["5 · Models"] = f"{len(run.self_check.blocking)} blocking check(s)"
    elif run.trust is not None:
        notes["5 · Models"] = f"evidence strength {run.trust.score}/100"

    if run.findings:
        notes["6 · Insights"] = f"{len(run.findings)} finding(s)"
    if run.recommendations:
        notes["7 · Recommendations"] = f"{len(run.recommendations)} action(s)"
    if run.best is not None:
        notes["8 · Score new data"] = "model ready"
    if len(getattr(state, "runs", [])) >= 2:
        notes["Projects"] = f"{len(state.runs)} runs to compare"
    return notes


def sidebar_chrome(state: Any, working: str = "") -> None:
    """The workspace chrome: the top strip, the navigation rail, and settings.

    Rendered on every page rather than only the landing page, because a
    multipage Streamlit app runs each page in isolation and the user needs the
    same orientation wherever they are. Called once per page, straight after
    ``apply_theme``.
    """
    import streamlit as st

    # The strip goes to the main area; everything after it goes to the rail.
    top_strip(state, working=working)

    with st.sidebar:
        # First focusable thing in the rail, so a keyboard user reaches the page
        # in two stops rather than after the whole navigation.
        st.markdown(
            '<a class="dsai-skip" href="#dsai-main">Skip to the main content</a>',
            unsafe_allow_html=True,
        )
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
            "Appearance", ["dark", "light"],
            index=0 if state.theme == "dark" else 1,
            horizontal=True, key="dsai_theme_toggle",
            help="Applies to the interface and the charts together.",
        )
        if chosen != state.theme:
            state.theme = chosen
            st.rerun()

        _accessibility_controls(state)


_TEXT_SIZES = {"Normal": 1.0, "Large": 1.15, "Larger": 1.3, "Largest": 1.5}


def _accessibility_controls(state: Any) -> None:
    """The accessibility settings, in the sidebar on every page.

    In an expander rather than a separate page: someone who needs larger text
    needs it on the page they are on, not after navigating somewhere with text
    they cannot read.
    """
    import streamlit as st

    access = getattr(state, "access", None)
    if access is None:
        return

    summary = "Accessibility"
    if access.any_enabled:
        summary += " · on"
    with st.expander(summary, expanded=False):
        current = next((label for label, value in _TEXT_SIZES.items()
                        if abs(value - access.text_scale) < 1e-6), "Normal")
        size = st.select_slider(
            "Text size", list(_TEXT_SIZES), value=current, key="dsai_text_size",
            help="Scales the whole interface, not just the body text.",
        )
        contrast = st.checkbox(
            "High contrast", access.high_contrast, key="dsai_contrast",
            help="Maximum contrast ink, heavier borders, no shadows. Charts keep their "
                 "own validated palette.",
        )
        motion = st.checkbox(
            "Reduce motion", access.reduce_motion, key="dsai_motion",
            help="Removes transitions and animation. Your operating system setting is "
                 "already honoured; this is for when it is not set.",
        )
        tables = st.checkbox(
            "Always show chart data", access.always_show_tables, key="dsai_tables",
            help="Every chart's numbers are shown as a table without needing to expand it.",
        )
        underline = st.checkbox(
            "Underline links", access.underline_links, key="dsai_underline",
            help="So a link is not distinguished from body text by colour alone.",
        )

        changed = (
            _TEXT_SIZES[size] != access.text_scale
            or contrast != access.high_contrast
            or motion != access.reduce_motion
            or tables != access.always_show_tables
            or underline != access.underline_links
        )
        if changed:
            access.text_scale = _TEXT_SIZES[size]
            access.high_contrast = contrast
            access.reduce_motion = motion
            access.always_show_tables = tables
            access.underline_links = underline
            st.rerun()

        st.caption(
            "Charts always carry their numbers as a table and a written caption, whether or "
            "not these are on."
        )


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
    # Markdown's two-space line break, and a bare newline, both become one.
    # Without this, separate evidence lines run together into one sentence.
    out = re.sub(r"[ \t]{2,}\n", "<br>", out)
    out = out.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return out


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "", step: str = "") -> None:
    import streamlit as st

    # The landing point for the skip link, and the start of the page's own
    # heading order — every page has exactly one h1, and it is this one.
    block = ['<span id="dsai-main" tabindex="-1"></span>']
    if step:
        block.append(f'<div class="dsai-eyebrow">{_esc(step)}</div>')
    block.append(f"<h1>{_esc(title)}</h1>")
    if subtitle:
        block.append(f'<p class="dsai-lede">{_esc(subtitle)}</p>')
    st.markdown("".join(block), unsafe_allow_html=True)


def chart(
    figure: Any,
    caption: str = "",
    key: str | None = None,
    table: pd.DataFrame | None = None,
    alt: str = "",
    container: Any = None,
) -> None:
    """A chart with the numbers behind it always reachable.

    A Plotly chart is a canvas: a screen reader finds nothing in it, and neither
    does anyone printing in black and white. So every chart in this app ships
    with its own data, extracted from the figure itself rather than assembled by
    hand at each call site — which means no chart can be added without one.

    The table is behind an expander by default and open by default for anyone
    who has asked for that in the accessibility settings.
    """
    import streamlit as st

    target = container if container is not None else st
    target.plotly_chart(figure, width='stretch', key=key)
    if caption:
        target.caption(caption)

    if table is None:
        table = figure_to_frame(figure)
    if table is None or table.empty:
        return

    workspace = st.session_state.get("dsai_workspace")
    always = bool(getattr(getattr(workspace, "access", None), "always_show_tables", False))
    label = alt or "The numbers behind this chart"
    with target.expander(label, expanded=always):
        st.dataframe(table, width='stretch', hide_index=True)


def figure_to_frame(figure: Any, max_rows: int = 500) -> pd.DataFrame | None:
    """The data inside a Plotly figure, as a table.

    Deliberately generic rather than clever: it handles the trace shapes this
    app actually draws (bar, scatter, line, heatmap, box) and returns ``None``
    for anything it does not understand, so a chart it cannot read is simply a
    chart with no table rather than a crash.
    """
    if figure is None:
        return None
    try:
        traces = list(figure.data)
    except Exception:
        return None
    if not traces:
        return None

    blocks: list[pd.DataFrame] = []
    for index, trace in enumerate(traces):
        name = getattr(trace, "name", None) or f"series {index + 1}"
        kind = getattr(trace, "type", "")

        if kind == "heatmap":
            z = getattr(trace, "z", None)
            if z is None:
                continue
            rows = list(getattr(trace, "y", []) or range(len(z)))
            columns = list(getattr(trace, "x", []) or range(len(z[0])))
            blocks.append(pd.DataFrame(z, index=rows, columns=columns).reset_index()
                          .rename(columns={"index": ""}))
            continue

        x = getattr(trace, "x", None)
        y = getattr(trace, "y", None)
        if x is None and y is None:
            continue
        length = max(len(x) if x is not None else 0, len(y) if y is not None else 0)
        if not length:
            continue
        block = pd.DataFrame({
            "Series": [name] * length,
            "X": list(x) if x is not None else [None] * length,
            "Y": list(y) if y is not None else [None] * length,
        })
        text = getattr(trace, "text", None)
        if text is not None and len(text) == length:
            block["Label"] = list(text)
        blocks.append(block)

    if not blocks:
        return None
    frame = pd.concat(blocks, ignore_index=True) if len(blocks) > 1 else blocks[0]

    # A single unnamed series does not need a column saying so.
    if "Series" in frame.columns and frame["Series"].nunique() == 1:
        frame = frame.drop(columns=["Series"])

    # Name the axes after the chart's own axis titles where it has them.
    try:
        layout = figure.layout
        renames = {}
        if getattr(layout.xaxis, "title", None) and layout.xaxis.title.text:
            renames["X"] = layout.xaxis.title.text
        if getattr(layout.yaxis, "title", None) and layout.yaxis.title.text:
            renames["Y"] = layout.yaxis.title.text
        frame = frame.rename(columns=renames)
    except Exception:
        pass
    return frame.head(max_rows)


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

    spoken = {"current": "current step", "done": "completed", "todo": "not started"}
    parts = ['<ol class="dsai-steps" aria-label="Workflow progress">']
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
            f'<span class="dsai-dot" aria-hidden="true">{"✓" if state_name == "done" else ""}</span>'
            f'<span class="dsai-step-label">{_esc(label)}</span>'
            f'<span class="dsai-sr"> — {spoken[state_name]}</span>'
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
    """The glyph is decoration; the word is the content.

    ``aria-hidden`` on the dots stops a screen reader announcing "black circle,
    black circle, white circle" before the word that actually says it.
    """
    mark, word = CONFIDENCE_MARK.get(confidence, ("○○○", "unknown"))
    return f'<span class="dsai-conf" aria-hidden="true">{mark}</span> {word}'


# --------------------------------------------------------------------------
# the workspace chrome
# --------------------------------------------------------------------------

def top_strip(state: Any, working: str = "") -> None:
    """The one strip of chrome above the page: where you are, and what the AI is doing.

    Deliberately a strip and not a bar. Streamlit owns the real browser-chrome
    header and cannot be given a functioning top navigation, so rather than fake
    one badly this carries only what a top bar is actually for: identity,
    context, and the state of the system.
    """
    import streamlit as st

    name = getattr(state, "dataset_name", "") or "no dataset"
    if getattr(state, "has_run", False) and not working:
        status, tone = "analysis complete", "good"
    elif working:
        status, tone = working, "accent"
    elif getattr(state, "has_data", False):
        status, tone = "ready to analyse", "accent"
    else:
        status, tone = "waiting for data", "neutral"

    st.markdown(
        '<div class="dsai-top">'
        '<div class="dsai-top-project">'
        '<span class="dsai-top-brand">DSAI</span>'
        f'<span class="dsai-top-name">{_esc(name)}</span>'
        "</div>"
        '<div class="dsai-top-spacer"></div>'
        f'<span class="dsai-top-state">{AI_MARK} {_esc(status)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def trust_panel(assessment: Any, compact: bool = False) -> None:
    """Why this result can or cannot carry weight, with the workings beside the score.

    The score and the list are one component on purpose. A number on its own
    invites the reader to treat an analyst's summary as a probability; the list
    is what makes it checkable, so it cannot be collapsed away.
    """
    import streamlit as st

    if assessment is None:
        return

    parts = [
        '<div class="dsai-trust">',
        '<div class="dsai-trust-head">',
        f'<span class="dsai-trust-score">{assessment.score}</span>'
        '<span class="dsai-trust-of">/ 100</span>',
        f'<span class="dsai-trust-caption">Evidence strength · {_esc(assessment.band)}</span>',
        f'<span class="dsai-trust-caption" style="margin-left:auto">'
        f'Assumption debt {assessment.assumption_debt} · {_esc(assessment.debt_band)}</span>',
        "</div>",
        f'<div class="dsai-trust-verdict">{_rich(assessment.verdict)}</div>',
    ]

    if not compact:
        columns = ['<div class="dsai-trust-cols">']
        for heading, factors, kind, mark in [
            ("Supporting", assessment.supporting, "up", "+"),
            ("Reducing confidence", assessment.reducing, "down", "!"),
        ]:
            if not factors:
                continue
            rows = "".join(
                f'<div class="dsai-trust-item" data-kind="{kind}"'
                + (f' title="{_esc(f.detail)}"' if f.detail else "")
                + f'><span class="dsai-trust-mark" aria-hidden="true">{mark}</span>'
                f"<span>{_esc(f.statement)}</span></div>"
                for f in factors
            )
            columns.append(f'<div class="dsai-trust-col"><h4>{heading}</h4>{rows}</div>')
        columns.append("</div>")
        parts.append("".join(columns))

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

    st.caption(
        "Evidence strength is a weighted summary of the validation checks that passed and "
        "failed. It is **not** a statistical confidence level and does not mean there is an "
        f"{assessment.score}% chance the conclusion is right."
    )


def known_unknowns(assessment: Any) -> None:
    """What this analysis cannot answer, whatever its numbers look like."""
    import streamlit as st

    if assessment is None or not assessment.unknowns:
        return
    st.markdown(
        '<div class="dsai-inference"><span class="dsai-inference-label">'
        "What this analysis cannot tell you</span>"
        + "".join(f'<div class="dsai-inference-body" style="margin-top:.35rem">— {_rich(u)}</div>'
                  for u in assessment.unknowns)
        + "</div>",
        unsafe_allow_html=True,
    )


#: The analytical stages, in order, and what each one means.
STATUS_STAGES = [
    ("Data", "A dataset is loaded and profiled"),
    ("Question", "An objective has been set"),
    ("Prepare", "A preprocessing pipeline exists"),
    ("Model", "Models have been trained and compared"),
    ("Validate", "The self-checks have run"),
    ("Explain", "The winner has been explained"),
    ("Decide", "Recommendations exist"),
]


def analysis_state(state: Any) -> dict[str, tuple[str, str]]:
    """Where the analysis actually stands, stage by stage.

    Read from the workspace rather than from which page is open: the point of a
    status bar is to say what has happened, not where the user is standing.
    """
    run = getattr(state, "run", None)
    out: dict[str, tuple[str, str]] = {}

    profile = getattr(state, "profile", None)
    if getattr(state, "has_data", False):
        critical = len(profile.issues_by_severity("critical")) if profile else 0
        out["Data"] = ("warn", f"{critical} critical") if critical else ("done", "ready")
    else:
        out["Data"] = ("todo", "none")

    objective = getattr(state, "objective", None)
    out["Question"] = ("done", objective.task_type.value.replace("_", " ")) if objective         else ("todo", "not set")

    pipeline = getattr(state, "pipeline", None)
    out["Prepare"] = ("done", f"{len(pipeline.active_steps)} steps") if pipeline is not None         else ("todo", "none")

    if run is None:
        out["Model"] = ("todo", "not run")
        out["Validate"] = ("todo", "—")
        out["Explain"] = ("todo", "—")
        out["Decide"] = ("todo", "—")
        return out

    out["Model"] = ("done", f"{len(run.results)} trained") if run.best is not None         else ("blocked", "no model")

    check = run.self_check
    if check is None:
        out["Validate"] = ("todo", "not run")
    elif check.blocking:
        out["Validate"] = ("blocked", f"{len(check.blocking)} blocking")
    else:
        failed = [c for c in check.checks if not c.passed]
        out["Validate"] = ("warn", f"{len(failed)} raised") if failed else ("done", "all passed")

    out["Explain"] = ("done", run.explanation.method) if run.explanation is not None         else ("todo", "—")
    out["Decide"] = ("done", f"{len(run.recommendations)} actions") if run.recommendations         else ("todo", "none")
    return out


_STATUS_MARK = {"done": "✓", "warn": "!", "blocked": "×", "todo": "·"}


def status_bar(state: Any) -> None:
    """The analytical state of the project, as seven cells.

    State is carried by a border, a glyph and a word — never by colour alone,
    so it survives a colour-blind reader and a black-and-white print.
    """
    import streamlit as st

    stages = analysis_state(state)
    cells = []
    for name, meaning in STATUS_STAGES:
        status, detail = stages.get(name, ("todo", "—"))
        mark = _STATUS_MARK[status]
        spoken = {"done": "done", "warn": "needs attention", "blocked": "blocked",
                  "todo": "not started"}[status]
        cells.append(
            f'<div class="dsai-status-cell" data-state="{status}" '
            f'title="{_esc(meaning)}">'
            f'<span class="dsai-status-name">{_esc(name)}</span>'
            f'<span class="dsai-status-value"><span aria-hidden="true">{mark}</span>'
            f"{_esc(detail)}</span>"
            f'<span class="dsai-sr"> — {spoken}</span></div>'
        )
    st.markdown(
        f'<div class="dsai-status" role="group" aria-label="Analysis status">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def lineage_view(rows: list[dict[str, Any]]) -> None:
    """A claim traced back through evidence, run, model, pipeline and dataset."""
    import streamlit as st

    if not rows:
        return
    body = "".join(
        f'<div class="dsai-lineage-row">'
        f'<span class="dsai-lineage-level">{_esc(row["level"])}</span>'
        f'<span class="dsai-lineage-id">{_esc(row.get("id", "") or "—")}</span>'
        f'<span class="dsai-lineage-label">{_esc(row.get("label", ""))}'
        + (f' <span class="dsai-lineage-kind">({_esc(row["kind"])})</span>'
           if row.get("kind") else "")
        + "</span></div>"
        for row in rows
    )
    st.markdown(f'<div class="dsai-lineage">{body}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "neutral") -> str:
    """A small state marker. Returns HTML — never sets state by colour alone."""
    return f'<span class="dsai-badge" data-tone="{_esc(tone)}">{_esc(text)}</span>'


def ai_panel(
    body: str,
    why: str = "",
    heading: str = "AI Analyst",
    confidence: Any = None,
    busy: bool = False,
    evidence: list[str] | None = None,
) -> None:
    """The AI speaking in its own voice, in its own surface.

    One panel, not a chat transcript: the platform is a colleague reporting a
    finding, not a bot waiting for the next message. Everything it says arrives
    with its reasoning and its confidence attached, because a claim without
    those is not something a reader can act on.
    """
    import streamlit as st

    parts = [
        f'<div class="dsai-ai" data-busy="{"yes" if busy else "no"}">',
        f'<div class="dsai-ai-head"><span class="dsai-ai-mark">{AI_MARK}</span>'
        f"<span>{_esc(heading)}</span>",
    ]
    if confidence is not None:
        parts.append(
            f'<span style="margin-left:auto;text-transform:none;letter-spacing:0;'
            f'font-weight:500;color:var(--ink-3)">confidence {_confidence_html(confidence)}</span>'
        )
    parts.append("</div>")
    parts.append(f'<div class="dsai-ai-body">{_rich(body)}</div>')

    if evidence:
        parts.append('<hr class="dsai-ai-rule">')
        items = "".join(f"<li>{_rich(item)}</li>" for item in evidence[:4])
        parts.append(
            '<div class="dsai-ai-why"><strong>Evidence</strong>'
            f'<ul style="margin:.3rem 0 0;padding-left:1.1rem">{items}</ul></div>'
        )
    if why:
        parts.append('<hr class="dsai-ai-rule">')
        parts.append(f'<div class="dsai-ai-why"><strong>Why</strong><br>{_rich(why)}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


#: The stages an analysis actually moves through, in order. Named here so the
#: progress display and the orchestrator's own trace agree about the work.
ANALYSIS_STAGES = [
    ("Understanding your dataset", "profile"),
    ("Checking data quality", "quality"),
    ("Working out what it can answer", "objective"),
    ("Designing preprocessing", "preprocess"),
    ("Choosing candidate models", "select"),
    ("Running experiments", "experiment"),
    ("Comparing results", "compare"),
    ("Explaining the winner", "explain"),
    ("Checking its own conclusions", "selfcheck"),
    ("Writing up findings", "insight"),
]

_STAGE_MARK = {"done": "✓", "running": "◆", "todo": "·", "failed": "✗",
               "warning": "!", "skipped": "–"}


def stage_progress(stages: list[tuple[str, str, str]]) -> str:
    """Named stages with their state. Returns HTML so it can be re-rendered in place.

    Each entry is ``(label, state, detail)``. A spinner says only "something is
    happening"; this says what, which is the difference between a system that
    feels slow and one that feels busy.
    """
    rows = []
    for label, state, detail in stages:
        mark = _STAGE_MARK.get(state, "·")
        extra = f' <span class="dsai-stage-detail">{_esc(detail)}</span>' if detail else ""
        rows.append(
            f'<li class="dsai-stage" data-state="{_esc(state)}">'
            f'<span class="dsai-stage-mark" aria-hidden="true">{mark}</span>'
            f"<span>{_esc(label)}{extra}</span></li>"
        )
    return f'<ul class="dsai-stages" aria-label="Analysis progress">{"".join(rows)}</ul>'


def live_stages(events: list[Any], limit: int = 14) -> str:
    """Trace events as named stages, one line per step, latest state per step.

    The steps are the ones the engine actually announced, not a script written
    ahead of time — so the display cannot claim work that did not happen. A
    warning stays visible rather than being replaced by the next step: a caveat
    that scrolls past is a caveat nobody read.
    """
    seen: dict[int, Any] = {}
    order: list[int] = []
    for event in events:
        key = id(event)
        if key not in seen:
            order.append(key)
        seen[key] = event
    live = [seen[k] for k in order]

    # Keep every warning and failure; trim only the routine successes.
    notable = [e for e in live if e.status in ("warning", "failed")]
    recent = live[-limit:]
    keep = [e for e in live if e in notable or e in recent]
    return stage_progress([(e.step, e.status, e.detail or "") for e in keep])


def skeleton(lines: int = 3, widths: list[int] | None = None) -> str:
    """A placeholder with the shape of the thing that is coming.

    Returned rather than written, so a caller can put it in a placeholder and
    swap it for real content without the page jumping.
    """
    widths = widths or [100, 82, 64][:lines] + [70] * max(0, lines - 3)
    bars = "".join(f'<span style="width:{w}%"></span>' for w in widths[:lines])
    return f'<div class="dsai-skeleton" aria-hidden="true">{bars}</div>'


def quality_bars(rows: list[tuple[str, float, str]]) -> None:
    """Score bars: label, 0-100 value, and a note.

    The bar is the second encoding, not the first — the number is written out
    beside it, so the reading does not depend on judging a length.
    """
    import streamlit as st

    out = ['<div class="dsai-bars">']
    for label, value, note in rows:
        value = max(0.0, min(100.0, float(value)))
        tone = "good" if value >= 90 else "warning" if value >= 70 else "critical"
        title = f"{label}: {value:.0f} out of 100" + (f". {note}" if note else "")
        out.append(
            f'<div class="dsai-bar-row" data-tone="{tone}" role="img" aria-label="{_esc(title)}">'
            f'<div class="dsai-bar-label">{_esc(label)}</div>'
            f'<div class="dsai-bar-track"><div class="dsai-bar-fill" style="width:{value:.0f}%"></div></div>'
            f'<div class="dsai-bar-value">{value:.0f}</div>'
            "</div>"
        )
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def hero(value: Any, unit: str = "", label: str = "", note: str = "") -> None:
    """One number, large. For the figure a page exists to deliver."""
    import streamlit as st

    st.markdown(
        (f'<div class="dsai-stat-label">{_esc(label)}</div>' if label else "")
        + '<div class="dsai-hero">'
        f'<span class="dsai-hero-value">{_esc(value)}</span>'
        + (f'<span class="dsai-hero-unit">{_esc(unit)}</span>' if unit else "")
        + "</div>"
        + (f'<div class="dsai-stat-note">{_esc(note)}</div>' if note else ""),
        unsafe_allow_html=True,
    )


def pipeline_graph(nodes: list[dict[str, Any]]) -> None:
    """The pipeline drawn as the chain it is.

    Each node is ``{"name", "scope", "detail", "kind"}``. Drag-and-drop needs a
    custom front-end component Streamlit does not provide, so ordering is
    changed with the controls beside the graph rather than by dragging — the
    graph is the picture, not the editor.
    """
    import streamlit as st

    out = ['<div class="dsai-pipe">']
    for index, node in enumerate(nodes):
        kind = node.get("kind", "step")
        number = "" if kind in ("source", "model") else f"{index:02d}"
        scope = node.get("scope", "")
        out.append(
            f'<div class="dsai-pipe-node" data-kind="{_esc(kind)}">'
            '<div class="dsai-pipe-row">'
            f'<span class="dsai-pipe-index">{_esc(number)}</span>'
            f'<span class="dsai-pipe-name">{_esc(node.get("name", ""))}</span>'
            + (f'<span class="dsai-pipe-scope">{_esc(scope)}</span>' if scope else "")
            + "</div>"
            + (f'<div class="dsai-pipe-detail">{_esc(node["detail"])}</div>'
               if node.get("detail") else "")
            + "</div>"
        )
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def rank_item(
    index: int,
    title: str,
    body: str = "",
    metrics: list[tuple[str, Any]] | None = None,
    badges: list[tuple[str, str]] | None = None,
    lead: bool = False,
) -> None:
    """One entry in a ranked list — a model, an insight, an action.

    The rank is the only ornament. Everything else is type: the title, the
    reasoning, and the figures that justify the position.
    """
    import streamlit as st

    marks = "".join(badge(text, tone) for text, tone in (badges or []))
    figures = ""
    if metrics:
        cells = "".join(
            f'<div class="dsai-rank-metric"><dt>{_esc(name)}</dt><dd>{_esc(value)}</dd></div>'
            for name, value in metrics
        )
        figures = f'<dl class="dsai-rank-metrics">{cells}</dl>'
    st.markdown(
        f'<div class="dsai-rank" data-lead="{"yes" if lead else "no"}">'
        f'<div class="dsai-rank-index">{index:02d}</div>'
        "<div>"
        f'<div class="dsai-rank-head"><span class="dsai-rank-title">{_esc(title)}</span>{marks}</div>'
        + figures
        + (f'<div class="dsai-rank-body">{_rich(body)}</div>' if body else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def leaderboard(rows: list[dict[str, Any]], columns: list[str] | None = None,
                lead_index: int = 0, numeric: set[str] | None = None) -> None:
    """A ranked table with the leading row marked in two ways, not one.

    Columns are whatever the caller passes, because the metrics that matter
    change with the problem: R² and RMSE for a regression, precision and recall
    for a classifier.
    """
    import streamlit as st

    if not rows:
        return
    columns = columns or list(rows[0])
    numeric = numeric or set()
    head = "".join(
        f'<th class="{"dsai-num" if c in numeric else ""}">{_esc(c)}</th>' for c in columns
    )
    body = []
    for position, row in enumerate(rows):
        lead = position == lead_index
        cells = "".join(
            f'<td class="{"dsai-num" if c in numeric else ""}">{_esc(row.get(c, "—"))}</td>'
            for c in columns
        )
        body.append(f'<tr data-lead="{"yes" if lead else "no"}">{cells}</tr>')
    st.markdown(
        '<div style="overflow-x:auto">'
        f'<table class="dsai-table dsai-board"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def error_state(title: str, body: str, fix: str = "") -> None:
    """An error that says what happened and what to do about it.

    "Model failed" tells the reader nothing they can act on. Every error surface
    in this app names the cause and proposes the next move.
    """
    import streamlit as st

    st.markdown(
        '<div class="dsai-error">'
        f'<div class="dsai-error-title"><span aria-hidden="true">▲</span>{_esc(title)}</div>'
        f'<div class="dsai-error-body">{_rich(body)}</div>'
        + (f'<div class="dsai-error-fix"><strong>Recommended fix</strong>{_rich(fix)}</div>'
           if fix else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def column_card(profile_column: Any, currency: str = "") -> None:
    """One variable, with the facts a reader needs before trusting it."""
    import streamlit as st

    column = profile_column
    facts: list[tuple[str, str]] = [
        ("Type", column.semantic_type.value.replace("_", " ")),
        ("Missing", f"{column.missing_pct:.1f}%"),
        ("Distinct", f"{column.n_unique:,}"),
    ]
    if column.mean is not None:
        from dsai.engines.metrics import human_number

        facts += [("Mean", human_number(column.mean)), ("Median", human_number(column.median)),
                  ("Range", f"{human_number(column.minimum)} – {human_number(column.maximum)}")]
    cells = "".join(f"<div class='dsai-colfact'><dt>{_esc(n)}</dt><dd>{_esc(v)}</dd></div>"
                    for n, v in facts)
    st.markdown(
        '<div class="dsai-colcard">'
        f'<div class="dsai-colcard-head"><span class="dsai-colcard-name">{_esc(column.name)}</span>'
        f'{badge(column.semantic_type.value.replace("_", " "))}</div>'
        f'<dl class="dsai-colfacts">{cells}</dl></div>',
        unsafe_allow_html=True,
    )


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
    st.markdown(
        f'<div class="dsai-stats" role="group" aria-label="Key figures">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


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

    st.dataframe(frame, width='stretch', hide_index=True, **kwargs)


def empty_state(title: str, detail: str = "", actions: list[tuple[str, str]] | None = None) -> None:
    """An empty state that offers the next move rather than reporting an absence.

    ``actions`` are ``(label, page_path)`` pairs rendered as real links, so the
    way out of the empty state is one click and not a hunt through the sidebar.
    """
    import streamlit as st

    st.markdown(
        f'<div class="dsai-empty"><strong>{_esc(title)}</strong>'
        f'<div class="dsai-empty-body">{_rich(detail)}</div></div>',
        unsafe_allow_html=True,
    )
    if actions:
        columns = st.columns(len(actions) + 2)
        for column, (label, page) in zip(columns, actions):
            label = label if label.endswith("→") else f"{label} →"
            try:
                column.page_link(page, label=label)
            except Exception:
                column.markdown(f"**{label}**")


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
        # The severity word is in the label as well as the glyph, so the
        # distinction survives a reader who cannot see the shape or the colour.
        mark = SEVERITY_MARK.get(issue.severity, "■")
        with st.expander(f"{mark}  {issue.severity.capitalize()} — {issue.message}",
                         expanded=issue.severity == "critical"):
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
        empty_state(
            "Start your first analysis",
            "Load a dataset and the platform will profile it, work out what question it can "
            "answer, and propose an approach — before you have to decide anything.",
            actions=[("Load a dataset", "pages/1_Data.py")],
        )
        return False
    return True


def require_run(state: Any) -> bool:
    if not state.has_run:
        empty_state(
            "Nothing has been analysed yet",
            "This page shows results. Choose what you want to find out on the Analysis page and "
            "run it — the plan is shown in full before anything executes.",
            actions=[("Set up the analysis", "pages/4_Analysis.py")],
        )
        return False
    return True


def show_notices(state: Any) -> None:
    """Transient feedback about what just happened. Genuinely an alert."""
    import streamlit as st

    for level, message in state.take_notices():
        {"success": st.success, "warning": st.warning, "error": st.error}.get(level, st.info)(message)
