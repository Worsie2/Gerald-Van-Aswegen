"""The workspace entry point.

Run with::

    streamlit run dsai/app/main.py

or::

    dsai app

Navigation is declared here rather than inferred from filenames. Twelve pages in
a flat list is not an information architecture, and Streamlit hides everything
past the tenth behind a "view more" control — so the pages are grouped by what
you are trying to do, and all of them stay visible.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DSAI — AI data science workspace",
    page_icon="◔",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = Path(__file__).parent / "pages"
VIEWS = Path(__file__).parent / "views"


def _page(path: Path, title: str, *, default: bool = False) -> st.Page:
    return st.Page(str(path), title=title, default=default)


SECTIONS = {
        "": [_page(VIEWS / "0_Overview.py", "Overview", default=True)],
        "Workflow": [
            _page(PAGES / "1_Data.py", "1 · Data"),
            _page(PAGES / "2_Context.py", "2 · Context"),
            _page(PAGES / "3_Preprocessing.py", "3 · Preprocessing"),
            _page(PAGES / "4_Analysis.py", "4 · Analysis"),
            _page(PAGES / "5_Models.py", "5 · Models"),
            _page(PAGES / "6_Insights.py", "6 · Insights"),
            _page(PAGES / "7_Recommendations.py", "7 · Recommendations"),
        ],
        "Investigate": [
            _page(PAGES / "8_Ask.py", "Ask a question"),
            _page(PAGES / "12_Statistics.py", "Statistics"),
        ],
        "Output": [
            _page(PAGES / "9_Report.py", "Report"),
            _page(PAGES / "10_Projects.py", "Projects"),
        ],
        "Reference": [
            _page(PAGES / "11_Model_Library.py", "Model library"),
        ],
}

# The sidebar renders these itself, so every page stays visible and the groups
# are shown rather than implied.
st.session_state["_dsai_sections"] = SECTIONS

navigation = st.navigation(SECTIONS)
navigation.run()
