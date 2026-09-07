"""Model library: the full registry, browsable, with the metadata that drives selection."""

from __future__ import annotations

from itertools import zip_longest

import pandas as pd
import streamlit as st

from dsai.app.components import apply_theme, caveat, sidebar_chrome, dataframe, page_header
from dsai.app.state import workspace
from dsai.core.schema import TaskType
from dsai.registry.base import REGISTRY, load_builtin_models, load_plugins
from dsai.preprocessing.steps import STEPS

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
load_builtin_models()

page_header(
    "Model library",
    "Nothing in this platform hard-codes a list of models. Algorithms are described by metadata, "
    "and the decision engine queries this registry for the ones that suit the data at hand.",
)

stats = REGISTRY.stats()
columns = st.columns(4)
columns[0].metric("Algorithms registered", stats["total"])
columns[1].metric("Available here", stats["available"],
                  help="The rest need an optional package installed.")
columns[2].metric("Categories", len(stats["by_category"]))
columns[3].metric("Preprocessing steps", len(STEPS))

models_tab, steps_tab, plugins_tab = st.tabs(["Algorithms", "Preprocessing steps", "Extending it"])

with models_tab:
    query = st.text_input(
        "Search", "", placeholder=f"Search {stats['total']} algorithms — name, family, or what it is good for…",
        label_visibility="collapsed", key="_library_search",
    )
    columns = st.columns(4)
    category = columns[0].selectbox("Category", ["all"] + REGISTRY.categories())
    family = columns[1].selectbox("Family", ["all"] + REGISTRY.families())
    task = columns[2].selectbox("Task", ["all"] + [t.value for t in TaskType])
    only_available = columns[3].checkbox("Only what is installed", value=True)

    specs = REGISTRY.all()
    if category != "all":
        specs = [s for s in specs if s.category == category]
    if family != "all":
        specs = [s for s in specs if s.family == family]
    if task != "all":
        specs = [s for s in specs if any(t.value == task for t in s.task_types)]
    if only_available:
        specs = [s for s in specs if s.is_available()]
    if query.strip():
        needle = query.strip().lower()
        specs = [
            s for s in specs
            if needle in s.name.lower() or needle in s.key.lower() or needle in s.family.lower()
            or needle in s.category.lower()
            or any(needle in item.lower() for item in (s.good_for or []))
            or any(needle in item.lower() for item in (s.advantages or []))
        ]
    elif category == "all" and family == "all" and task == "all":
        # Untouched filters: registration order would otherwise put every
        # regression algorithm first (they are registered first in code) and
        # the "first 24 as cards" preview would look like a regression-only
        # library. Round-robin across categories so the first screen a new
        # user sees is representative of the whole registry.
        by_category: dict[str, list] = {}
        order: list[str] = []
        for spec in specs:
            if spec.category not in by_category:
                by_category[spec.category] = []
                order.append(spec.category)
            by_category[spec.category].append(spec)
        specs = [spec for row in zip_longest(*(by_category[c] for c in order))
                 for spec in row if spec is not None]

    st.caption(
        f"{len(specs)} algorithm(s) match. Showing the first 24 as cards — narrow the search or "
        "use the table below for the full list."
    )

    # Cards, not a wall of rows: the spec's point is that a library nobody can
    # scan is not a library. Three across, capped, with the table underneath for
    # anyone who wants everything at once.
    _INTERPRETABILITY_BARS = {"transparent": 5, "high": 4, "moderate": 3, "low": 2, "opaque": 1}
    grid = st.columns(3, gap="medium")
    for position, spec in enumerate(specs[:24]):
        with grid[position % 3], st.container(border=True):
            tasks = " · ".join(t.value.replace("_", " ") for t in spec.task_types[:2])
            filled = _INTERPRETABILITY_BARS.get(spec.interpretability.value, 3)
            st.markdown(
                f'<div class="dsai-card-title">{spec.name}</div>'
                f'<div class="dsai-meta">{tasks}</div>'
                f'<div style="font-size:.78rem;color:var(--ink-2);margin-bottom:.35rem">'
                f'<span style="font-family:var(--mono);letter-spacing:.1em" aria-hidden="true">'
                f'{"|" * filled}{"·" * (5 - filled)}</span>'
                f'&nbsp;{spec.interpretability.value} interpretability · {spec.cost.value.replace("_", " ")} cost'
                "</div>"
                + (f'<div style="font-size:.8125rem;color:var(--ink-2);line-height:1.55">'
                   f'<strong style="color:var(--ink-3);font-size:.68rem;text-transform:uppercase;'
                   f'letter-spacing:.07em">Best for</strong><br>{spec.good_for[0]}</div>'
                   if spec.good_for else "")
                + ("" if spec.is_available() else
                   '<div style="font-size:.75rem;color:var(--warning);margin-top:.4rem">'
                   "needs an optional package</div>"),
                unsafe_allow_html=True,
            )

    with st.expander(f"All {len(specs)} as a table"):
        dataframe(pd.DataFrame([
            {
                "Name": s.name, "Key": s.key, "Category": s.category, "Family": s.family,
                "Interpretability": s.interpretability.value, "Cost": s.cost.value,
                "Nonlinear": s.nonlinear, "Handles missing": s.handles_missing,
                "Needs scaling": s.requires_scaling, "Available": s.is_available(),
            }
            for s in specs
        ]))

    st.divider()
    if specs:
        chosen = st.selectbox("Inspect an algorithm", [s.key for s in specs],
                              format_func=lambda k: REGISTRY.get(k).name)
        spec = REGISTRY.get(chosen)
        st.subheader(spec.name)
        columns = st.columns(4)
        columns[0].metric("Interpretability", spec.interpretability.value)
        columns[1].metric("Training cost", spec.cost.value)
        columns[2].metric("Minimum rows", spec.min_rows)
        columns[3].metric("Available", "yes" if spec.is_available() else "no")

        left, right = st.columns(2)
        with left:
            st.markdown("**Advantages**")
            for item in spec.advantages or ["—"]:
                st.markdown(f"- {item}")
            st.markdown("**Good for**")
            for item in spec.good_for or ["—"]:
                st.markdown(f"- {item}")
        with right:
            st.markdown("**Limitations**")
            for item in spec.limitations or ["—"]:
                st.markdown(f"- {item}")
            st.markdown("**Assumptions**")
            for item in spec.assumptions or ["—"]:
                st.markdown(f"- {item}")

        if spec.hyperparameters:
            st.markdown("**Hyper-parameters**")
            dataframe(pd.DataFrame([
                {
                    "Parameter": h.name, "Type": h.kind, "Default": str(h.default),
                    "Range": f"{h.low} – {h.high}" if h.low is not None else ", ".join(map(str, h.choices)),
                    "Tuned automatically": h.tune, "What it does": h.description,
                }
                for h in spec.hyperparameters
            ]))
        if spec.metrics:
            st.caption("Evaluated on: " + ", ".join(spec.metrics))
        if spec.docs_url:
            st.markdown(f"[Documentation]({spec.docs_url})")

with steps_tab:
    category = st.selectbox("Category", ["all"] + STEPS.categories(), key="step_category")
    steps = STEPS.all() if category == "all" else STEPS.by_category(category)
    dataframe(pd.DataFrame([
        {
            "Name": s.name, "Key": s.key, "Category": s.category, "Scope": s.scope,
            "Leakage-safe": s.leakage_safe, "Needs target": s.needs_target,
            "What it does": s.description,
        }
        for s in steps
    ]))
    st.caption(
        "'Leakage-safe' means the step learns nothing from the data and can be applied before "
        "splitting. Everything else is fitted inside each cross-validation fold."
    )

with plugins_tab:
    st.markdown(
        """
Adding an algorithm does not require touching the platform. Register a `ModelSpec`:

```python
from dsai.registry.base import register, ModelSpec, Cost, Interpretability
from dsai.core.schema import TaskType
from dsai.registry._helpers import lazy, hp

register(ModelSpec(
    key="my_regressor",
    name="My Regressor",
    category="regression",
    family="ensemble",
    task_types=[TaskType.REGRESSION],
    builder=lazy("my_package.models:MyRegressor"),
    module="my_package",
    requires_scaling=False,
    supports_feature_importance=True,
    interpretability=Interpretability.MODERATE,
    cost=Cost.MEDIUM,
    nonlinear=True,
    assumptions=["..."],
    advantages=["..."],
    limitations=["..."],
    hyperparameters=[hp("depth", "int", 6, 2, 20, description="Tree depth.")],
    metrics=["r2", "rmse", "mae"],
))
```

The decision engine will consider it from that point on, filtered by the same metadata as
everything else. A separate package can register a whole pack through an entry point:

```toml
[project.entry-points."dsai.models"]
my_pack = "my_package.models:register_all"
```
        """
    )
    if st.button("Reload plugins"):
        loaded = load_plugins()
        st.success(f"Loaded {len(loaded)} plugin pack(s): {', '.join(loaded) or 'none found'}")
