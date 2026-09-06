"""Model library: the full registry, browsable, with the metadata that drives selection."""

from __future__ import annotations

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

    st.caption(f"{len(specs)} algorithm(s)")
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
