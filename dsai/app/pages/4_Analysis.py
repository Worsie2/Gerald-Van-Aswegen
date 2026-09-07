"""Analysis page: choose the objective, review the plan, run it."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, badge, caveat, chart, decision_panel, error_state, inference,
    leaderboard, live_stages, metric_row, override_notice, page_header, rank_item, require_data,
    show_notices, sidebar_chrome, simple_table, skeleton, status_bar, trace_view, workflow_nav,
)
from dsai.engines.brief import build_brief, build_contract
from dsai.viz.theme import AI_MARK
from dsai.app.state import scientist, workspace
from dsai.core.schema import Objective, TaskType
from dsai.engines.decision import Constraints, plan_analysis
from dsai.registry.base import REGISTRY
from dsai.engines.orchestrator import RunSettings



def _failure_advice(exc: Exception, objective) -> str:
    """Turn an exception into the next thing to try.

    "Model failed" tells a reader nothing they can act on. These are the
    failures this platform actually produces, each paired with the move that
    resolves it.
    """
    message = str(exc).lower()
    if "could not convert" in message or "invalid literal" in message or "dtype" in message:
        return (
            "A column holding text reached a model that only accepts numbers. Build a pipeline on "
            "the **Preprocessing** page — the recommended one encodes categorical columns — or "
            "check the Data page for a variable typed as text that should be numeric."
        )
    if "unencoded" in message or "categorical" in message:
        return (
            "Categorical columns need encoding before most models will accept them. The "
            "**Preprocessing** page builds a pipeline that does this correctly, fitted inside each "
            "cross-validation fold so nothing leaks."
        )
    if "nan" in message or "missing" in message or "infinity" in message:
        return (
            "Missing or infinite values reached a model that cannot take them. Add an imputation "
            "step on the **Preprocessing** page, or drop the affected rows there."
        )
    if "n_splits" in message or "too few" in message or "n_samples" in message:
        return (
            "There are not enough rows for the validation strategy chosen. Lower the hold-out share "
            "in Settings, or pick a simpler objective — a class with only a handful of rows cannot "
            "be cross-validated."
        )
    if "memory" in message:
        return (
            "The run ran out of memory. Reduce **Models to compare**, set the time budget to "
            "*fast*, or reduce the number of variables on the Preprocessing page."
        )
    if objective is not None and objective.task_type.is_supervised and not objective.target:
        return (
            f"{objective.task_type.value.replace('_', ' ').capitalize()} needs a target variable "
            "and none is set. Choose one above."
        )
    return (
        "Open **Why these choices** above to see the plan that was about to run, and the "
        "**Data** page for anything flagged in the dataset. If the message names a column, that "
        "column is where to look."
    )


state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Analysis",
    "Choose what to find out, review exactly what will run, then run it.",
    "Step 4 of 7",
)
workflow_nav("Analysis", state)

if not require_data(state):
    st.stop()

profile = state.profile
frame = state.typed_frame if state.typed_frame is not None else state.frame

# --------------------------------------------------------------------------
# mode
# --------------------------------------------------------------------------
mode = st.radio(
    "How do you want to work?",
    ["AI automatic", "AI assisted", "Manual / expert"],
    horizontal=True,
    captions=[
        "Do the analysis for me, end to end.",
        "Recommend an approach, let me approve or change it, then run.",
        "I choose the objective, models and settings myself.",
    ],
    help="All three use the same underlying engine — only how much you decide changes.",
)
mode_key = {"AI automatic": "automatic", "AI assisted": "assisted", "Manual / expert": "manual"}[mode]
state.mode = "advanced" if mode_key == "manual" else "guided"

st.divider()

# --------------------------------------------------------------------------
# what the platform makes of this dataset, before anything is asked of the user
# --------------------------------------------------------------------------
_objectives = state.objectives or []
if _objectives:
    _lead = _objectives[0]
    _issues = len(profile.quality_issues) if profile else 0
    _available = len(REGISTRY.find(task_type=_lead.task_type, only_available=True))
    ai_panel(
        f"I have profiled **{profile.n_rows:,} rows** across **{profile.n_columns} variables** "
        f"and found **{len(_objectives)} question{'s' if len(_objectives) != 1 else ''}** this data "
        f"can actually answer. **{_available} algorithms** in the registry suit the strongest one.",
        why=_lead.rationale,
        evidence=(
            [f"Data quality scores {profile.quality_score}/100"
             + (f", with {_issues} issue(s) flagged" if _issues else ", with nothing flagged")]
            + [f"Strongest candidate: {o.label().replace('_', ' ')}" for o in _objectives[:1]]
            + ([f"{len(profile.leakage_suspects)} column(s) look like they leak the answer"]
               if getattr(profile, "leakage_suspects", None) else [])
        ),
        confidence=_lead.confidence,
    )

# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
st.subheader("What do you want to find out?")
override_notice(
    "These are ranked suggestions. Pick any of them, or define your own below — the platform "
    "recommends, you decide."
)

objectives = state.objectives or []
labels = [
    f"{o.label()}  ·  {'you asked for this' if o.source.startswith('user') else 'detected from the data'}"
    for o in objectives
]
# Whatever was chosen on the Preprocessing page is pre-selected here, so the two
# pages cannot silently disagree about what is being analysed.
all_choices = labels + ["Define my own"]
default_index = 0
if state.objective is not None:
    for position, candidate in enumerate(objectives):
        if (candidate.task_type, candidate.target) == (state.objective.task_type, state.objective.target):
            default_index = position
            break
    else:
        default_index = len(all_choices) - 1

choice = st.radio("Suggested objectives", all_choices, index=default_index)

if choice == "Define my own":
    columns = st.columns(3)
    task = columns[0].selectbox(
        "Analysis type", list(TaskType),
        format_func=lambda t: t.value.replace("_", " "),
        index=list(TaskType).index(state.objective.task_type) if state.objective is not None else 0,
    )
    needs_target = task.is_supervised
    target = columns[1].selectbox(
        "Target variable", ["— none —"] + list(profile.columns),
        disabled=not needs_target,
    )
    time_column = columns[2].selectbox(
        "Time column", ["— none —"] + profile.datetime_columns,
        disabled=task is not TaskType.TIME_SERIES_FORECAST,
    )
    feature_columns = st.multiselect(
        "Variables to use (leave empty to use everything appropriate)",
        [c for c in profile.columns],
    )
    extra = st.columns(2)
    n_clusters = extra[0].number_input("Number of segments (0 = let the platform choose)", 0, 30, 0,
                                       disabled=task is not TaskType.CLUSTERING)
    horizon = extra[1].number_input("Forecast horizon (0 = default)", 0, 120, 0,
                                    disabled=task is not TaskType.TIME_SERIES_FORECAST)

    extras: dict = {}
    if task is TaskType.ASSOCIATION_RULES:
        st.markdown("**Basket layout**")
        st.caption(
            "Association mining needs one row per item per transaction. Name the two columns that "
            "identify them, or leave both blank and the platform will try to work the layout out."
        )
        basket = st.columns(4)
        transaction_column = basket[0].selectbox(
            "Transaction ID column", ["— detect —"] + list(profile.columns))
        item_column = basket[1].selectbox(
            "Item column", ["— detect —"] + list(profile.columns))
        min_support = basket[2].slider(
            "Minimum support", 0.005, 0.5, 0.05, 0.005,
            help="Share of transactions an itemset must appear in. Lower finds more rules, "
                 "most of them noise.")
        min_confidence = basket[3].slider(
            "Minimum confidence", 0.05, 0.95, 0.30, 0.05,
            help="How often the rule must hold when its condition is met.")
        extras = {
            "transaction_column": None if transaction_column.startswith("—") else transaction_column,
            "item_column": None if item_column.startswith("—") else item_column,
            "min_support": min_support,
            "min_confidence": min_confidence,
        }

    if task is TaskType.CLUSTERING and not n_clusters:
        st.caption(
            "With no number set, the platform sweeps a range and compares the elbow, silhouette, "
            "Calinski-Harabasz and Davies-Bouldin measures. Where they agree that is a real signal; "
            "where they disagree the data has no sharp cluster structure and the number is yours to choose."
        )

    objective = Objective(
        task_type=task,
        target=None if target.startswith("—") else target,
        features=feature_columns,
        time_column=None if time_column.startswith("—") else time_column,
        n_clusters=int(n_clusters) or None,
        horizon=int(horizon) or None,
        rationale="You defined this objective directly.",
        source="user_override",
        priority=1.0,
        extras=extras,
    )
else:
    objective = objectives[labels.index(choice)]
    inference(objective.rationale, label="Why this objective")

state.objective = objective

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
st.subheader("Settings")
columns = st.columns(4)
max_models = columns[0].slider("Models to compare", 2, 20, state.settings.max_models)
time_budget = columns[1].selectbox("Time budget", ["fast", "balanced", "thorough"],
                                   index=["fast", "balanced", "thorough"].index(state.settings.time_budget))
interpretability = columns[2].selectbox(
    "How important is interpretability?", ["low", "moderate", "high", "critical"],
    index=["low", "moderate", "high", "critical"].index(state.settings.interpretability_need),
    help="'Critical' restricts the field to models whose mechanism can be read directly.",
)
test_size = columns[3].slider("Hold-out share", 0.1, 0.4, state.settings.test_size, 0.05)

# A pipeline built by hand on the Preprocessing page used to be ignored unless the
# mode happened to be Manual, which made the work look lost. It is now an explicit
# choice, shown whenever a pipeline exists.
use_my_pipeline = False
if state.pipeline is not None:
    use_my_pipeline = st.checkbox(
        f"Use my '{state.active_pipeline}' pipeline instead of building one for each model",
        value=True,
        help="One pipeline is applied to every model in the comparison. Leaving this off lets the "
             "platform tailor preprocessing to each model — a tree keeps its outliers, a KNN does not.",
    )

advanced = st.expander("Advanced settings", expanded=state.mode == "advanced")
with advanced:
    columns = st.columns(3)
    random_state = columns[0].number_input("Random seed", 0, 10_000, state.settings.random_state,
                                           help="Recorded in the manifest so the run can be reproduced exactly.")
    tune = columns[1].checkbox("Tune hyper-parameters", state.settings.tune_hyperparameters,
                               help="Slower. Uses a randomised search inside cross-validation.")
    aggressiveness = columns[2].selectbox("Preprocessing depth", ["minimal", "standard", "thorough"],
                                          index=1)
    candidates = [s.key for s in REGISTRY.find(task_type=objective.task_type, only_available=True)]
    include = st.multiselect("Always include these models", candidates)
    exclude = st.multiselect("Never use these models", candidates)

settings = RunSettings(
    mode=mode_key,
    max_models=max_models,
    time_budget=time_budget,
    interpretability_need=interpretability,
    tune_hyperparameters=tune,
    test_size=test_size,
    random_state=int(random_state),
    preprocessing_aggressiveness=aggressiveness,
    include_models=include,
    exclude_models=exclude,
)
state.settings = settings

# --------------------------------------------------------------------------
# plan preview
# --------------------------------------------------------------------------
st.divider()
st.subheader("What will run")

plan = plan_analysis(
    profile, objective, state.context,
    Constraints(
        max_models=max_models, time_budget=time_budget,
        interpretability_need=interpretability,
        include_models=include, exclude_models=exclude,
    ),
    frame=frame,
)

metric_row([
    ("Models to train", str(len(plan.candidates)), "Chosen to span different model families"),
    ("Ranked on", plan.primary_metric, "The metric the comparison is ranked by"),
    ("Validation", plan.validation_strategy.get("strategy", "—").replace("_", " "),
     plan.validation_strategy.get("reason", "")),
    ("Folds", str(plan.validation_strategy.get("n_splits", "—")), "Cross-validation splits"),
])

for warning in plan.warnings:
    caveat(warning)

# The brief comes first: someone should be able to approve or reject the work
# before it runs, without reading a model shortlist.
brief = build_brief(profile, objective, plan, state.context, frame)
contract = build_contract(profile, objective, frame)

verdict_tone = {"suitable": "good", "caution": "warning", "unsuitable": "critical"}[brief.verdict]
st.markdown(
    f'### Analysis brief &nbsp; {badge(brief.verdict_label, verdict_tone)}',
    unsafe_allow_html=True,
)
if brief.verdict == "unsuitable":
    error_state(
        "This data cannot defensibly answer this question",
        "  \n".join(f"- {r}" for r in brief.verdict_reasons),
        "Change the objective above, fix what is flagged on the **Data** page, or accept that "
        "the honest answer here is that the data will not support a conclusion. Running it "
        "anyway will still produce numbers.",
    )
elif brief.verdict == "caution":
    caveat(
        "This will run and produce results. The reasons below are what a reader should be told "
        "alongside them.",
        label="Proceed with caution",
    )

brief_tab, contract_tab, candidates_tab, table_tab, reasoning_tab = st.tabs(
    ["The brief", "Data contract", "Candidate models", "As a table", "Why these choices"]
)

with brief_tab:
    st.markdown(brief.render())

with contract_tab:
    st.caption(
        "What this analysis requires of its data. Stored with the run and checked again when new "
        "rows arrive to be scored — a model is only valid on data meeting the contract it was "
        "trained under."
    )
    simple_table([
        {"Requirement": check.requirement,
         "Status": ("met" if check.passed
                    else "NOT MET — stops the analysis" if check.severity == "blocking"
                    else "raised"),
         "Detail": check.detail}
        for check in contract.checks
    ])
    if not contract.satisfied:
        caveat(contract.summary(), label="Contract not satisfied")
with candidates_tab:
    ai_panel(
        f"Based on your objective and this dataset, I would evaluate these "
        f"**{len(plan.candidates)} models**, ranked by how well each suits the data rather than by "
        f"reputation. They are compared on **{plan.primary_metric}** against the same "
        f"{plan.validation_strategy.get('strategy', 'validation').replace('_', ' ')} split.",
        heading="AI model selection",
    )
    for position, candidate in enumerate(plan.candidates, start=1):
        marks = []
        if position == 1:
            marks.append(("recommended", "accent"))
        elif position <= 3:
            marks.append(("strong alternative", "neutral"))
        if candidate.family == "baseline":
            marks.append(("baseline", "neutral"))
        if candidate.concerns:
            marks.append(("has a caveat", "warning"))
        rank_item(
            position,
            candidate.name,
            body=(candidate.reasons[0] if candidate.reasons else "General-purpose fit for this shape of data.")
                 + (f"  \n**Concern.** {candidate.concerns[0]}" if candidate.concerns else ""),
            metrics=[
                ("Fit score", f"{candidate.score:.2f}"),
                ("Family", candidate.family.replace("_", " ")),
                ("Interpretability", candidate.interpretability.replace("_", " ")),
                ("Cost", candidate.cost.replace("_", " ")),
            ],
            badges=marks,
            lead=position == 1,
        )
    st.caption(
        "The shortlist deliberately spans model families. If a simple model matches a complex one, "
        "that is itself a finding. Nothing here has been trained yet — these are candidates."
    )

with table_tab:
    leaderboard(
        [
            {
                "#": position, "Model": c.name, "Family": c.family,
                "Fit score": f"{c.score:.3f}", "Interpretability": c.interpretability,
                "Cost": c.cost,
                "Chosen because": c.reasons[0] if c.reasons else "general-purpose fit",
                "Concern": c.concerns[0] if c.concerns else "—",
            }
            for position, c in enumerate(plan.candidates, start=1)
        ],
        columns=["#", "Model", "Family", "Fit score", "Interpretability", "Cost",
                 "Chosen because", "Concern"],
        numeric={"#", "Fit score"},
    )

with reasoning_tab:
    decision_panel(plan.decisions)

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
st.divider()
label = {
    "automatic": "▶  Run autonomous analysis",
    "assisted": "▶  Run the approved plan",
    "manual": "▶  Run with my settings",
}[mode_key]

if st.button(label, type="primary", use_container_width=True):
    placeholder = st.empty()
    events: list = []

    # Named stages rather than a spinner: a long operation should say what it is
    # doing, both because it is more useful and because it is the truth.
    placeholder.markdown(
        f'<div class="dsai-ai" data-busy="yes">'
        f'<div class="dsai-ai-head"><span class="dsai-ai-mark">{AI_MARK}</span>'
        f"<span>AI Analyst at work</span></div>{skeleton(3)}</div>",
        unsafe_allow_html=True,
    )

    def on_event(event):
        events.append(event)
        placeholder.markdown(
            f'<div class="dsai-ai" data-busy="yes">'
            f'<div class="dsai-ai-head"><span class="dsai-ai-mark">{AI_MARK}</span>'
            f"<span>AI Analyst at work</span></div>{live_stages(events)}</div>",
            unsafe_allow_html=True,
        )

    engine = scientist()
    engine.trace_callback = on_event
    try:
        if True:
            run = engine.understand(state.frame, state.dataset_name, state.context, state.source)
            run.objective = objective
            engine.plan(run, objective, settings)
            if use_my_pipeline and state.pipeline is not None:
                run.pipeline = state.pipeline
            engine.execute(run)
            engine.interpret(run)
        state.run = run
        state.runs.append(run)
        if run.pipeline is not None:
            state.add_pipeline("from last run", run.pipeline)
        state.typed_frame = engine._typed_frame
        state.notify("success", f"Analysis complete in {run.duration_s}s. {run.summary()}")
        st.rerun()
    except Exception as exc:
        error_state(
            "The analysis could not finish",
            f"It stopped at `{type(exc).__name__}` — {exc}",
            _failure_advice(exc, objective),
        )
    finally:
        engine.trace_callback = None

if state.run is not None:
    st.divider()
    st.subheader("Last run")
    st.markdown(state.run.summary())
    trace_view(state.run.trace)
    st.markdown("Continue to **Models** to see the comparison, or **Insights** for the findings.")
