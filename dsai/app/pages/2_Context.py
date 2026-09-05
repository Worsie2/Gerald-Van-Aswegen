"""Context page: tell the platform what the data represents."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import page_header, require_data, show_notices, workflow_nav
from dsai.app.state import workspace
from dsai.core.profiler import apply_user_overrides
from dsai.core.schema import BusinessContext
from dsai.engines.objective import describe_objectives, detect_objectives

st.set_page_config(page_title="Context · DSAI", page_icon="💬", layout="wide")
state = workspace()
show_notices(state)

page_header(
    "Tell the platform about your data",
    "Optional, but it changes the analysis. Everything you enter here is treated as an assumption "
    "and labelled as such — it is never mixed with what was measured.",
    "Step 2 of 7",
)
workflow_nav("Context")

if not require_data(state):
    st.stop()

context = state.context
profile = state.profile

with st.form("context_form"):
    st.subheader("What this data is")
    description = st.text_area(
        "What does this dataset represent?",
        value=context.description, height=90,
        placeholder="e.g. 200 customers of a water instrumentation company, one row per customer, "
                    "covering the 2023 and 2024 financial years.",
    )
    columns = st.columns(2)
    data_source = columns[0].text_input("Where did it come from?", context.data_source,
                                        placeholder="e.g. exported from our CRM")
    industry = columns[1].text_input("Industry", context.industry, placeholder="e.g. water instrumentation")
    time_period = columns[0].text_input("Time period covered", context.time_period, placeholder="e.g. 2023-2024")
    geography = columns[1].text_input("Geographic context", context.geography, placeholder="e.g. South Africa")

    st.subheader("What you want to know")
    business_problem = st.text_area("The business problem", context.business_problem, height=70,
                                    placeholder="e.g. we want to grow revenue from the existing customer base")
    research_question = st.text_area("The research question", context.research_question, height=70,
                                     placeholder="e.g. which customers are likely to buy additional products?")
    stated_objective = st.text_area(
        "In one sentence, what do you want the platform to do?",
        context.stated_objective, height=70,
        placeholder="e.g. identify customer segments and predict which customers will purchase additional products",
    )
    desired_outcome = st.text_input("What would a good outcome look like?", context.desired_outcome)

    columns = st.columns(2)
    analysis_style = columns[0].selectbox(
        "What kind of analysis is this?",
        ["", "exploratory", "predictive", "explanatory", "decision-oriented"],
        index=(["", "exploratory", "predictive", "explanatory", "decision-oriented"].index(context.analysis_style)
               if context.analysis_style in ["", "exploratory", "predictive", "explanatory", "decision-oriented"] else 0),
        help=(
            "Exploratory: understand what is there. Predictive: forecast accurately. "
            "Explanatory: understand why, which favours interpretable models. "
            "Decision-oriented: support a specific choice."
        ),
    )
    currency = columns[1].text_input("Currency", context.currency or "ZAR")

    st.subheader("Variables")
    all_columns = list(profile.columns)
    target_variable = st.selectbox(
        "Which column is the outcome you care about? (optional)",
        ["— let the platform decide —"] + all_columns,
        index=(all_columns.index(context.target_variable) + 1
               if context.target_variable in all_columns else 0),
    )
    columns = st.columns(2)
    include_variables = columns[0].multiselect(
        "Only use these variables (leave empty to use all)", all_columns, context.include_variables
    )
    exclude_variables = columns[1].multiselect(
        "Never use these variables", all_columns, context.exclude_variables,
        help="Identifiers, anything unavailable at prediction time, or anything you are not allowed to use.",
    )

    st.subheader("Assumptions and limits")
    assumptions = st.text_area(
        "Assumptions you are making (one per line)", "\n".join(context.assumptions), height=80,
        placeholder="e.g. next year's behaviour will resemble the last three years",
    )
    limitations = st.text_area(
        "Known limitations of the data (one per line)", "\n".join(context.limitations), height=80,
        placeholder="e.g. no competitor pricing; customers who left before 2022 are not in the file",
    )

    with st.expander("Explain what individual variables mean"):
        meanings = {}
        for name in all_columns:
            value = st.text_input(f"`{name}`", context.variable_meanings.get(name, ""), key=f"meaning_{name}")
            if value.strip():
                meanings[name] = value.strip()

    submitted = st.form_submit_button("Save context", type="primary")

if submitted:
    state.context = BusinessContext(
        description=description.strip(),
        data_source=data_source.strip(),
        business_problem=business_problem.strip(),
        research_question=research_question.strip(),
        desired_outcome=desired_outcome.strip(),
        industry=industry.strip(),
        time_period=time_period.strip(),
        geography=geography.strip(),
        assumptions=[a.strip() for a in assumptions.splitlines() if a.strip()],
        limitations=[l.strip() for l in limitations.splitlines() if l.strip()],
        variable_meanings=meanings,
        include_variables=include_variables,
        exclude_variables=exclude_variables,
        target_variable=None if target_variable.startswith("—") else target_variable,
        analysis_style=analysis_style,
        currency=currency.strip() or "ZAR",
        stated_objective=stated_objective.strip(),
    )
    apply_user_overrides(state.profile, state.context)
    state.objectives = detect_objectives(state.profile, state.context)
    state.reset_analysis()
    state.notify("success", "Context saved. The suggested objectives below have been updated.")
    st.rerun()

st.divider()
st.subheader("What the platform now thinks you want")
if state.objectives:
    st.markdown(describe_objectives(state.objectives))
    st.caption(
        "Objectives marked 'you asked for this' come from what you typed. The rest were inferred "
        "from the data alone. You choose which to run on the Analysis page."
    )
else:
    st.info("No objectives identified yet.")

if not state.context.is_empty:
    with st.expander("What the platform will treat as your assumption"):
        st.code(state.context.as_prompt_block(), language=None)
        st.caption(
            "None of this is verified against the data. It shapes which analysis is chosen and how "
            "results are described, and it is reported separately from what was measured."
        )
