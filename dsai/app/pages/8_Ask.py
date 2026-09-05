"""Ask page: put a question in plain language and see the plan before it runs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import dataframe, page_header, require_data, show_notices
from dsai.app.state import scientist, workspace
from dsai.engines.nl import answer_question, intent_to_objective, parse_command
from dsai.engines.orchestrator import RunSettings

st.set_page_config(page_title="Ask · DSAI", page_icon="💭", layout="wide")
state = workspace()
show_notices(state)

page_header(
    "Ask a question",
    "Type it in plain language. The platform shows you exactly what it intends to run before it "
    "runs anything expensive.",
)

if not require_data(state):
    st.stop()

EXAMPLES = [
    "Which customers are most valuable?",
    "What drives annual spend?",
    "Create five customer segments.",
    "Find unusual customers.",
    "Forecast sales for the next six months.",
    "Is the difference between regions statistically significant?",
    "Try every suitable regression model.",
    "What should I investigate next?",
]
st.caption("Examples: " + " · ".join(f"*{e}*" for e in EXAMPLES[:4]))

for message in state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("table") is not None:
            dataframe(message["table"])

question = st.chat_input("Ask about your data…")
if question:
    state.chat.append({"role": "user", "content": question})
    intent = parse_command(question, state.profile, state.context, state.run)
    st.session_state["_pending_intent"] = intent
    st.rerun()

intent = st.session_state.get("_pending_intent")
if intent is not None:
    with st.chat_message("assistant"):
        st.markdown("**Here is what I understood, and what I would run:**")
        st.markdown(intent.describe())

        if intent.clarification:
            st.info(intent.clarification, icon="❓")

        if intent.action in {"describe", "correlate", "test", "rank"} and not intent.clarification:
            frame = state.typed_frame if state.typed_frame is not None else state.frame
            result = answer_question(question, frame, state.profile, state.context, state.run)
            if result.get("filters"):
                st.caption("Filters applied: " + "; ".join(result["filters"]))
            if result.get("answer"):
                st.markdown(result["answer"])
                state.chat.append({"role": "assistant", "content": result["answer"]})
            for key in ("numeric_summary", "categorical_summary", "correlations"):
                table = result.get(key)
                if table is not None and not table.empty:
                    dataframe(table.round(4))
            if intent.action == "rank" and intent.target:
                identifier = state.profile.identifier_columns[0] if state.profile.identifier_columns else None
                top_n = intent.parameters.get("top_n", 20)
                columns = [c for c in ([identifier] if identifier else []) + [intent.target]
                           if c in frame.columns]
                ranked = frame.nlargest(top_n, intent.target)[columns + [
                    c for c in frame.columns if c not in columns
                ][:4]]
                dataframe(ranked)
                share = frame.nlargest(top_n, intent.target)[intent.target].sum() / frame[intent.target].sum()
                st.info(
                    f"The top {top_n} rows account for {share:.1%} of total {intent.target}.",
                    icon="📊",
                )
            st.session_state["_pending_intent"] = None

        elif intent.action in {"explain", "recommend"} and state.run is not None:
            if intent.action == "explain" and state.run.explanation:
                for line in state.run.explanation.plain_english:
                    st.markdown(f"- {line}")
                state.chat.append({
                    "role": "assistant",
                    "content": "\n".join(f"- {l}" for l in state.run.explanation.plain_english),
                })
            elif intent.action == "recommend":
                for recommendation in state.run.recommendations[:6]:
                    st.markdown(f"- **{recommendation.action}** — {recommendation.reason}")
            st.session_state["_pending_intent"] = None

        elif intent.task_type is not None:
            st.warning(
                f"This will train models and may take a while (estimated cost: {intent.estimated_cost}). "
                "Nothing runs until you approve it.",
                icon="⚠️",
            )
            columns = st.columns([1, 1, 4])
            if columns[0].button("Run it", type="primary"):
                objective = intent_to_objective(intent, state.profile)
                settings = RunSettings(
                    mode="automatic",
                    max_models=intent.parameters.get("max_models", 8),
                    time_budget=intent.parameters.get("time_budget", "balanced"),
                    interpretability_need=intent.parameters.get("interpretability_need", "moderate"),
                    tune_hyperparameters=intent.parameters.get("tune_hyperparameters", False),
                    include_models=intent.model_keys,
                )
                engine = scientist()
                with st.spinner("Running…"):
                    run = engine.analyse(
                        state.frame, state.dataset_name, state.context, settings,
                        state.source, objective,
                    )
                state.run = run
                state.runs.append(run)
                state.pipeline = run.pipeline
                state.typed_frame = engine._typed_frame
                state.chat.append({"role": "assistant", "content": f"Done. {run.summary()}"})
                st.session_state["_pending_intent"] = None
                st.rerun()
            if columns[1].button("Cancel"):
                st.session_state["_pending_intent"] = None
                st.rerun()
        else:
            st.session_state["_pending_intent"] = None

if st.button("Clear conversation"):
    state.chat = []
    st.session_state["_pending_intent"] = None
    st.rerun()
