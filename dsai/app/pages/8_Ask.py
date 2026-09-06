"""Ask page: put a question in plain language and see the plan before it runs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, dataframe, decision_needed, inference, page_header,
    require_data, show_notices, sidebar_chrome,
)
from dsai.app.state import scientist, workspace
from dsai.engines.nl import answer_question, intent_to_objective, parse_command
from dsai.engines.orchestrator import RunSettings

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
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
ai_panel(
    f"Ask me anything about **{state.dataset_name}**. I will show you what I understood and "
    "exactly what I intend to run — the plan first, then the work, so nothing expensive or "
    "wrong-headed happens without you seeing it coming.",
    heading="Ask the AI Analyst",
)

# Real buttons, not a caption listing examples: an example you have to retype is
# an example most people do not try.
st.caption("Try one of these, or type your own below.")
for row in (EXAMPLES[:4], EXAMPLES[4:8]):
    for column, example in zip(st.columns(len(row)), row):
        if column.button(example, key=f"eg_{example[:18]}", use_container_width=True):
            state.chat.append({"role": "user", "content": example})
            st.session_state["_pending_intent"] = parse_command(
                example, state.profile, state.context, state.run
            )
            st.rerun()

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
            caveat(intent.clarification, label="Needs clarifying")

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
                inference(
                    f"The top {top_n} rows account for {share:.1%} of total {intent.target}.",
                    label="Concentration",
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
            decision_needed(
                f"This will train models and may take a while (estimated cost: {intent.estimated_cost}). "
                "Nothing runs until you approve it.",
                label="Waiting on you",
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
                if run.pipeline is not None:
                    state.add_pipeline("from last run", run.pipeline)
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
