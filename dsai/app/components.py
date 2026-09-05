"""Shared UI pieces: headers, cards, confidence badges, the AI-decision panel."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsai.core.schema import Confidence, Decision, EvidenceKind, Finding, Recommendation

CONFIDENCE_ICON = {
    Confidence.HIGH: "🟢", Confidence.MODERATE: "🟡",
    Confidence.LOW: "🟠", Confidence.SPECULATIVE: "🔴",
}
EVIDENCE_LABEL = {
    EvidenceKind.OBSERVED: "Measured in the data",
    EvidenceKind.STATISTICAL: "Statistical test",
    EvidenceKind.MODEL: "Model-derived",
    EvidenceKind.INTERPRETATION: "Platform interpretation",
    EvidenceKind.USER_ASSUMPTION: "Your assumption — not verified",
}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}


def page_header(title: str, subtitle: str = "", step: str = "") -> None:
    import streamlit as st

    if step:
        st.caption(step)
    st.title(title)
    if subtitle:
        st.markdown(f"<p style='color:#6b7280;margin-top:-0.6rem'>{subtitle}</p>", unsafe_allow_html=True)


def workflow_nav(current: str) -> None:
    """The seven-stage workflow, with the current stage marked."""
    import streamlit as st

    stages = ["Data", "Context", "Preprocessing", "Analysis", "Models", "Insights", "Recommendations"]
    # Raw HTML throughout: markdown emphasis is not processed inside an
    # unsafe_allow_html block, so ** would render literally.
    rendered = " → ".join(
        f"<strong>{s}</strong>" if s.lower() == current.lower()
        else f"<span style='color:#9ca3af'>{s}</span>"
        for s in stages
    )
    st.markdown(
        f"<div style='font-size:0.82rem;letter-spacing:.02em;margin-bottom:1.25rem'>{rendered}</div>",
        unsafe_allow_html=True,
    )


def confidence_badge(confidence: Confidence) -> str:
    return f"{CONFIDENCE_ICON.get(confidence, '⚪')} {confidence.value}"


def metric_row(items: list[tuple[str, Any, str]]) -> None:
    import streamlit as st

    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text or None)


def finding_card(finding: Finding, expanded: bool = False) -> None:
    import streamlit as st

    header = f"{confidence_badge(finding.confidence)}  {finding.title}"
    with st.expander(header, expanded=expanded):
        st.caption(EVIDENCE_LABEL.get(finding.kind, finding.kind.value))
        st.write(finding.detail)
        if finding.evidence:
            st.markdown("**Evidence**")
            for item in finding.evidence[:6]:
                st.markdown(f"- {item}")
        if finding.caveats:
            for caveat in finding.caveats[:3]:
                st.warning(caveat, icon="⚠️")


def recommendation_card(recommendation: Recommendation, index: int = 0) -> None:
    import streamlit as st

    with st.container(border=True):
        st.markdown(f"**{recommendation.action}**")
        st.caption(f"{confidence_badge(recommendation.confidence)} · {recommendation.category.replace('_', ' ')}")
        st.write(recommendation.reason)
        if recommendation.evidence:
            with st.expander("Evidence", expanded=False):
                for item in recommendation.evidence:
                    st.markdown(f"- {item}")
                if recommendation.traceable_to:
                    st.caption("Traces back to: " + ", ".join(recommendation.traceable_to))
        if recommendation.expected_impact:
            st.markdown(f"*Expected impact:* {recommendation.expected_impact}")
        for caveat in recommendation.caveats[:2]:
            st.warning(caveat, icon="⚠️")


def decision_panel(decisions: list[Decision], stage: str | None = None) -> None:
    """The AI decision log: what was chosen, why, and what was rejected."""
    import streamlit as st

    shown = [d for d in decisions if stage is None or d.stage == stage]
    if not shown:
        st.info("No decisions recorded for this stage yet.")
        return
    for decision in shown:
        with st.expander(f"{confidence_badge(decision.confidence)}  {decision.decision}", expanded=False):
            st.caption(f"Stage: {decision.stage}"
                       + ("  ·  overridden by you" if decision.overridden_by_user else ""))
            st.write(decision.reason)
            if decision.evidence:
                st.markdown("**Evidence**")
                for item in decision.evidence:
                    st.markdown(f"- {item}")
            if decision.rejected:
                st.markdown("**Considered and set aside**")
                for rejected in decision.rejected:
                    st.markdown(f"- **{rejected['option']}** — {rejected['reason']}")


def quality_issues(profile: Any) -> None:
    import streamlit as st

    if not profile.quality_issues:
        st.success("No data-quality problems detected.")
        return
    for issue in sorted(profile.quality_issues,
                        key=lambda i: {"critical": 0, "warning": 1, "info": 2}.get(i.severity, 3)):
        icon = SEVERITY_ICON.get(issue.severity, "⚪")
        with st.expander(f"{icon}  {issue.message}", expanded=issue.severity == "critical"):
            if issue.columns:
                st.markdown("**Columns:** " + ", ".join(f"`{c}`" for c in issue.columns[:20]))
            if issue.suggested_action:
                st.markdown(f"**Suggested action:** {issue.suggested_action}")
            if issue.detail:
                st.json(issue.detail, expanded=False)


def trace_view(trace: Any, limit: int = 60) -> None:
    import streamlit as st

    icons = {"done": "✓", "running": "⏳", "warning": "⚠️", "failed": "✗", "skipped": "–"}
    lines = []
    for event in trace.events[-limit:]:
        timing = f"  ({event.elapsed_s:.2f}s)" if event.elapsed_s else ""
        detail = f" — {event.detail}" if event.detail else ""
        lines.append(f"{icons.get(event.status, '·')} {event.step}{detail}{timing}")
    st.code("\n".join(lines), language=None)


def override_notice(message: str = "") -> None:
    import streamlit as st

    st.caption(
        message or "Every choice on this page is a recommendation. Change anything you disagree with — "
                   "your override is recorded in the decision log."
    )


def dataframe(frame: pd.DataFrame, **kwargs: Any) -> None:
    import streamlit as st

    st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)


def require_data(state: Any) -> bool:
    import streamlit as st

    if not state.has_data:
        st.info("Load a dataset on the **Data** page first.", icon="📄")
        return False
    return True


def require_run(state: Any) -> bool:
    import streamlit as st

    if not state.has_run:
        st.info("Run an analysis on the **Analysis** page first.", icon="🧪")
        return False
    return True


def show_notices(state: Any) -> None:
    import streamlit as st

    for level, message in state.take_notices():
        {"success": st.success, "warning": st.warning, "error": st.error}.get(level, st.info)(message)
