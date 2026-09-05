"""The final report, written twice: for a data scientist and for a manager.

Same analysis, same numbers, two registers. The management version leads with
what to do and what it rests on; the technical version leads with what was done
and how it was validated. Neither is allowed to claim more than the evidence
supports — the caveats travel with the findings into both.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import Confidence, EvidenceKind, Finding, Recommendation, TaskType
from dsai.engines import metrics as M
from dsai.engines.insight import group_by_evidence
from dsai.engines.recommend import group_by_category


@dataclass
class Report:
    title: str = ""
    generated_at: str = field(default_factory=lambda: _dt.datetime.now().strftime("%d %B %Y, %H:%M"))
    executive_summary: str = ""
    sections: list[tuple[str, str]] = field(default_factory=list)
    audience: str = "both"

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"*Generated {self.generated_at}*", ""]
        if self.executive_summary:
            lines += ["## Executive summary", "", self.executive_summary, ""]
        for heading, body in self.sections:
            lines += [f"## {heading}", "", body, ""]
        return "\n".join(lines)

    def to_text(self) -> str:
        """Plain text for a terminal, with markdown emphasis stripped."""
        text = self.to_markdown()
        for token in ("**", "`"):
            text = text.replace(token, "")
        return text


def build_report(run: Any, audience: str = "both", currency: str | None = None) -> Report:
    """Assemble the full report from a completed :class:`AnalysisRun`."""
    currency = currency or (run.context.currency if run.context else "ZAR")
    report = Report(
        title=f"Analysis report — {run.dataset_name}",
        audience=audience,
    )
    report.executive_summary = _executive_summary(run, currency)

    sections: list[tuple[str, str]] = []
    if audience in ("both", "business"):
        sections.append(("What this means for the business", _business_narrative(run, currency)))
        sections.append(("What to do", _recommendations_section(run, plain=True)))
    sections.append(("The data", _dataset_section(run)))
    sections.append(("Data quality", _quality_section(run)))
    if audience in ("both", "technical"):
        sections.append(("Preprocessing applied", _preprocessing_section(run)))
        sections.append(("Exploratory findings", _exploratory_section(run)))
        sections.append(("Statistical findings", _statistical_section(run)))
        sections.append(("Models evaluated", _models_section(run)))
        sections.append(("Model comparison", _comparison_section(run)))
        sections.append(("Selected model and performance", _performance_section(run)))
        sections.append(("What drives the outcome", _drivers_section(run)))
    sections.append(("Key findings", _findings_section(run)))
    if audience == "technical":
        sections.append(("Recommendations", _recommendations_section(run, plain=False)))
    sections.append(("Risks and limitations", _limitations_section(run)))
    sections.append(("Suggested next analyses", _next_steps_section(run)))
    if audience in ("both", "technical"):
        sections.append(("How the platform decided", _decision_log_section(run)))
        sections.append(("Reproducibility", _reproducibility_section(run)))

    report.sections = [(heading, body) for heading, body in sections if body.strip()]
    return report


def _markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    """Render a DataFrame as a markdown table.

    Written here rather than using ``DataFrame.to_markdown`` so that report
    generation does not depend on tabulate being installed.
    """
    if frame.empty:
        return "_(no rows)_"
    frame = frame.head(max_rows)
    headers = [str(c) for c in frame.columns]

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        if isinstance(value, float):
            return f"{value:,.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:,.0f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    if len(frame) == max_rows:
        lines.append(f"| _…truncated at {max_rows} rows_ |" + " |" * (len(headers) - 1))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _executive_summary(run: Any, currency: str) -> str:
    profile, objective = run.profile, run.objective
    parts: list[str] = []

    if profile:
        parts.append(
            f"**{run.dataset_name}** holds {profile.n_rows:,} records across {profile.n_columns} "
            f"variables. Data quality scores {profile.quality_score} out of 100"
            + (f", with {len(profile.issues_by_severity('critical'))} issue(s) needing attention"
               if profile.issues_by_severity("critical") else ", with no critical problems")
            + "."
        )
    if run.context and run.context.description:
        parts.append(f"You described this as: *{run.context.description}*")
    if objective:
        parts.append(
            f"The analysis was framed as **{objective.task_type.value.replace('_', ' ')}**"
            + (f" on `{objective.target}`" if objective.target else "")
            + f", {'because you asked for it' if objective.source.startswith('user') else 'based on what the data supports'}."
        )

    if run.tournament and run.tournament.recommended:
        best = run.tournament.recommended
        metric = run.tournament.primary_metric
        parts.append(
            f"Of {len(run.results)} models tested, **{best.model_name}** performed best "
            f"({M.METRIC_LABELS.get(metric, metric)} = "
            f"{M.format_metric(metric, best.primary_score)}, {best.score_source}). "
            + (f"That is {best.lift_over_baseline:+.0f}% against a model that ignores every predictor. "
               if best.lift_over_baseline is not None else "")
            + "This is the best model for this dataset under the validation strategy used — not a "
              "claim about models in general."
        )

    if run.explanation and run.explanation.importances:
        drivers = ", ".join(f"`{f.feature}`" for f in run.explanation.importances[:3])
        parts.append(f"The strongest drivers are {drivers}.")

    if run.segmentation and getattr(run.segmentation, "clusters", None):
        real = [c for c in run.segmentation.clusters if not c.is_noise]
        parts.append(f"{len(real)} distinct segment(s) were identified.")

    high_confidence = [r for r in run.recommendations if r.confidence in (Confidence.HIGH, Confidence.MODERATE)]
    if high_confidence:
        parts.append(
            f"{len(run.recommendations)} recommendation(s) follow, of which {len(high_confidence)} "
            "carry moderate or high confidence."
        )

    if run.self_check and not run.self_check.passed:
        parts.append(
            "**These conclusions did not pass every validation check.** "
            + " ".join(run.self_check.blocking)
        )
    return "\n\n".join(parts)


def _business_narrative(run: Any, currency: str) -> str:
    """The manager's version: what was found, what it rests on, what to do."""
    lines: list[str] = []
    objective = run.objective

    if objective and objective.target:
        lines.append(f"**The question:** what explains and predicts `{objective.target}`?")
    if run.context and run.context.business_problem:
        lines.append(f"**The business problem you stated:** {run.context.business_problem}")

    if run.tournament and run.tournament.recommended and run.best:
        metric, score = _readable_metric(run.objective, run.best, run.tournament.primary_metric)
        lines.append("**How reliable is this?** " + _plain_reliability(objective, metric, score, run.best))

    top_findings = [f for f in run.findings if f.confidence in (Confidence.HIGH, Confidence.MODERATE)][:5]
    if top_findings:
        lines.append("**What the analysis found:**")
        for finding in top_findings:
            lines.append(f"- {finding.title}. {finding.detail}")

    if run.segmentation and getattr(run.segmentation, "clusters", None):
        lines.append("**The segments:**")
        for cluster in run.segmentation.clusters:
            if cluster.is_noise:
                continue
            lines.append(f"- **{cluster.name}** — {cluster.description}")

    lines.append(
        "**One thing to keep in mind:** this analysis shows what moves together, not what causes "
        "what. Where a recommendation implies changing something, treat it as a hypothesis worth "
        "testing on a small group before rolling it out — that is the only way to find out whether "
        "the relationship survives being acted upon."
    )
    return "\n\n".join(lines)


def _readable_metric(objective, best, primary_metric: str) -> tuple[str, float | None]:
    """Pick the metric a non-specialist can actually interpret.

    RMSE and MAE are in the target's units, which is precise but says nothing
    about whether the model is any good. R² and AUC are bounded and therefore
    readable, so the business narrative prefers them when they exist.
    """
    preferred = {
        TaskType.REGRESSION: ["r2", "rmse", "mae"],
        TaskType.BINARY_CLASSIFICATION: ["roc_auc", "pr_auc", "f1"],
        TaskType.MULTICLASS_CLASSIFICATION: ["balanced_accuracy", "f1_macro", "accuracy"],
        TaskType.TIME_SERIES_FORECAST: ["mase", "mape", "rmse"],
    }.get(objective.task_type if objective else TaskType.EXPLORATORY, [primary_metric])
    for metric in preferred:
        score = best.primary(metric)
        if score is not None and np.isfinite(score):
            return metric, score
    return primary_metric, best.primary(primary_metric)


def _plain_reliability(objective, metric: str, score: float | None, best) -> str:
    if score is None:
        return "The models did not produce a usable score, so nothing here should be relied on."
    if metric == "r2":
        if score < 0.3:
            return (
                f"Weak. The model explains {score:.0%} of the variation in the outcome, which means "
                "most of what drives it is not in this data. Useful for spotting patterns, not for "
                "predicting individual cases."
            )
        if score < 0.6:
            return (
                f"Moderate. The model explains {score:.0%} of the variation — enough to rank and "
                "prioritise, not enough to promise a number for any individual case."
            )
        return (
            f"Good. The model explains {score:.0%} of the variation, which is strong for data about "
            "human or commercial behaviour."
        )
    if metric in {"roc_auc", "pr_auc"}:
        if score < 0.65:
            return f"Weak. At {score:.2f} the model is only slightly better than guessing."
        if score < 0.8:
            return (
                f"Moderate. At {score:.2f} the model ranks cases usefully — good for deciding who to "
                "contact first, not for a yes/no decision on one case."
            )
        return f"Strong. At {score:.2f} the model separates the two outcomes reliably."
    if metric == "mase":
        return (
            f"MASE {score:.2f} — the forecast is {'better' if score < 1 else 'no better'} than simply "
            "assuming next period looks like the last one."
        )
    if metric in {"rmse", "mae"}:
        return (
            f"Typical prediction error is about {M.format_metric(metric, score)} in the units of "
            f"`{objective.target}`. Whether that is acceptable depends on what decisions rest on it — "
            "compare it against the size of the amounts you are working with."
        )
    return f"{M.METRIC_LABELS.get(metric, metric)} = {M.format_metric(metric, score)}."


def _dataset_section(run: Any) -> str:
    profile = run.profile
    if profile is None:
        return ""
    lines = [
        f"- **Rows:** {profile.n_rows:,}",
        f"- **Columns:** {profile.n_columns}",
        f"- **Size in memory:** {profile.memory_mb:.2f} MB",
        f"- **Duplicate rows:** {profile.n_duplicate_rows:,} ({profile.duplicate_pct:.2f}%)",
        "",
        "**Variables by type**",
        "",
        "| Type | Count | Columns |",
        "| --- | --- | --- |",
    ]
    groups = {
        "Numeric": profile.numeric_columns,
        "Categorical": profile.categorical_columns,
        "Date/time": profile.datetime_columns,
        "Text": profile.text_columns,
        "Identifier": profile.identifier_columns,
        "Constant": profile.constant_columns,
    }
    for label, columns in groups.items():
        if columns:
            shown = ", ".join(f"`{c}`" for c in columns[:8])
            if len(columns) > 8:
                shown += f" (+{len(columns) - 8} more)"
            lines.append(f"| {label} | {len(columns)} | {shown} |")

    if run.context and not run.context.is_empty:
        lines += ["", "**Context you supplied** (assumption, not measurement)", "", run.context.as_prompt_block()]
    return "\n".join(lines)


def _quality_section(run: Any) -> str:
    profile = run.profile
    if profile is None:
        return ""
    lines = [f"**Quality score: {profile.quality_score}/100**", ""]
    if not profile.quality_issues:
        return lines[0] + "\n\nNo data-quality problems were detected."
    lines += ["| Severity | Issue | Affected columns | Suggested action |", "| --- | --- | --- | --- |"]
    for issue in sorted(profile.quality_issues, key=lambda i: {"critical": 0, "warning": 1, "info": 2}.get(i.severity, 3)):
        columns = ", ".join(f"`{c}`" for c in issue.columns[:5]) or "—"
        if len(issue.columns) > 5:
            columns += f" (+{len(issue.columns) - 5})"
        lines.append(
            f"| {issue.severity} | {issue.message} | {columns} | {issue.suggested_action or '—'} |"
        )
    return "\n".join(lines)


def _preprocessing_section(run: Any) -> str:
    pipeline = run.pipeline
    if pipeline is None or not pipeline.active_steps:
        return "No preprocessing was applied."
    lines = ["The pipeline, in order:", ""]
    lines += [f"{i + 1}. {step.describe()}" for i, step in enumerate(pipeline.active_steps)]
    leakage = pipeline.leakage_report()
    lines += [
        "",
        "**Leakage control**",
        "",
        leakage["explanation"],
        "",
        f"- Fitted inside each cross-validation fold: {', '.join(leakage['fitted_inside_cross_validation']) or 'none'}",
        f"- Applied before splitting: {', '.join(leakage['applied_before_split']) or 'none'}",
    ]
    return "\n".join(lines)


def _exploratory_section(run: Any) -> str:
    profile = run.profile
    if profile is None:
        return ""
    lines: list[str] = []
    strong = [(a, b, r) for a, b, r in profile.correlation_pairs if abs(r) >= 0.3][:12]
    if strong:
        lines += ["**Strongest correlations**", "", "| Variable A | Variable B | r | Reading |", "| --- | --- | --- | --- |"]
        from dsai.statistics.descriptive import interpret_correlation

        for a, b, r in strong:
            lines.append(f"| `{a}` | `{b}` | {r:.3f} | {interpret_correlation(r)} |")

    if profile.multicollinearity:
        worst = {k: v for k, v in list(profile.multicollinearity.items())[:8] if v >= 2}
        if worst:
            from dsai.statistics.descriptive import interpret_vif

            lines += ["", "**Multicollinearity (VIF)**", "", "| Variable | VIF | Reading |", "| --- | --- | --- |"]
            for name, value in worst.items():
                lines.append(f"| `{name}` | {value:.1f} | {interpret_vif(value)} |")

    nonlinear = [r for r in profile.relationships if r.kind == "nonlinearity"][:5]
    if nonlinear:
        lines += ["", "**Nonlinear relationships detected**", ""]
        lines += [f"- {r.message}" for r in nonlinear]

    if run.series_analysis:
        lines += ["", "**Time-series structure**", ""]
        for finding in run.series_analysis.get("findings", []):
            lines.append(f"- {finding}")
        for issue in run.series_analysis.get("issues", []):
            lines.append(f"- ⚠ {issue}")
    return "\n".join(lines)


def _statistical_section(run: Any) -> str:
    statistical = [f for f in run.findings if f.kind is EvidenceKind.STATISTICAL]
    if not statistical:
        return ""
    lines = []
    for finding in statistical:
        lines.append(f"**{finding.title}**")
        lines.append(finding.detail)
        if finding.evidence:
            lines += [f"  - {e}" for e in finding.evidence[:4]]
        lines.append("")
    return "\n".join(lines)


def _models_section(run: Any) -> str:
    if not run.results:
        return ""
    lines = [
        f"{len(run.results)} model(s) were trained and validated. Candidates were chosen by matching "
        "each algorithm's requirements against the measured characteristics of this dataset — sample "
        "size, dimensionality, missingness, nonlinearity, collinearity and interpretability need.",
        "",
        "| Model | Family | Interpretability | Status | Training time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in run.results:
        lines.append(
            f"| {result.model_name} | {result.family} | {result.interpretability} | "
            f"{result.status} | {result.training_time_s:.2f}s |"
        )
    failed = [r for r in run.results if r.status != "success"]
    if failed:
        lines += ["", "**Models that did not complete**", ""]
        lines += [f"- {r.model_name}: {r.error}" for r in failed]
    return "\n".join(lines)


def _comparison_section(run: Any) -> str:
    tournament = run.tournament
    if tournament is None or not tournament.table:
        return ""
    lines = [_markdown_table(pd.DataFrame(tournament.table)), ""]
    lines.append(
        "**How the ranking works.** The composite score weights performance "
        f"({tournament.weights['performance']:.0%}), generalisation "
        f"({tournament.weights['generalisation']:.0%}), stability across folds "
        f"({tournament.weights['stability']:.0%}), interpretability "
        f"({tournament.weights['interpretability']:.0%}) and training cost "
        f"({tournament.weights['efficiency']:.0%}). A model that scores highest while overfitting "
        "does not win on that basis alone."
    )
    if tournament.simplest_acceptable:
        lines.append(
            f"\n**Simplest acceptable model:** {tournament.simplest_acceptable.model_name} — within 5% "
            "of the best performance, and easier to explain."
        )
    if tournament.warnings:
        lines += ["", "**Warnings from the comparison**", ""]
        lines += [f"- {w}" for w in tournament.warnings]
    return "\n".join(lines)


def _performance_section(run: Any) -> str:
    best = run.best
    if best is None:
        return ""
    lines = [f"**Selected model: {best.model_name}**", ""]
    if best.hyperparameters:
        lines += ["Hyper-parameters:", ""]
        lines += [f"- `{k}` = `{v}`" for k, v in best.hyperparameters.items() if v is not None]
        lines.append("")

    lines += [
        f"- Trained on {best.n_train:,} rows, tested on {best.n_test:,} held-out rows",
        f"- {best.n_features_out} feature(s) after preprocessing",
        f"- Random seed {best.random_seed}",
        "",
        "| Metric | Cross-validated | ± std | Hold-out test | Training |",
        "| --- | --- | --- | --- | --- |",
    ]
    metric_names = M.default_metrics(best.task_type) or list(best.test_scores)
    for metric in metric_names:
        cv = best.validation.mean_scores.get(metric)
        std = best.validation.std_scores.get(metric)
        lines.append(
            f"| {M.METRIC_LABELS.get(metric, metric)} | {M.format_metric(metric, cv)} | "
            f"{M.format_metric(metric, std)} | {M.format_metric(metric, best.test_scores.get(metric))} | "
            f"{M.format_metric(metric, best.train_scores.get(metric))} |"
        )
    lines += ["", "**What these mean**", ""]
    lines += [f"- **{M.METRIC_LABELS.get(m, m)}:** {M.explain_metric(m)}" for m in metric_names[:4]]

    if best.overfitting_gap is not None:
        lines += [
            "",
            f"**Train-to-held-out gap:** {best.overfitting_gap:.4f}. "
            + ("Small — the model generalises." if best.overfitting_gap < 0.1
               else "Large enough to matter: the training figure overstates real-world performance."),
        ]
    if best.warnings:
        lines += ["", "**Warnings raised for this model**", ""]
        lines += [f"- {w}" for w in best.warnings]

    if run.diagnostics and run.diagnostics.get("usable"):
        lines += ["", "**Diagnostics**", "", run.diagnostics.get("interpretation", "")]
        for issue in run.diagnostics.get("issues", []):
            lines.append(f"- {issue}")
    return "\n".join(lines)


def _drivers_section(run: Any) -> str:
    explanation = run.explanation
    if explanation is None or not explanation.importances:
        return ""
    lines = [
        f"**Method:** {explanation.method}. {explanation.method_note}",
        "",
        "| Rank | Variable | Importance | Direction |",
        "| --- | --- | --- | --- |",
    ]
    for feature in explanation.importances[:15]:
        lines.append(
            f"| {feature.rank} | `{feature.feature}` | {feature.importance:,.4g} | "
            f"{feature.direction or '—'} |"
        )
    if explanation.plain_english:
        lines += ["", "**In plain terms**", ""]
        lines += [f"- {line}" for line in explanation.plain_english]

    significant = [c for c in explanation.coefficients if c.get("p_value") is not None]
    if significant:
        lines += [
            "", "**Coefficients with inference**", "",
            "| Term | Coefficient | Std error | p-value | 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in significant[:15]:
            lines.append(
                f"| `{row['term']}` | {row['coefficient']:,.4g} | {row.get('std_error', float('nan')):,.4g} | "
                f"{row['p_value']:.4f} | {row.get('ci_lower', float('nan')):,.4g} to "
                f"{row.get('ci_upper', float('nan')):,.4g} |"
            )
    if explanation.caveats:
        lines += ["", "**Caveats**", ""]
        lines += [f"- {c}" for c in explanation.caveats]
    return "\n".join(lines)


def _findings_section(run: Any) -> str:
    if not run.findings:
        return ""
    grouped = group_by_evidence(run.findings)
    lines: list[str] = []
    for heading, findings in grouped.items():
        lines.append(f"### {heading}")
        lines.append("")
        for finding in findings:
            lines.append(f"**{finding.title}** *(confidence: {finding.confidence.value})*")
            lines.append("")
            lines.append(finding.detail)
            if finding.evidence:
                lines.append("")
                lines += [f"- {e}" for e in finding.evidence[:5]]
            if finding.caveats:
                lines.append("")
                lines += [f"- ⚠ {c}" for c in finding.caveats[:3]]
            lines.append("")
    return "\n".join(lines)


def _recommendations_section(run: Any, plain: bool) -> str:
    if not run.recommendations:
        return ""
    grouped = group_by_category(run.recommendations)
    lines: list[str] = []
    for heading, recommendations in grouped.items():
        lines.append(f"### {heading}")
        lines.append("")
        for i, recommendation in enumerate(recommendations, 1):
            lines.append(f"**{i}. {recommendation.action}**")
            lines.append("")
            lines.append(f"*Why:* {recommendation.reason}")
            if recommendation.evidence:
                lines.append("")
                lines.append("*Evidence:*")
                lines += [f"- {e}" for e in recommendation.evidence[:4]]
            if recommendation.expected_impact:
                lines.append("")
                lines.append(f"*Expected impact:* {recommendation.expected_impact}")
            lines.append("")
            lines.append(f"*Confidence:* {recommendation.confidence.value}")
            if recommendation.caveats and not plain:
                lines.append("")
                lines += [f"- ⚠ {c}" for c in recommendation.caveats[:3]]
            elif recommendation.caveats:
                lines.append("")
                lines.append(f"*Before acting:* {recommendation.caveats[0]}")
            lines.append("")
    return "\n".join(lines)


def _limitations_section(run: Any) -> str:
    lines: list[str] = []
    if run.self_check:
        failed = [c for c in run.self_check.checks if not c.passed]
        if failed:
            lines += ["**Validation checks that raised something**", ""]
            for check in failed:
                lines.append(f"- **{check.question}** {check.detail}")
            lines.append("")
        else:
            lines.append("All validation checks passed.\n")

    if run.warnings:
        lines += ["**Warnings raised during the analysis**", ""]
        lines += [f"- {w}" for w in dict.fromkeys(run.warnings)]
        lines.append("")

    if run.context and run.context.limitations:
        lines += ["**Limitations you flagged**", ""]
        lines += [f"- {l}" for l in run.context.limitations]
        lines.append("")
    if run.context and run.context.assumptions:
        lines += ["**Assumptions you supplied** (every conclusion inherits these)", ""]
        lines += [f"- {a}" for a in run.context.assumptions]
        lines.append("")

    lines += [
        "**Inherent limitations of this analysis**",
        "",
        "- This is observational data. Relationships found are associations; only a controlled test "
        "can establish that acting on one changes the other.",
        "- Every result is conditional on the data supplied. Populations, periods or segments not "
        "represented here are outside what this analysis can speak to.",
        "- Model performance is measured against the past. It holds only while the underlying "
        "process behaves as it did.",
    ]
    return "\n".join(lines)


def _next_steps_section(run: Any) -> str:
    next_steps = [r for r in run.recommendations if r.category == "further_analysis"]
    lines: list[str] = []
    for i, recommendation in enumerate(next_steps, 1):
        lines.append(f"{i}. **{recommendation.action}** — {recommendation.reason}")
    if run.objectives and len(run.objectives) > 1:
        lines += ["", "**Other questions this dataset can answer**", ""]
        for objective in run.objectives[1:4]:
            lines.append(f"- {objective.label()} — {objective.rationale}")
    return "\n".join(lines)


def _decision_log_section(run: Any) -> str:
    if not run.decisions:
        return ""
    lines = [
        "Each decision the platform made, with its reason and the alternatives it set aside. "
        "This is a record of conclusions, not of internal reasoning.",
        "",
    ]
    for decision in run.decisions:
        lines.append(f"**[{decision.stage}] {decision.decision}**")
        lines.append("")
        lines.append(f"*Reason:* {decision.reason}")
        if decision.evidence:
            lines.append("")
            lines += [f"- {e}" for e in decision.evidence[:5]]
        if decision.rejected:
            lines.append("")
            lines.append("*Rejected:*")
            lines += [f"- {r['option']} — {r['reason']}" for r in decision.rejected[:5]]
        lines.append("")
        lines.append(f"*Confidence:* {decision.confidence.value}"
                     + ("  (overridden by you)" if decision.overridden_by_user else ""))
        lines.append("")
    return "\n".join(lines)


def _reproducibility_section(run: Any) -> str:
    from dsai.repro.provenance import build_manifest

    manifest = build_manifest(run)
    return "```\n" + manifest.render() + "\n```"
