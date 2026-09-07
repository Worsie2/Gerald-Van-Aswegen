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
from dsai.engines.metrics import human_number
from dsai.engines.insight import group_by_evidence
from dsai.engines.recommend import group_by_category


@dataclass
class Report:
    title: str = ""
    generated_at: str = field(default_factory=lambda: _dt.datetime.now().strftime("%d %B %Y, %H:%M"))
    executive_summary: str = ""
    sections: list[tuple[str, str]] = field(default_factory=list)
    audience: str = "both"
    #: Charts belonging to the sections above, each naming the section it follows.
    figures: list[Any] = field(default_factory=list)

    def figures_for(self, heading: str) -> list[Any]:
        return [f for f in self.figures if f.after_section == heading]

    def to_markdown(self, chart_dir: str | None = None, chart_files: dict[str, str] | None = None) -> str:
        """Markdown, with images when the charts have been written to disk.

        ``chart_files`` maps a figure key to a path relative to the markdown
        file. Without it the caption and the table stand in for the picture, so
        the point the chart makes still reaches a reader looking at plain text.
        """
        chart_files = chart_files or {}
        lines = [f"# {self.title}", "", f"*Generated {self.generated_at}*", ""]
        if self.executive_summary:
            lines += ["## Executive summary", "", self.executive_summary, ""]
        for heading, body in self.sections:
            lines += [f"## {heading}", "", body, ""]
            for figure in self.figures_for(heading):
                lines += [f"### {figure.title}", ""]
                if figure.key in chart_files:
                    lines += [f"![{figure.alt}]({chart_files[figure.key]})", ""]
                lines += [figure.caption, ""]
                if figure.has_table:
                    lines += ["<!-- the numbers behind the chart -->",
                              _markdown_table(figure.table, max_rows=20), ""]
        return "\n".join(lines)

    def to_text(self) -> str:
        """Plain text for a terminal, with markdown emphasis stripped."""
        text = self.to_markdown()
        for token in ("**", "`"):
            text = text.replace(token, "")
        return text


def build_report(
    run: Any,
    audience: str = "both",
    currency: str | None = None,
    include_charts: bool = True,
    frame: pd.DataFrame | None = None,
    mode: str = "light",
    include_methodology: bool = False,
) -> Report:
    """Assemble the full report from a completed :class:`AnalysisRun`.

    ``frame`` is optional: given it, the report can draw the relationships in the
    raw data as a heatmap; without it, the measured pairs on the profile carry
    the same point.

    ``include_methodology`` adds three sections that answer a different question
    from the rest of the report — how the analysis was run, the arithmetic behind
    every figure it reports, and whether anything went wrong producing it. Off by
    default because most readers do not want them; available always because the
    reader who does should not have to ask anyone for them.
    """
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
    # Placed straight after the findings, not in an appendix: whether to believe
    # a conclusion is not a footnote to the conclusion.
    sections.append(("Why trust this", _trust_section(run)))
    if audience == "technical":
        sections.append(("Recommendations", _recommendations_section(run, plain=False)))
    sections.append(("Risks and limitations", _limitations_section(run)))
    sections.append(("Suggested next analyses", _next_steps_section(run)))
    if include_methodology:
        sections.append(("The analysis brief", _brief_section(run)))
        sections.append(("Data contract", _contract_section(run)))
        sections.append(("Evidence ledger", _evidence_section(run)))
        sections.append(("Model card", _model_card_section(run)))
        sections.append(("Methodology", _methodology_section(run)))
        sections.append(("The calculations", _calculations_section(run)))
        sections.append(("Did it run cleanly", _run_integrity_section(run)))
    if audience in ("both", "technical"):
        sections.append(("How the platform decided", _decision_log_section(run)))
        sections.append(("Reproducibility", _reproducibility_section(run)))

    report.sections = [(heading, body) for heading, body in sections if body.strip()]

    if include_charts:
        try:
            from dsai.reporting.figures import build_figures

            headings = {heading for heading, _ in report.sections}
            report.figures = [
                figure for figure in build_figures(run, mode=mode, frame=frame)
                if figure.after_section in headings
            ]
        except Exception:
            # Charts illustrate the report; they are not the report. A plotting
            # library that is missing or unhappy must not cost the reader the text.
            report.figures = []

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
            f"| {feature.rank} | `{feature.feature}` | {human_number(feature.importance)} | "
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
                f"| `{row['term']}` | {human_number(row['coefficient'])} | {human_number(row.get('std_error', float('nan')))} | "
                f"{row['p_value']:.4f} | {human_number(row.get('ci_lower', float('nan')))} to "
                f"{human_number(row.get('ci_upper', float('nan')))} |"
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


# --------------------------------------------------------------------------
# methodology, calculations, and did it actually run
#
# These three answer a different question from the rest of the report. The rest
# says what was found; these say how, with what, and whether anything went wrong
# on the way. A reader who cannot check the second set has to take the first on
# trust.
# --------------------------------------------------------------------------

def _trust_section(run: Any) -> str:
    """How far this can be trusted, and what nothing here can settle."""
    trust = getattr(run, "trust", None)
    if trust is None:
        return ""
    lines = [
        f"**Evidence strength: {trust.score}/100 — {trust.band}.** "
        f"Assumption debt {trust.assumption_debt}/100 ({trust.debt_band}).",
        "",
        "Evidence strength is a weighted summary of the validation checks that passed and "
        f"failed. It is **not** a statistical confidence level: it does not mean there is an "
        f"{trust.score}% chance the conclusion is right.",
        "",
        trust.verdict,
        "",
    ]
    if trust.supporting:
        lines += ["**Supporting**", ""] + [f"- {f.statement}" for f in trust.supporting] + [""]
    if trust.reducing:
        lines += ["**Reducing confidence**", ""] + [f"- {f.statement}" for f in trust.reducing] + [""]
    if trust.unknowns:
        lines += [
            "**What this analysis cannot tell you**",
            "",
            "Not deficiencies to be fixed — the boundary of what this design can establish.",
            "",
        ] + [f"- {u}" for u in trust.unknowns] + [""]
    if trust.debt_items:
        lines += [
            "**Assumption debt**",
            "",
            "What would have to be true for the conclusion to hold, and has not been shown.",
            "",
        ] + [f"- {item}" for item in trust.debt_items]
    return "\n".join(lines)


def _brief_section(run: Any) -> str:
    """The plan as it stood before the work, with its verdict."""
    brief = getattr(run, "brief", None)
    if brief is None:
        return ""
    return brief.render()


def _contract_section(run: Any) -> str:
    """What the analysis required of its data, and whether it got it."""
    contract = getattr(run, "contract", None)
    if contract is None:
        return ""
    lines = [
        contract.summary(),
        "",
        "These requirements are stored with the run and checked again whenever new data is "
        "scored: a model is valid only on data meeting the contract it was trained under.",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in contract.checks:
        status = ("met" if check.passed
                  else "**not met — blocking**" if check.severity == "blocking"
                  else "raised")
        lines.append(f"| {check.requirement} | {status} | {check.detail.replace('|', chr(92) + '|')} |")
    return "\n".join(lines)


def _evidence_section(run: Any) -> str:
    """Every claim with an identifier, and what it rests on."""
    ledger = getattr(run, "ledger", None)
    if ledger is None or not ledger.items:
        return ""
    lines = [
        f"Every finding and recommendation carries an identifier derived from its own content, so "
        f"a claim can be cited and traced after it has left this report. This run holds "
        f"**{len(ledger.items)} piece(s) of evidence** behind **{len(ledger.supports)} finding(s)**.",
        "",
        f"Run `{ledger.run_id}` · fingerprint `{ledger.fingerprint}` · model {ledger.model} · "
        f"pipeline {ledger.pipeline} · dataset {ledger.dataset}",
        "",
        f"{ledger.measured_share:.0%} of the evidence is measurement (counted or tested); the "
        "rest is model-derived, interpretation, or something you told the platform.",
        "",
    ]
    for finding in run.findings:
        if not finding.id:
            continue
        lines += [f"**{finding.id} — {finding.title}**", ""]
        supporting = ledger.evidence_for(finding.id)
        if supporting:
            lines += [f"- `{item.id}` {item.statement} *({item.kind.value.replace('_', ' ')})*"
                      for item in supporting]
        else:
            lines.append("- *(no structured evidence recorded)*")
        for caveat in finding.caveats[:2]:
            lines.append(f"- Limit: {caveat}")
        lines.append("")

    linked = [(r, ledger.rests_on.get(r.id, [])) for r in run.recommendations if r.id]
    if linked:
        lines += ["**Recommendations and what they rest on**", "",
                  "| ID | Action | Rests on |", "| --- | --- | --- |"]
        for recommendation, findings in linked:
            names = ", ".join(f"`{f}`" for f in findings) or "—"
            lines.append(f"| `{recommendation.id}` | {recommendation.action} | {names} |")
    return "\n".join(lines)


def _model_card_section(run: Any) -> str:
    if run.best is None:
        return ""
    try:
        from dsai.reporting.model_card import build_model_card

        return build_model_card(run).to_markdown().split("\n", 1)[1].strip()
    except Exception:
        return ""


def _methodology_section(run: Any) -> str:
    """The method in full: the design, the split, the metrics and their definitions."""
    plan, objective, best = run.plan, run.objective, run.best
    lines: list[str] = []

    if objective is not None:
        lines += [
            "**The question**",
            "",
            f"- Problem type: {objective.task_type.value.replace('_', ' ')}",
            f"- Target variable: `{objective.target}`" if objective.target
            else "- No target variable — this is an unsupervised problem",
        ]
        if objective.features:
            lines.append(f"- Predictors restricted to {len(objective.features)} named column(s)")
        lines += [f"- Chosen because: {objective.rationale}", ""]

    if plan is not None:
        strategy = plan.validation_strategy
        lines += [
            "**Validation design**",
            "",
            f"- Strategy: {strategy.get('strategy', '—').replace('_', ' ')}",
            f"- Folds: {strategy.get('n_splits', '—')}",
            f"- Chosen because: {strategy.get('reason', '—')}",
            f"- Ranked on: {plan.primary_metric}",
            "",
            "Every model in the comparison was given the same split, the same folds and the same "
            "random seed. A comparison where the models saw different data is not a comparison.",
            "",
        ]

    if best is not None:
        total = best.n_train + best.n_test
        lines += [
            "**How the rows were divided**",
            "",
            "| | Rows | Share | Used for |",
            "| --- | ---: | ---: | --- |",
            f"| Training | {best.n_train:,} | {best.n_train / total:.1%} | "
            "Fitting the model, and fitting every preprocessing step that learns anything |",
            f"| Held out | {best.n_test:,} | {best.n_test / total:.1%} | "
            "Scored once, at the end. Never seen during fitting or selection |",
            f"| Total | {total:,} | 100% | |",
            "",
            f"Within the training rows, {run.plan.validation_strategy.get('n_splits', '—') if run.plan else '—'} "
            "cross-validation folds were cut. Each fold refits the preprocessing from scratch on "
            "its own training portion, so a value from the fold's validation rows cannot reach "
            "back into how those rows were transformed.",
            "",
            f"- Random seed: {best.random_seed} (recorded, so the split can be reproduced exactly)",
            f"- Features after preprocessing: {best.n_features_out}",
            "",
        ]

    metric_names = M.default_metrics(best.task_type) if best else []
    if metric_names:
        lines += [
            "**What each metric measures, and how it is computed**",
            "",
            "| Metric | Direction | Definition | What it means |",
            "| --- | --- | --- | --- |",
        ]
        for metric in metric_names:
            better = "higher is better" if M.higher_is_better(metric) else "lower is better"
            formula = M.metric_formula(metric).replace("|", "\\|")
            lines.append(
                f"| {M.METRIC_LABELS.get(metric, metric)} | {better} | "
                f"{formula or '—'} | {M.explain_metric(metric)} |"
            )
        lines.append("")

    if run.pipeline is not None and run.pipeline.active_steps:
        leakage = run.pipeline.leakage_report()
        lines += [
            "**Where each preprocessing step was fitted**",
            "",
            "| Step | Applies to | Fitted |",
            "| --- | --- | --- |",
        ]
        for step in run.pipeline.active_steps:
            where = ("before the split (changes which rows exist)" if step.spec.scope == "row"
                     else "inside each fold" if not step.spec.leakage_safe
                     else "inside each fold (learns nothing, so it could not leak anyway)")
            lines.append(f"| {step.spec.name} | {_step_scope(step)} | {where} |")
        lines += ["", leakage["explanation"], ""]

    return "\n".join(lines)


def _step_scope(step: Any) -> str:
    if not step.columns:
        return "all applicable columns"
    if isinstance(step.columns, str):
        return str(step.columns)
    listed = ", ".join(f"`{c}`" for c in step.columns[:6])
    return listed + (f" and {len(step.columns) - 6} more" if len(step.columns) > 6 else "")


def _calculations_section(run: Any) -> str:
    """The numbers behind every reported figure, fold by fold.

    A cross-validated score is an average of numbers nobody normally sees. Shown,
    they answer the question the average cannot: was the model consistently
    decent, or brilliant on one fold and useless on another?
    """
    if not run.results:
        return ""
    lines: list[str] = []
    successful = [r for r in run.results if r.status == "success"]
    if not successful:
        return "No model completed, so there is nothing to show the arithmetic for."

    primary = run.plan.primary_metric if run.plan else None
    lines += [
        "**Fold-by-fold scores**",
        "",
        "Each model's score on every cross-validation fold, and the mean those folds produce. "
        "A wide spread across folds means the reported average is not a reliable estimate of how "
        "the model will do on new data, however good that average looks.",
        "",
    ]

    for result in successful:
        folds = result.validation.fold_scores.get(primary or "", []) if primary else []
        if not folds:
            continue
        mean = result.validation.mean_scores.get(primary, float("nan"))
        std = result.validation.std_scores.get(primary, float("nan"))
        header = "| Model | " + " | ".join(f"Fold {i + 1}" for i in range(len(folds))) + \
                 " | Mean | Std | Spread |"
        if header not in lines:
            lines += [header, "| --- | " + " | ".join("---:" for _ in folds) + " | ---: | ---: | ---: |"]
        spread = (max(folds) - min(folds)) if folds else 0.0
        cells = " | ".join(human_number(v) for v in folds)
        lines.append(
            f"| {result.model_name} | {cells} | **{human_number(mean)}** | "
            f"{human_number(std)} | {human_number(spread)} |"
        )
    lines += [
        "",
        f"The mean is the plain arithmetic mean of the fold scores: "
        f"(fold 1 + … + fold n) ÷ n. It is reported for {primary or 'the primary metric'}, "
        "which is what the comparison was ranked on.",
        "",
    ]

    lines += [
        "**Training against held-out performance**",
        "",
        "The gap between them is the whole question of whether a model has learned the data or "
        "memorised it.",
        "",
        f"| Model | Training {primary or ''} | Held-out {primary or ''} | Raw difference | "
        "Relative gap | Reading |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in successful:
        train = result.train_scores.get(primary) if primary else None
        test = result.test_scores.get(primary) if primary else None
        raw = (abs(test - train) if isinstance(train, (int, float))
               and isinstance(test, (int, float)) else None)
        gap = result.overfitting_gap
        reading = (
            "—" if gap is None else
            "memorising the training rows" if gap > 0.15 else
            "some overfitting, watch it" if gap > 0.05 else
            "generalising well"
        )
        lines.append(
            f"| {result.model_name} | {human_number(train)} | {human_number(test)} | "
            f"{human_number(raw)} | {human_number(gap)} | {reading} |"
        )
    lines += [
        "",
        "The **raw difference** is in the metric's own units. The **relative gap** is that "
        "difference scaled so it is comparable across models and across metrics — it is what the "
        "ranking uses, because a difference of 3,000 means something very different on a target "
        "measured in thousands than on one measured in millions. A relative gap above 0.15 is the "
        "threshold at which this platform calls a model overfitted.",
        "",
    ]

    best = run.best
    if best is not None and best.hyperparameters:
        lines += [
            f"**Hyper-parameters of the selected model ({best.model_name})**",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
        ]
        lines += [f"| `{k}` | `{v}` |" for k, v in sorted(best.hyperparameters.items())
                  if v is not None]
        lines.append("")

    if run.tournament is not None and getattr(run.tournament, "weights", None):
        weights = run.tournament.weights
        lines += [
            "**How the composite ranking was computed**",
            "",
            "The ranking is not the raw score. Each model's figures are normalised against the "
            "field, then combined with these weights:",
            "",
            "| Component | Weight | What it measures |",
            "| --- | ---: | --- |",
            f"| Performance | {weights.get('performance', 0):.0%} | The primary metric on held-out rows |",
            f"| Generalisation | {weights.get('generalisation', 0):.0%} | How small the train-to-test gap is |",
            f"| Stability | {weights.get('stability', 0):.0%} | How little the score varies across folds |",
            f"| Interpretability | {weights.get('interpretability', 0):.0%} | Whether the mechanism can be read |",
            f"| Efficiency | {weights.get('efficiency', 0):.0%} | Training cost |",
            "",
            "A model that wins on the headline metric can therefore rank below one that is "
            "marginally worse and far steadier. That is deliberate: the steadier one is the one "
            "that will still work next quarter.",
            "",
        ]

    if run.explanation is not None and run.explanation.importances:
        lines += [
            "**Feature importance, as numbers**",
            "",
            f"Method: {run.explanation.method}. {run.explanation.method_note}",
            "",
            "| Rank | Variable | Importance | ± std | Direction |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
        for item in run.explanation.top(20):
            lines.append(
                f"| {item.rank} | `{item.feature}` | {human_number(item.importance)} | "
                f"{human_number(getattr(item, 'std', None))} | {item.direction or '—'} |"
            )
        lines.append("")

    if run.explanation is not None and run.explanation.coefficients:
        lines += [
            "**Model coefficients**",
            "",
            "| Variable | Coefficient | Reading |",
            "| --- | ---: | --- |",
        ]
        for row in run.explanation.coefficients[:25]:
            lines.append(
                f"| `{row.get('feature', '')}` | {human_number(row.get('coefficient'))} | "
                f"{row.get('interpretation', '—')} |"
            )
        lines.append("")

    if run.diagnostics.get("confusion"):
        confusion = run.diagnostics["confusion"]
        labels = confusion.get("labels", [])
        matrix = confusion.get("matrix", [])
        if labels and matrix:
            lines += [
                "**Confusion matrix (counts)**",
                "",
                "Rows are what actually happened; columns are what the model said.",
                "",
                "| Actual \\ Predicted | " + " | ".join(str(l) for l in labels) + " | Total |",
                "| --- | " + " | ".join("---:" for _ in labels) + " | ---: |",
            ]
            for label, row in zip(labels, matrix):
                lines.append(f"| **{label}** | " + " | ".join(f"{v:,}" for v in row) +
                             f" | {sum(row):,} |")
            lines.append("")

    return "\n".join(lines)


def _run_integrity_section(run: Any) -> str:
    """Did it run smoothly — every step, its status, and how long it took.

    The honest answer to "can I trust this" starts with whether anything went
    wrong while producing it. A failure that is reported is a fact; a failure
    that is quietly absent from the report is a problem.
    """
    lines: list[str] = []
    events = list(getattr(run.trace, "events", []))
    failures = [e for e in events if e.status == "failed"]
    warnings = [e for e in events if e.status == "warning"]

    verdict = (
        "Every stage completed." if not failures and not warnings else
        f"{len(failures)} stage(s) failed and {len(warnings)} raised a warning."
        if failures else
        f"Every stage completed, with {len(warnings)} warning(s) raised along the way."
    )
    lines += [
        f"**{verdict}** The run took {run.duration_s:.1f}s in total and its status is "
        f"`{run.status}`.",
        "",
    ]

    if events:
        lines += [
            "| Stage | Status | Detail | Time |",
            "| --- | --- | --- | ---: |",
        ]
        for event in events:
            detail = (event.detail or "—").replace("|", "\\|")[:180]
            timing = f"{event.elapsed_s:.2f}s" if event.elapsed_s else "—"
            lines.append(f"| {event.step} | {event.status} | {detail} | {timing} |")
        lines.append("")

    attempted = run.results
    if attempted:
        succeeded = [r for r in attempted if r.status == "success"]
        lines += [
            f"**{len(succeeded)} of {len(attempted)} model(s) trained successfully.**",
            "",
        ]
        broken = [r for r in attempted if r.status != "success"]
        if broken:
            lines += ["| Model | Status | What went wrong |", "| --- | --- | --- |"]
            for result in broken:
                lines.append(f"| {result.model_name} | {result.status} | "
                             f"{(result.error or '—')[:200]} |")
            lines += [
                "",
                "A model that failed is reported rather than hidden. Most failures here mean the "
                "algorithm's requirements were not met by this data — too few rows, unencoded "
                "categories, or values it cannot accept — not that the analysis is unsound.",
                "",
            ]

    if run.self_check is not None:
        checks = run.self_check.checks
        passed = [c for c in checks if c.passed]
        lines += [
            f"**Self-check: {len(passed)} of {len(checks)} question(s) passed.**",
            "",
            "| Question | Result | What was found |",
            "| --- | --- | --- |",
        ]
        for check in checks:
            lines.append(
                f"| {check.question} | {'pass' if check.passed else 'raised'} | "
                f"{(check.detail or '—').replace('|', chr(92) + '|')[:200]} |"
            )
        lines.append("")
        if run.self_check.blocking:
            lines += ["**Blocking problems**", ""]
            lines += [f"- {item}" for item in run.self_check.blocking]
            lines.append("")

    if run.warnings:
        lines += ["**Warnings raised during the run**", ""]
        lines += [f"- {w}" for w in run.warnings[:20]]
        lines.append("")

    if run.environment:
        lines += [
            "**Environment**",
            "",
            "| Component | Version |",
            "| --- | --- |",
        ]
        lines += [f"| {k} | {v} |" for k, v in sorted(run.environment.items())]
        lines.append("")

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
