"""A structured review of an analysis: what a second pair of eyes should check.

Reviewing someone else's analysis is mostly a search problem — the decisions are
in one place, the caveats in another, the validation somewhere else, and the
reviewer has to assemble the picture before they can question it. This assembles
it for them.

Deliberately not a verdict. The review says what each area looks like and what
is worth questioning; whether the analysis is acceptable is a judgement the
reviewer makes, and a tool that made it for them would be answering the question
they were brought in to answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsai.engines.metrics import human_number

__all__ = ["ReviewArea", "ReviewPackage", "build_review"]

OK = "ok"
QUESTION = "question"
CONCERN = "concern"


@dataclass
class ReviewArea:
    name: str
    status: str = OK
    summary: str = ""
    #: What a reviewer should actually look at, phrased as things to check.
    checks: list[str] = field(default_factory=list)
    #: Where in the application to look.
    where: str = ""


@dataclass
class ReviewPackage:
    dataset: str = ""
    run_id: str = ""
    fingerprint: str = ""
    areas: list[ReviewArea] = field(default_factory=list)
    overall: str = ""
    n_questions: int = 0
    n_concerns: int = 0

    def to_markdown(self) -> str:
        marks = {OK: "ok", QUESTION: "worth asking about", CONCERN: "needs an answer"}
        lines = [
            f"# Review — {self.dataset}", "",
            f"Run `{self.run_id}` · fingerprint `{self.fingerprint}`", "",
            self.overall, "",
            "| Area | Status | Summary |", "| --- | --- | --- |",
        ]
        for area in self.areas:
            lines.append(f"| {area.name} | {marks[area.status]} | {area.summary} |")
        lines.append("")
        for area in self.areas:
            if not area.checks:
                continue
            lines += [f"## {area.name}", "", area.summary, ""]
            lines += [f"- {check}" for check in area.checks]
            if area.where:
                lines += ["", f"*Where to look: {area.where}*"]
            lines.append("")
        lines += [
            "---", "",
            "This is a structured view of the analysis, not a verdict on it. Whether it is "
            "acceptable is the reviewer's judgement — which is the question a reviewer was "
            "brought in to answer, and not one a tool should answer for them.",
        ]
        return "\n".join(lines)


def build_review(run: Any) -> ReviewPackage:
    package = ReviewPackage(
        dataset=run.dataset_name,
        run_id=run.id,
        fingerprint=(run.ledger.fingerprint if getattr(run, "ledger", None) else ""),
    )
    package.areas = [
        _question_area(run),
        _data_area(run),
        _preprocessing_area(run),
        _validation_area(run),
        _model_area(run),
        _evidence_area(run),
        _recommendation_area(run),
    ]
    package.n_questions = sum(1 for a in package.areas if a.status == QUESTION)
    package.n_concerns = sum(1 for a in package.areas if a.status == CONCERN)

    if package.n_concerns:
        package.overall = (
            f"**{package.n_concerns} area(s) need an answer** before this analysis should carry a "
            f"decision, and {package.n_questions} more are worth asking about."
        )
    elif package.n_questions:
        package.overall = (
            f"Nothing blocking. **{package.n_questions} area(s) are worth asking about** — none "
            "of them means the analysis is wrong, and all of them are things a reviewer should "
            "hear an answer to."
        )
    else:
        package.overall = (
            "Every area checks out against what this platform can test. That is not the same as "
            "the analysis being right: the questions a tool cannot ask — whether the data means "
            "what everyone assumes, whether the sample is the right one — are the reviewer's."
        )
    return package


def _question_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Question", where="Analysis → The brief")
    objective = run.objective
    if objective is None:
        return ReviewArea("Question", CONCERN, "No objective was recorded.",
                          ["Ask what this analysis was trying to establish."])
    area.summary = (f"{objective.task_type.value.replace('_', ' ')}"
                    + (f" on `{objective.target}`" if objective.target else ""))
    area.checks = [
        f"Is `{objective.target}` really the outcome that matters, or the one that was available?"
        if objective.target else "Is an unsupervised framing the right one here?",
        f"The platform chose this because: {objective.rationale}",
    ]
    brief = getattr(run, "brief", None)
    if brief is not None:
        if brief.verdict == "unsuitable":
            area.status = CONCERN
            area.summary += " — the brief said this data cannot defensibly answer it"
            area.checks += [f"Unresolved: {r}" for r in brief.verdict_reasons]
        elif brief.verdict == "caution":
            area.status = QUESTION
            area.checks += [f"Flagged at planning: {r}" for r in brief.verdict_reasons[:3]]
        if objective.source.startswith("ai"):
            area.checks.append("Nobody stated this objective — it was inferred from the data. "
                               "Confirm it matches what was actually wanted.")
    return area


def _data_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Data", where="Data → Data quality")
    profile = run.profile
    if profile is None:
        return ReviewArea("Data", CONCERN, "No profile recorded.", [])
    critical = profile.issues_by_severity("critical")
    area.summary = (f"{profile.n_rows:,} rows × {profile.n_columns} columns, "
                    f"quality {profile.quality_score:.0f}/100")
    if critical:
        area.status = CONCERN
        area.checks += [f"Unresolved critical issue: {i.message}" for i in critical]
    if profile.leakage_suspects:
        area.status = CONCERN
        area.checks += [
            f"Possible leakage in `{s['column']}` — {s['reason']}. Was this ruled out?"
            for s in profile.leakage_suspects[:4]
        ]
    missing = sum(c.n_missing for c in profile.columns.values())
    if missing:
        cells = max(profile.n_rows * profile.n_columns, 1)
        if area.status == OK:
            area.status = QUESTION
        area.checks.append(
            f"{missing:,} values ({missing / cells:.1%} of cells) were imputed. Is missingness "
            "here random, or does it fall on a particular kind of row?"
        )
    overridden = [n for n, c in profile.columns.items() if c.semantic_type_overridden]
    if overridden:
        area.checks.append(f"Column type(s) corrected by hand: {', '.join(overridden)}. "
                           "Were those corrections right?")
    if not area.checks:
        area.checks.append("Nothing was flagged in the data. Does it match what the source "
                           "system is understood to contain?")
    return area


def _preprocessing_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Preprocessing", where="Preprocessing → The pipeline")
    pipeline = run.pipeline
    if pipeline is None or not pipeline.active_steps:
        area.status = QUESTION
        area.summary = "No pipeline recorded"
        area.checks.append("The model was trained on the data as loaded. Was any cleaning needed?")
        return area
    steps = pipeline.active_steps
    area.summary = f"{len(steps)} step(s)"
    fitted = [s for s in steps if not s.spec.leakage_safe]
    area.checks = [
        f"{len(fitted)} step(s) learn from the data and are refitted inside each fold — "
        "structurally they cannot leak. Confirm the remaining steps are what was intended.",
    ]
    imputers = [s for s in steps if s.step_key.startswith("impute")]
    if imputers:
        area.status = QUESTION
        area.checks.append(
            f"{len(imputers)} imputation step(s). Every filled gap is an estimate the model then "
            "treats as data — is that acceptable for the columns involved?"
        )
    dropped = [s for s in steps if s.step_key.startswith("drop")]
    if dropped:
        area.checks.append(f"{len(dropped)} step(s) remove rows or columns. What went, and why?")
    return area


def _validation_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Validation", where="Report → Methodology & workings")
    if run.plan is None or run.best is None:
        return ReviewArea("Validation", CONCERN, "Nothing was validated.", [])
    strategy = run.plan.validation_strategy
    area.summary = (f"{str(strategy.get('strategy','')).replace('_',' ')}, "
                    f"{strategy.get('n_splits','—')} folds, ranked on {run.plan.primary_metric}")
    area.checks = [f"Chosen because: {strategy.get('reason', '—')}"]

    check = run.self_check
    if check is not None:
        failed = [c for c in check.checks if not c.passed]
        if check.blocking:
            area.status = CONCERN
            area.checks += [f"Blocking: {b}" for b in check.blocking]
        elif failed:
            area.status = QUESTION
            area.checks += [f"Raised: {c.question} — {c.detail[:140]}" for c in failed]

    metric = run.plan.primary_metric
    spread = run.best.validation.std_scores.get(metric)
    mean = run.best.validation.mean_scores.get(metric)
    if spread and mean and abs(mean) > 1e-9:
        area.checks.append(
            f"Fold-to-fold spread is {spread / abs(mean):.0%} of the mean "
            f"({human_number(spread)} on {human_number(mean)}). Any single figure for performance "
            "is approximate to about that much."
        )
    return area


def _model_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Model", where="Models → Model card")
    if run.best is None:
        return ReviewArea("Model", CONCERN, "No model completed.",
                          ["Every candidate failed. What stopped them?"])
    area.summary = f"{run.best.model_name} of {len(run.results)} tried"
    area.checks = [
        "The ranking is composite — performance 50%, generalisation 20%, stability 15%, "
        "interpretability 10%, cost 5%. Is that weighting right for this decision?",
    ]
    gap = run.best.overfitting_gap
    if gap is not None and gap > 0.1:
        area.status = QUESTION
        area.checks.append(f"Train-to-held-out gap of {gap:.0%} — it has partly memorised the "
                           "training rows. Was a simpler model considered?")
    failed = [r for r in run.results if r.status != "success"]
    if failed:
        area.checks.append(f"{len(failed)} candidate(s) failed to train: "
                           + ", ".join(r.model_name for r in failed[:4])
                           + ". Would any of them have been preferable?")
    return area


def _evidence_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Evidence", where="Insights")
    ledger = getattr(run, "ledger", None)
    trust = getattr(run, "trust", None)
    if not run.findings:
        return ReviewArea("Evidence", CONCERN, "No findings were produced.", [])
    area.summary = f"{len(run.findings)} finding(s)"
    if ledger is not None:
        area.summary += f", {len(ledger.items)} piece(s) of evidence"
        area.checks.append(
            f"{ledger.measured_share:.0%} of the evidence is measurement; the rest is "
            "model-derived, interpretation or an assumption somebody supplied."
        )
        if ledger.measured_share < 0.3:
            area.status = QUESTION
    if trust is not None:
        area.checks.append(f"Evidence strength {trust.score}/100 ({trust.band}). "
                           f"Assumption debt {trust.assumption_debt}/100.")
        if trust.score < 55:
            area.status = CONCERN if trust.score < 35 else QUESTION
        area.checks += [f"Weakening the conclusion: {f.statement}" for f in trust.reducing[:4]]
    speculative = [f for f in run.findings if f.confidence.value in ("low", "speculative")]
    if speculative:
        area.checks.append(f"{len(speculative)} finding(s) are low-confidence or speculative. "
                           "Are they carrying any weight in the conclusion?")
    return area


def _recommendation_area(run: Any) -> ReviewArea:
    area = ReviewArea(name="Recommendations", where="Recommendations")
    if not run.recommendations:
        return ReviewArea("Recommendations", QUESTION, "None were produced.",
                          ["Was the analysis expected to produce actions?"])
    area.summary = f"{len(run.recommendations)} action(s)"
    ledger = getattr(run, "ledger", None)
    unlinked = []
    if ledger is not None:
        unlinked = [r for r in run.recommendations
                    if r.id and not ledger.rests_on.get(r.id)]
    if unlinked:
        area.status = QUESTION
        area.checks.append(
            f"{len(unlinked)} recommendation(s) are not linked to a specific finding: "
            + ", ".join(f"`{r.id}`" for r in unlinked[:4])
            + ". What is each one resting on?"
        )
    area.checks.append(
        "Every relationship here is associational. Does any recommendation assume that changing "
        "a predictor would change the outcome? That assumption is not supported by this design."
    )
    risky = [r for r in run.recommendations if r.confidence.value in ("low", "speculative")]
    if risky:
        area.checks.append(f"{len(risky)} action(s) rest on low-confidence evidence.")
    return area
