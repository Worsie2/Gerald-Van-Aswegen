"""End-to-end: the platform must run, and must be honest about what it found."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.core.schema import BusinessContext, Confidence, EvidenceKind, Objective, TaskType
from dsai.engines.orchestrator import AIDataScientist, RunSettings


@pytest.fixture(scope="module")
def analysed(request):
    """One full regression run, reused across assertions to keep the suite quick."""
    rng = np.random.default_rng(42)
    n = 400
    tier = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    frame = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "region": rng.choice(["Gauteng", "Western Cape"], n),
        "service_tier": tier,
        "employees": rng.lognormal(3, 0.8, n).round(),
        "meters_installed": rng.poisson(12, n),
    })
    frame["annual_spend"] = (
        18_000 + 900 * frame.meters_installed + 40 * frame.employees
        + pd.Series(tier).map({"basic": 0, "standard": 15_000, "premium": 60_000}).to_numpy()
        + rng.normal(0, 8_000, n)
    ).clip(1_000)
    context = BusinessContext(
        description="400 customers of a water instrumentation company",
        target_variable="annual_spend", currency="ZAR",
        assumptions=["next year resembles last year"],
    )
    scientist = AIDataScientist()
    run = scientist.analyse(frame, "water", context, RunSettings(max_models=5))
    return run, scientist, frame


def test_run_completes_with_every_stage(analysed):
    run, _, _ = analysed
    assert run.status == "complete"
    assert run.profile is not None
    assert run.objective.task_type is TaskType.REGRESSION
    assert run.plan is not None and run.plan.candidates
    assert run.pipeline is not None
    assert run.results and run.tournament is not None
    assert run.best is not None
    assert run.findings and run.recommendations
    assert run.self_check is not None


def test_trace_records_the_workflow(analysed):
    run, _, _ = analysed
    steps = " ".join(e.step.lower() for e in run.trace.events)
    for expected in ["profiling", "objective", "candidate methods", "preprocessing",
                     "training", "comparing", "findings", "recommendations"]:
        assert expected in steps, f"the trace never mentions {expected}"
    assert not run.trace.failures()


def test_model_recovers_the_planted_relationship(analysed):
    run, _, _ = analysed
    assert run.best.primary("r2") > 0.6
    drivers = [f.feature for f in run.explanation.importances[:3]]
    assert any("service_tier" in d or "meters" in d for d in drivers), \
        "the strongest planted driver should surface in the top three"


def test_recommendations_are_traceable_and_caveated(analysed):
    run, _, _ = analysed
    for recommendation in run.recommendations:
        assert recommendation.reason.strip()
        assert recommendation.evidence or recommendation.category == "further_analysis"
    actions = [r for r in run.recommendations if r.category == "action"]
    assert actions
    assert any(r.traceable_to for r in actions)


def test_causation_caveat_is_never_dropped(analysed):
    run, _, _ = analysed
    text = " ".join(
        [c for r in run.recommendations for c in r.caveats]
        + [c for f in run.findings for c in f.caveats]
        + run.self_check.caveats
    ).lower()
    assert "association" in text and "cause" in text


def test_user_assumptions_stay_labelled_as_assumptions(analysed):
    run, _, _ = analysed
    assumption_findings = [f for f in run.findings if f.kind is EvidenceKind.USER_ASSUMPTION]
    assert assumption_findings, "context the user supplied must appear as an assumption"
    for finding in assumption_findings:
        assert "not verified" in " ".join(finding.evidence).lower() or \
               "your assumption" in " ".join(finding.evidence).lower()


def test_decision_log_records_reasons_and_rejections(analysed):
    run, _, _ = analysed
    assert run.decisions
    for decision in run.decisions:
        assert decision.decision.strip() and decision.reason.strip()
    assert any(d.rejected for d in run.decisions), "alternatives considered must be recorded"


def test_recommendation_never_claims_universal_superiority(analysed):
    run, _, _ = analysed
    text = " ".join(d.reason for d in run.decisions if d.stage == "model_recommendation").lower()
    assert "not best in general" in text or "for this dataset" in text


def test_report_and_exports_are_produced(analysed, tmp_path):
    from dsai.reporting.exporters import export_all, to_markdown

    run, _, _ = analysed
    markdown = to_markdown(run)
    assert "Executive summary" in markdown
    assert "Recommendations" in markdown or "What to do" in markdown
    assert "Reproducibility" in markdown
    assert "bound method" not in markdown, "a method reference leaked into the report text"

    written = export_all(run, tmp_path, "data.csv")
    for label in ("markdown", "html", "json", "python", "excel"):
        assert not str(written[label]).startswith("failed"), f"{label} export failed: {written[label]}"


def test_generated_code_reproduces_the_analysis(analysed, tmp_path):
    """The honest test of reproducibility: run the exported script."""
    import subprocess
    import sys

    from dsai.repro.codegen import generate_script

    run, _, frame = analysed
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    script_path = tmp_path / "reproduce.py"
    script_path.write_text(generate_script(run, str(data_path)))

    completed = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, f"generated script failed:\n{completed.stderr[-2000:]}"
    assert "R2" in completed.stdout or "Cross-validated" in completed.stdout


def test_manifest_fingerprint_is_stable(analysed):
    from dsai.repro.provenance import build_manifest

    run, _, _ = analysed
    first = build_manifest(run).fingerprint()
    second = build_manifest(run).fingerprint()
    assert first == second and len(first) == 16


def test_clustering_run_profiles_its_segments(cluster_frame):
    scientist = AIDataScientist()
    objective = Objective(task_type=TaskType.CLUSTERING,
                          features=list(cluster_frame.columns), n_clusters=3)
    run = scientist.understand(cluster_frame, "segments")
    scientist.plan(run, objective, RunSettings(max_models=3))
    scientist.execute(run)
    scientist.interpret(run)

    assert run.segmentation is not None
    assert run.segmentation.n_clusters >= 2
    for cluster in run.segmentation.clusters:
        if not cluster.is_noise:
            assert cluster.description and cluster.name


def test_forecast_run_analyses_the_series(series_frame):
    scientist = AIDataScientist()
    objective = Objective(task_type=TaskType.TIME_SERIES_FORECAST, target="sales",
                          time_column="month", horizon=12)
    run = scientist.understand(series_frame, "sales")
    scientist.plan(run, objective, RunSettings(max_models=4))
    scientist.execute(run)
    scientist.interpret(run)

    assert run.series_analysis["trend"]["direction"] == "increasing"
    assert run.best is not None and run.best.extras.get("forecast")


def test_assisted_mode_stops_before_running(regression_frame):
    scientist = AIDataScientist()
    run = scientist.analyse(regression_frame, "water", BusinessContext(target_variable="annual_spend"),
                            RunSettings(mode="assisted", max_models=3))
    assert run.status == "planned"
    assert not run.results, "assisted mode must wait for approval before training anything"
    assert run.plan is not None and run.plan.candidates
