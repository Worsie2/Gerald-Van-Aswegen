"""Trust, evidence, the brief, the contract and the model card.

These carry a shared obligation: none of them may overstate what the analysis
established. Several tests below check wording rather than numbers, because the
failure mode for all of this is a score that reads like a probability or a claim
that reads like a cause.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dsai.core.evidence import Evidence, build_ledger, evidence_id, finding_id, recommendation_id
from dsai.core.schema import BusinessContext, EvidenceKind, TaskType
from dsai.engines.brief import CAUTION, SUITABLE, UNSUITABLE, build_brief, build_contract
from dsai.engines.trust import assess_trust
from dsai.reporting.model_card import build_model_card


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------

def test_ids_are_derived_from_content_not_a_counter():
    """A counter renumbers everything when one item disappears; a hash does not."""
    assert finding_id("Premium customers spend more") == finding_id("Premium customers spend more")
    assert finding_id("a") != finding_id("b")
    assert finding_id("x").startswith("F-")
    assert recommendation_id("x").startswith("R-")
    assert evidence_id("x", "observed_in_data").startswith("E-")


def test_evidence_assigns_itself_an_id():
    item = Evidence(statement="median difference R42,300", kind=EvidenceKind.STATISTICAL)
    assert item.id.startswith("E-")
    assert item.is_measurement


def test_interpretation_is_not_a_measurement():
    assert not Evidence(statement="x", kind=EvidenceKind.INTERPRETATION).is_measurement
    assert not Evidence(statement="x", kind=EvidenceKind.USER_ASSUMPTION).is_measurement
    assert not Evidence(statement="x", kind=EvidenceKind.MODEL).is_measurement


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

def test_ledger_gives_every_finding_an_id(regression_run):
    run, _ = regression_run
    ledger = build_ledger(run)
    assert run.findings
    assert all(f.id.startswith("F-") for f in run.findings)
    assert all(r.id.startswith("R-") for r in run.recommendations)
    assert ledger.fingerprint


def test_ledger_chain_reaches_the_dataset(regression_run):
    run, _ = regression_run
    ledger = build_ledger(run)
    chain = ledger.chain(run.recommendations[0].id)
    levels = [row["level"] for row in chain]
    for expected in ("Recommendation", "Run", "Model", "Pipeline", "Dataset"):
        assert expected in levels


def test_ledger_records_user_assumptions_as_unverified():
    from dsai.app.samples import build_sample
    from dsai.engines.orchestrator import AIDataScientist, RunSettings

    frame, source = build_sample("water_customers")
    engine = AIDataScientist()
    run = engine.analyse(
        frame, "wc", BusinessContext(assumptions=["Prices held constant"]),
        RunSettings(max_models=2, time_budget="fast"), source,
    )
    kinds = {item.kind for item in run.ledger.items.values()}
    assert EvidenceKind.USER_ASSUMPTION in kinds
    assumption = next(i for i in run.ledger.items.values()
                      if i.kind is EvidenceKind.USER_ASSUMPTION)
    assert not assumption.is_measurement
    assert "not verified" in assumption.render()


def test_ledger_does_not_invent_links(regression_run):
    """A recommendation naming no finding is linked to none, not to a guess."""
    run, _ = regression_run
    ledger = build_ledger(run)
    for recommendation, linked in ledger.rests_on.items():
        for found in linked:
            assert found in ledger.labels


# --------------------------------------------------------------------------
# trust
# --------------------------------------------------------------------------

def test_trust_never_calls_itself_a_probability(regression_run):
    run, _ = regression_run
    trust = assess_trust(run)
    assert 0 <= trust.score <= 100
    assert trust.band in ("strong", "moderate", "weak", "very weak", "not usable")
    # The verdict must not promise causation.
    assert "proof" not in trust.verdict.lower() or "not proof" in trust.verdict.lower()


def test_trust_lists_what_it_counted(regression_run):
    run, _ = regression_run
    trust = assess_trust(run)
    assert trust.supporting or trust.reducing
    assert all(f.statement for f in trust.supporting + trust.reducing)


def test_known_unknowns_are_always_present(regression_run):
    """The boundary of the method does not depend on how well the model did."""
    run, _ = regression_run
    trust = assess_trust(run)
    assert len(trust.unknowns) >= 3
    joined = " ".join(trust.unknowns).lower()
    assert "cause" in joined
    assert "population" in joined or "represent" in joined


def test_observational_data_always_reduces_confidence(regression_run):
    run, _ = regression_run
    trust = assess_trust(run)
    assert any("bservational" in f.statement for f in trust.reducing)


def test_a_run_with_no_self_check_says_so():
    class Bare:
        self_check = None
    trust = assess_trust(Bare())
    assert trust.score == 0
    assert "not been validated" in trust.verdict


def test_assumption_debt_rises_with_unverified_assumptions():
    from dsai.app.samples import build_sample
    from dsai.engines.orchestrator import AIDataScientist, RunSettings

    frame, source = build_sample("water_customers")
    settings = RunSettings(max_models=2, time_budget="fast")
    plain = AIDataScientist().analyse(frame, "wc", BusinessContext(description="x"), settings, source)
    loaded = AIDataScientist().analyse(
        frame, "wc",
        BusinessContext(description="x", assumptions=["a", "b", "c"]), settings, source,
    )
    assert loaded.trust.assumption_debt > plain.trust.assumption_debt


# --------------------------------------------------------------------------
# the brief
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def planned():
    from dsai.app.samples import build_sample
    from dsai.core.profiler import profile_dataset
    from dsai.engines.decision import Constraints, plan_analysis
    from dsai.engines.objective import detect_objectives

    frame, _ = build_sample("water_customers")
    profile, typed = profile_dataset(frame, name="water_customers")
    context = BusinessContext(description="water customers")
    objective = detect_objectives(profile, context)[0]
    plan = plan_analysis(profile, objective, context, Constraints(max_models=5), frame=typed)
    return profile, objective, plan, context, typed


def test_brief_asks_the_question_in_plain_english(planned):
    brief = build_brief(*planned[:4], planned[4])
    assert "?" in brief.question
    assert "regression" not in brief.question.lower()      # a task type is not a question


def test_brief_reaches_a_verdict(planned):
    brief = build_brief(*planned[:4], planned[4])
    assert brief.verdict in (SUITABLE, CAUTION, UNSUITABLE)
    assert brief.verdict_reasons


def test_brief_always_names_the_observational_constraint(planned):
    brief = build_brief(*planned[:4], planned[4])
    assert any("bservational" in c for c in brief.constraints)


def test_a_tiny_dataset_cannot_defensibly_answer_anything():
    from dsai.core.profiler import profile_dataset
    from dsai.engines.decision import Constraints, plan_analysis
    from dsai.engines.objective import detect_objectives

    frame = pd.DataFrame({"a": range(12), "b": range(12), "y": range(12)})
    profile, typed = profile_dataset(frame, name="tiny")
    context = BusinessContext()
    objective = detect_objectives(profile, context)[0]
    plan = plan_analysis(profile, objective, context, Constraints(max_models=3), frame=typed)
    brief = build_brief(profile, objective, plan, context, typed)
    assert brief.verdict == UNSUITABLE
    assert any("rows" in r for r in brief.verdict_reasons)


def test_leakage_downgrades_the_verdict():
    from dsai.app.samples import build_sample
    from dsai.core.profiler import profile_dataset
    from dsai.engines.decision import Constraints, plan_analysis
    from dsai.engines.objective import detect_objectives

    frame, _ = build_sample("messy_survey")
    profile, typed = profile_dataset(frame, name="messy")
    context = BusinessContext()
    objective = detect_objectives(profile, context)[0]
    plan = plan_analysis(profile, objective, context, Constraints(max_models=3), frame=typed)
    brief = build_brief(profile, objective, plan, context, typed)
    assert brief.verdict != SUITABLE
    assert any("give away the answer" in r for r in brief.verdict_reasons)


def test_an_empty_registry_does_not_masquerade_as_a_verdict(planned):
    """"No algorithm suits your data" must mean that, not "nothing was loaded"."""
    profile, objective, plan, *_ = planned
    assert plan.candidates


# --------------------------------------------------------------------------
# the data contract
# --------------------------------------------------------------------------

def test_contract_records_what_the_analysis_needs(planned):
    profile, objective, _, _, typed = planned
    contract = build_contract(profile, objective, typed)
    assert contract.target == objective.target
    assert contract.required_columns
    assert contract.checks
    assert contract.satisfied


def test_contract_blocks_on_a_missing_target(planned):
    from dsai.engines.brief import check_contract

    profile, objective, _, _, typed = planned
    contract = build_contract(profile, objective, typed)
    contract.target = "not_a_column"
    checks = check_contract(contract, profile, objective, typed)
    target_check = next(c for c in checks if c.requirement == "Target exists")
    assert not target_check.passed
    assert target_check.severity == "blocking"


def test_contract_captures_the_categories_seen_in_training(planned):
    profile, objective, _, _, typed = planned
    contract = build_contract(profile, objective, typed)
    assert contract.categories
    for column, values in contract.categories.items():
        assert values == sorted(values)


def test_contract_is_stored_with_the_run(regression_run):
    run, _ = regression_run
    assert run.contract is not None
    assert run.brief is not None
    assert run.trust is not None
    assert run.ledger is not None


# --------------------------------------------------------------------------
# the model card
# --------------------------------------------------------------------------

def test_model_card_says_what_the_model_is_not_for(regression_run):
    run, frame = regression_run
    card = build_model_card(run, frame=frame)
    assert card.inappropriate_uses
    joined = " ".join(card.inappropriate_uses).lower()
    assert "cause" in joined
    assert "different population" in joined


def test_model_card_never_promises_performance_elsewhere(regression_run):
    run, frame = regression_run
    markdown = build_model_card(run, frame=frame).to_markdown()
    assert "not a guarantee of performance anywhere else" in markdown


def test_model_card_carries_the_fingerprint(regression_run):
    run, frame = regression_run
    card = build_model_card(run, frame=frame)
    assert card.fingerprint
    assert card.fingerprint in card.to_markdown()


def test_model_card_subgroups_need_enough_rows(regression_run):
    """A group of nine with a bad score is noise, not a finding."""
    run, frame = regression_run
    card = build_model_card(run, frame=frame)
    assert all(row["n"] >= 30 for row in card.subgroups)


def test_model_card_survives_a_run_with_no_model():
    class Empty:
        best = None
    card = build_model_card(Empty())
    assert card.model_name == "none"


# --------------------------------------------------------------------------
# reports and the audit package
# --------------------------------------------------------------------------

def test_report_carries_the_trust_section(regression_run):
    from dsai.reporting.builder import build_report

    run, frame = regression_run
    body = dict(build_report(run, frame=frame).sections).get("Why trust this", "")
    assert "Evidence strength" in body
    assert "not** a statistical confidence level" in body
    assert "cannot tell you" in body


def test_methodology_adds_brief_contract_evidence_and_card(regression_run):
    from dsai.reporting.builder import build_report

    run, frame = regression_run
    headings = [h for h, _ in build_report(run, frame=frame, include_methodology=True).sections]
    for wanted in ("The analysis brief", "Data contract", "Evidence ledger", "Model card"):
        assert wanted in headings


def test_audit_package_writes_everything_with_one_fingerprint(regression_run, tmp_path):
    from dsai.reporting.exporters import export_audit_package

    run, frame = regression_run
    written = export_audit_package(run, tmp_path / "analysis", frame=frame)
    root = tmp_path / "analysis"
    for name in ("README.md", "methodology.md", "model_card.md", "findings.json",
                 "recommendations.json", "decision_log.json", "evidence.json", "trust.json",
                 "data_contract.json", "dataset_manifest.json"):
        assert (root / name).exists(), f"{name} missing"

    readme = (root / "README.md").read_text()
    assert run.ledger.fingerprint in readme
    assert "association" in readme.lower()

    trust = json.loads((root / "trust.json").read_text())
    assert "not a statistical confidence level" in trust["note"]
    assert trust["known_unknowns"]


def test_audit_package_names_what_it_could_not_write(regression_run, tmp_path, monkeypatch):
    """A missing file is reported, not quietly absent."""
    import dsai.reporting.exporters as exporters

    run, frame = regression_run
    monkeypatch.setattr(exporters, "_model_card_markdown",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    written = exporters.export_audit_package(run, tmp_path / "analysis", frame=frame)
    assert written["model_card"].startswith("failed")
    assert "Not written" in (tmp_path / "analysis" / "README.md").read_text()


# --------------------------------------------------------------------------
# the status bar
# --------------------------------------------------------------------------

def test_analysis_state_reads_the_workspace_not_the_page():
    from dsai.app.components import STATUS_STAGES, analysis_state
    from dsai.app.state import Workspace

    space = Workspace()
    stages = analysis_state(space)
    assert set(stages) == {name for name, _ in STATUS_STAGES}
    assert all(status == "todo" for status, _ in stages.values())


def test_analysis_state_reflects_a_finished_run(regression_run):
    from dsai.app.components import analysis_state
    from dsai.app.state import Workspace

    run, frame = regression_run
    space = Workspace()
    space.frame = frame
    space.profile = run.profile
    space.objective = run.objective
    space.run = run
    stages = analysis_state(space)
    assert stages["Model"][0] == "done"
    assert stages["Decide"][0] == "done"
    assert stages["Validate"][0] in ("done", "warn")
