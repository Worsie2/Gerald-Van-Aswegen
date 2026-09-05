"""End-to-end walkthrough of the platform.

Run it: python examples/quickstart.py

Each section is a stage of the workflow, using the same components the workspace
and the CLI use.
"""

from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from dsai.app.samples import build_sample
from dsai.core.schema import BusinessContext, Objective, TaskType
from dsai.engines.orchestrator import AIDataScientist, RunSettings
from dsai.engines.selection import tournament_frame
from dsai.reporting.exporters import to_markdown
from dsai.repro.provenance import build_manifest


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# 1. Regression, run end to end
# --------------------------------------------------------------------------
rule("1. AUTONOMOUS ANALYSIS — what drives customer spend?")

frame, source = build_sample("water_customers")
context = BusinessContext(
    description="600 customers of a South African water instrumentation company",
    business_problem="We want to grow revenue from the existing customer base",
    stated_objective="predict annual customer spend and understand what drives it",
    industry="water instrumentation",
    geography="South Africa",
    currency="ZAR",
    analysis_style="predictive",
    variable_meanings={"meters_installed": "number of flow meters at customer sites"},
    assumptions=["Next year's spending behaves like the past three years"],
    limitations=["No competitor pricing data is available"],
)

scientist = AIDataScientist()
run = scientist.analyse(frame, "water_customers", context, RunSettings(max_models=7), source)

print(run.trace.render())

rule("MODEL TOURNAMENT")
table = tournament_frame(run.tournament)
print(table[["Rank", "Model", "Composite", "R²", "RMSE", "Interpretability", "Overfit gap"]].to_string(index=False))

print(f"\nRecommended:         {run.tournament.recommended.model_name}")
print(f"Highest raw score:   {run.tournament.highest_performance.model_name}")
print(f"Simplest acceptable: {run.tournament.simplest_acceptable.model_name}")
print(f"Baseline:            {run.tournament.baseline.model_name}")

rule("WHY — the decision log")
for decision in run.decisions:
    if decision.stage == "model_recommendation":
        print(f"\n· {decision.decision}\n  {decision.reason}")
        for rejected in decision.rejected[:3]:
            print(f"  ✗ {rejected['option']}: {rejected['reason']}")

rule("FINDINGS — grouped by the evidence behind them")
from dsai.engines.insight import group_by_evidence

for heading, findings in group_by_evidence(run.findings).items():
    print(f"\n{heading.upper()}")
    for finding in findings[:3]:
        print(f"  [{finding.confidence.value}] {finding.title}")
        print(f"      {finding.detail[:150]}")

rule("RECOMMENDATIONS")
for recommendation in run.recommendations[:5]:
    print(f"\n→ [{recommendation.category} · {recommendation.confidence.value}] {recommendation.action}")
    print(f"  Why:      {recommendation.reason}")
    for item in recommendation.evidence[:2]:
        print(f"  Evidence: {item}")
    if recommendation.caveats:
        print(f"  Caveat:   {recommendation.caveats[0]}")

rule("VALIDATION CHECKS")
for check in run.self_check.checks:
    print(f"  [{'ok' if check.passed else '!!'}] {check.question}")

# --------------------------------------------------------------------------
# 2. Segmentation
# --------------------------------------------------------------------------
rule("2. SEGMENTATION — who are these customers?")

segments_frame, _ = build_sample("customer_segments")
scientist = AIDataScientist()
run = scientist.understand(segments_frame, "segments")
objective = Objective(
    task_type=TaskType.CLUSTERING,
    features=["annual_spend", "visits_per_year", "age", "tenure_months", "products_held"],
    rationale="Find natural groupings in the customer base.",
)
scientist.plan(run, objective, RunSettings(max_models=5))
scientist.execute(run)
scientist.interpret(run)

print(run.segmentation.narrative())
print("\nWhat separates them:")
for feature in run.segmentation.separating_features[:4]:
    strength = feature.get("variance_explained", feature.get("cramers_v", 0))
    print(f"  {feature['feature']:<20} {strength:.1%}")

# --------------------------------------------------------------------------
# 3. Forecasting
# --------------------------------------------------------------------------
rule("3. FORECASTING — where are sales heading?")

sales_frame, _ = build_sample("monthly_sales")
scientist = AIDataScientist()
run = scientist.understand(sales_frame, "sales")
objective = Objective(
    task_type=TaskType.TIME_SERIES_FORECAST, target="sales", time_column="month", horizon=12,
    rationale="Project sales twelve months forward.",
)
scientist.plan(run, objective, RunSettings(max_models=6))
scientist.execute(run)
scientist.interpret(run)

analysis = run.series_analysis
print(f"Trend:        {analysis['trend']['interpretation']}")
print(f"Seasonality:  {analysis['seasonality'].get('interpretation', 'none found')}")
print(f"Stationarity: {analysis['stationarity']['interpretation']}")
for issue in analysis["issues"]:
    print(f"  ! {issue}")

print(f"\nBest forecaster: {run.best.model_name}")
print(f"MASE on hold-out: {run.best.test_scores.get('mase', float('nan')):.3f}  "
      "(below 1.0 beats simply repeating the last value)")
forecast = run.best.extras["forecast"]
print("Next 6 periods:", [round(v) for v in forecast[:6]])

# --------------------------------------------------------------------------
# 4. Market basket
# --------------------------------------------------------------------------
rule("4. MARKET BASKET — what sells together?")

from dsai.engines.association import mine_associations

baskets, _ = build_sample("product_baskets")
result = mine_associations(baskets, "transaction_id", "item", min_support=0.05, min_confidence=0.35)
print(f"{result.shape.n_transactions:,} transactions, {result.shape.n_items} items, "
      f"average basket {result.shape.mean_basket_size}\n")
for rule_row in result.rules[:5]:
    print(f"· {rule_row['statement']}")
print()
for recommendation in result.recommendations[:2]:
    print(f"→ {recommendation.action}")
    print(f"  {recommendation.caveats[0]}")

# --------------------------------------------------------------------------
# 5. Asking in plain language
# --------------------------------------------------------------------------
rule("5. NATURAL LANGUAGE — the plan is shown before anything runs")

from dsai.core.profiler import profile_dataset
from dsai.engines.nl import parse_command

profile, _ = profile_dataset(frame if False else build_sample("water_customers")[0])
for question in [
    "which customers are most valuable?",
    "create four customer segments",
    "predict annual spend with random forest",
    "find unusual customers",
]:
    intent = parse_command(question, profile)
    print(f'\n"{question}"')
    print(f"  → {intent.action} / {intent.task_type.value if intent.task_type else '—'}"
          f"  (cost: {intent.estimated_cost}, needs approval: {intent.requires_confirmation})")
    for step in intent.plan_preview[:3]:
        print(f"     · {step}")

rule("DONE")
print("Explore further:")
print("  dsai app                                    the workspace UI")
print("  dsai analyse data.csv --target y --export .  full report + runnable Python")
print("  dsai models --detail random_forest_regressor  algorithm metadata")
