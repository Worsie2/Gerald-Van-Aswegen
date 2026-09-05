"""Command-line interface.

Everything the workspace does, available for scripting and CI::

    dsai profile data.csv
    dsai analyse data.csv --target annual_spend --export ./reports
    dsai ask data.csv "which customers are most valuable?"
    dsai models --task regression
    dsai app
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsai",
        description="An AI data-science platform: profile, decide, run, validate, explain, recommend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dsai profile customers.csv\n"
            "  dsai analyse customers.csv --target annual_spend --context 'SA water instrumentation customers'\n"
            "  dsai analyse sales.csv --task time_series_forecast --target sales --time-column month\n"
            "  dsai ask customers.csv 'what drives annual spend?'\n"
            "  dsai models --task classification --available\n"
            "  dsai app\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Profile a dataset and report its quality.")
    profile_parser.add_argument("path", help="Path to the dataset.")
    profile_parser.add_argument("--json", action="store_true", help="Emit the profile as JSON.")
    profile_parser.add_argument("--target", help="Column to check for leakage against.")

    analyse_parser = subparsers.add_parser("analyse", aliases=["analyze"],
                                           help="Run the full analysis end to end.")
    analyse_parser.add_argument("path")
    analyse_parser.add_argument("--target", help="Column to predict or explain.")
    analyse_parser.add_argument("--task", help="Force the analysis type (regression, clustering, …).")
    analyse_parser.add_argument("--time-column", help="Date column, for forecasting.")
    analyse_parser.add_argument("--context", default="", help="What this data represents.")
    analyse_parser.add_argument("--objective", default="", help="What you want to find out, in a sentence.")
    analyse_parser.add_argument("--models", type=int, default=8, help="How many models to compare.")
    analyse_parser.add_argument("--budget", choices=["fast", "balanced", "thorough"], default="balanced")
    analyse_parser.add_argument("--interpretability", choices=["low", "moderate", "high", "critical"],
                                default="moderate")
    analyse_parser.add_argument("--seed", type=int, default=42)
    analyse_parser.add_argument("--tune", action="store_true", help="Tune hyper-parameters (slower).")
    analyse_parser.add_argument("--export", help="Directory to write every export format into.")
    analyse_parser.add_argument("--report", choices=["both", "business", "technical"], default="both")
    analyse_parser.add_argument("--quiet", action="store_true", help="Only print the summary.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question in plain language.")
    ask_parser.add_argument("path")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--run", action="store_true",
                            help="Execute the plan instead of only showing it.")

    models_parser = subparsers.add_parser("models", help="Browse the model registry.")
    models_parser.add_argument("--task", help="Filter by task type.")
    models_parser.add_argument("--category", help="Filter by category.")
    models_parser.add_argument("--available", action="store_true", help="Only what is installed here.")
    models_parser.add_argument("--detail", help="Show full metadata for one model key.")
    models_parser.add_argument("--json", action="store_true")

    steps_parser = subparsers.add_parser("steps", help="Browse the preprocessing catalogue.")
    steps_parser.add_argument("--category")

    subparsers.add_parser("app", help="Launch the workspace UI.")

    args = parser.parse_args(argv)
    handler = {
        "profile": _profile,
        "analyse": _analyse,
        "analyze": _analyse,
        "ask": _ask,
        "models": _models,
        "steps": _steps,
        "app": _app,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _load(path: str):
    from dsai.dataio.loaders import load_dataset

    return load_dataset(path)


def _profile(args) -> int:
    from dsai.core.profiler import profile_dataset

    frame, source = _load(args.path)
    profile, _ = profile_dataset(frame, name=Path(args.path).name, target=args.target)

    if args.json:
        print(profile.to_json())
        return 0

    print(f"\n{profile.name}: {profile.n_rows:,} rows × {profile.n_columns} columns "
          f"({profile.memory_mb:.1f} MB)")
    print(f"Data quality score: {profile.quality_score}/100\n")

    print(f"{'COLUMN':<28} {'DETECTED AS':<28} {'MISSING':>8} {'DISTINCT':>9}")
    print("-" * 76)
    for name, column in profile.columns.items():
        print(f"{name[:27]:<28} {column.semantic_type.value:<28} "
              f"{column.missing_pct:>7.1f}% {column.n_unique:>9,}")

    if profile.quality_issues:
        print("\nDATA QUALITY")
        for issue in sorted(profile.quality_issues,
                            key=lambda i: {"critical": 0, "warning": 1, "info": 2}.get(i.severity, 3)):
            marker = {"critical": "!!", "warning": " !", "info": "  "}.get(issue.severity, "  ")
            print(f"{marker} {issue.message}")
            if issue.suggested_action:
                print(f"      → {issue.suggested_action}")

    if profile.target_candidates:
        print("\nPOSSIBLE TARGET VARIABLES")
        for candidate in profile.target_candidates[:5]:
            print(f"   {candidate['column']:<24} {candidate['score']:.2f}  "
                  f"({candidate['implied_task'].replace('_', ' ')}) — {'; '.join(candidate['reasons'][:2])}")

    if profile.leakage_suspects:
        print("\nPOSSIBLE LEAKAGE")
        for suspect in profile.leakage_suspects:
            print(f" ! {suspect['column']}: {suspect['reason']}")
    print()
    return 0


def _analyse(args) -> int:
    from dsai.core.schema import BusinessContext, Objective, TaskType
    from dsai.engines.orchestrator import AIDataScientist, RunSettings

    frame, source = _load(args.path)
    context = BusinessContext(
        description=args.context,
        stated_objective=args.objective,
        target_variable=args.target,
    )
    settings = RunSettings(
        mode="automatic",
        max_models=args.models,
        time_budget=args.budget,
        interpretability_need=args.interpretability,
        random_state=args.seed,
        tune_hyperparameters=args.tune,
    )

    objective = None
    if args.task:
        try:
            task = TaskType(args.task)
        except ValueError:
            print(f"Unknown task '{args.task}'. Options: {', '.join(t.value for t in TaskType)}",
                  file=sys.stderr)
            return 2
        objective = Objective(
            task_type=task, target=args.target, time_column=args.time_column,
            rationale="You specified this on the command line.", source="user_override", priority=1.0,
        )

    def on_event(event):
        if args.quiet:
            return
        icons = {"done": "✓", "running": "…", "warning": "!", "failed": "✗", "skipped": "–"}
        timing = f"  ({event.elapsed_s:.2f}s)" if event.elapsed_s else ""
        detail = f" — {event.detail}" if event.detail else ""
        print(f"{icons.get(event.status, '·')} {event.step}{detail}{timing}")

    scientist = AIDataScientist(trace_callback=None if args.quiet else on_event)
    run = scientist.analyse(frame, Path(args.path).stem, context, settings, source, objective)

    print("\n" + "=" * 78)
    print(run.summary())
    print("=" * 78)

    if run.tournament and run.tournament.table:
        from dsai.engines.selection import tournament_frame

        print("\nMODEL TOURNAMENT")
        print(tournament_frame(run.tournament).to_string(index=False))
        for warning in run.tournament.warnings:
            print(f"\n! {warning}")

    if run.findings:
        print("\nKEY FINDINGS")
        for finding in run.findings[:8]:
            print(f"\n[{finding.kind.value} · {finding.confidence.value}] {finding.title}")
            print(f"   {finding.detail}")

    if run.recommendations:
        print("\nRECOMMENDATIONS")
        for recommendation in run.recommendations[:8]:
            print(f"\n[{recommendation.category} · {recommendation.confidence.value}] {recommendation.action}")
            print(f"   Why: {recommendation.reason}")
            for item in recommendation.evidence[:2]:
                print(f"   - {item}")

    if run.self_check:
        failed = [c for c in run.self_check.checks if not c.passed]
        if failed:
            print("\nVALIDATION CHECKS THAT RAISED SOMETHING")
            for check in failed:
                print(f" ! {check.question}\n   {check.detail}")

    if args.export:
        from dsai.reporting.exporters import export_all

        written = export_all(run, args.export, args.path)
        print("\nEXPORTED")
        for label, target in written.items():
            print(f"   {label:18} {target}")
    print()
    return 0


def _ask(args) -> int:
    from dsai.core.profiler import profile_dataset
    from dsai.core.schema import BusinessContext
    from dsai.engines.nl import answer_question, intent_to_objective, parse_command
    from dsai.engines.orchestrator import AIDataScientist, RunSettings
    from dsai.registry.base import load_builtin_models

    load_builtin_models()
    frame, source = _load(args.path)
    profile, typed = profile_dataset(frame, name=Path(args.path).name)

    intent = parse_command(args.question, profile)
    print(f'\nQuestion: "{args.question}"\n')
    print(intent.describe().replace("**", "").replace("`", ""))

    if intent.action in {"describe", "correlate", "test", "rank"} and not intent.clarification:
        result = answer_question(args.question, typed, profile)
        if result.get("answer"):
            print("\n" + result["answer"].replace("**", ""))
        return 0

    if not args.run:
        print("\n(Add --run to execute this plan.)")
        return 0

    objective = intent_to_objective(intent, profile)
    if objective is None:
        print("\nNothing to run — the question needs to be more specific.", file=sys.stderr)
        return 2

    scientist = AIDataScientist()
    run = scientist.analyse(
        frame, Path(args.path).stem, BusinessContext(), RunSettings(mode="automatic"),
        source, objective,
    )
    print("\n" + run.summary())
    for recommendation in run.recommendations[:5]:
        print(f"\n→ {recommendation.action}\n  {recommendation.reason}")
    return 0


def _models(args) -> int:
    from dsai.core.schema import TaskType
    from dsai.registry.base import REGISTRY, load_builtin_models

    load_builtin_models()

    if args.detail:
        spec = REGISTRY.get(args.detail)
        print(f"\n{spec.name}  [{spec.key}]")
        print(f"  Category:          {spec.category} / {spec.family}")
        print(f"  Tasks:             {', '.join(t.value for t in spec.task_types)}")
        print(f"  Interpretability:  {spec.interpretability.value}")
        print(f"  Training cost:     {spec.cost.value}")
        print(f"  Needs scaling:     {spec.requires_scaling}")
        print(f"  Handles missing:   {spec.handles_missing}")
        print(f"  Minimum rows:      {spec.min_rows}")
        print(f"  Available here:    {spec.is_available()}")
        for label, items in [("Advantages", spec.advantages), ("Limitations", spec.limitations),
                             ("Assumptions", spec.assumptions), ("Good for", spec.good_for)]:
            if items:
                print(f"\n  {label}:")
                for item in items:
                    print(f"    - {item}")
        if spec.hyperparameters:
            print("\n  Hyper-parameters:")
            for param in spec.hyperparameters:
                bounds = (f"{param.low} – {param.high}" if param.low is not None
                          else ", ".join(map(str, param.choices)) or "—")
                print(f"    {param.name:<22} default={param.default!s:<14} range={bounds}")
                if param.description:
                    print(f"      {param.description}")
        print()
        return 0

    task = None
    if args.task:
        matches = [t for t in TaskType if args.task in t.value]
        task = matches[0] if matches else None
    specs = REGISTRY.find(task_type=task, category=args.category, only_available=args.available)

    if args.json:
        print(json.dumps([s.to_dict() for s in specs], indent=2, default=str))
        return 0

    stats = REGISTRY.stats()
    print(f"\n{stats['total']} algorithms registered, {stats['available']} available here.")
    print(f"Showing {len(specs)}.\n")
    print(f"{'KEY':<34} {'NAME':<40} {'INTERP':<13} {'COST':<10} OK")
    print("-" * 106)
    for spec in sorted(specs, key=lambda s: (s.category, s.name)):
        print(f"{spec.key[:33]:<34} {spec.name[:39]:<40} {spec.interpretability.value:<13} "
              f"{spec.cost.value:<10} {'yes' if spec.is_available() else 'no'}")
    print("\nUse --detail <key> for full metadata.\n")
    return 0


def _steps(args) -> int:
    from dsai.preprocessing.steps import STEPS

    steps = STEPS.by_category(args.category) if args.category else STEPS.all()
    print(f"\n{len(STEPS)} preprocessing steps. Showing {len(steps)}.\n")
    print(f"{'KEY':<30} {'CATEGORY':<20} {'SCOPE':<8} {'SAFE':<6} NAME")
    print("-" * 100)
    for step in sorted(steps, key=lambda s: (s.category, s.name)):
        print(f"{step.key[:29]:<30} {step.category:<20} {step.scope:<8} "
              f"{'yes' if step.leakage_safe else 'fold':<6} {step.name}")
    print("\n'SAFE' = can be applied before splitting. 'fold' = fitted inside each "
          "cross-validation fold to prevent leakage.\n")
    return 0


def _app(args) -> int:
    import subprocess

    entry = Path(__file__).parent / "app" / "main.py"
    print(f"Launching the workspace: streamlit run {entry}")
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(entry)])
    except FileNotFoundError:
        print("Streamlit is not installed. Run: pip install 'dsai[app]'", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
