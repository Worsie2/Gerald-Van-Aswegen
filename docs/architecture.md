# Architecture

## The shape of it

Six engines, each with one job, exchanging plain dataclasses:

```
Data Engine  ──▶  Decision Engine  ──▶  Experiment Engine
    │                    │                      │
    │                    ▼                      ▼
    │              Model Registry        Selection Engine
    ▼                                           │
Preprocessing  ◀────────────────────────────────┤
                                                ▼
                              Insight ──▶ Self-Check ──▶ Recommendation
```

Every object exchanged is a dataclass with `to_dict()` / `to_json()`, so a run serialises
whole, diffs between experiments, and renders through any front end.

## Why the pieces are separate

**The registry is metadata, not code.** `ModelSpec` describes what an algorithm needs,
what it assumes, what it costs and what it is bad at. The decision engine queries that
metadata. Adding an algorithm therefore never touches the decision logic — which is the
only way "hundreds of models" is maintainable rather than a switch statement.

**Builders are lazy.** A spec for an uninstalled library sits in the registry harmlessly
and reports `is_available() == False`. Nothing imports until a model is actually selected.

**Preprocessing is split by scope.** Row operations change which observations exist and run
before splitting. Column operations are fitted transformations compiled into an unfitted
scikit-learn pipeline handed to cross-validation. That split is the leakage guarantee, and
it is structural rather than a convention someone has to remember.

**DataFrames all the way through.** `FrameTransformer` wraps every step so column names
survive the pipeline. Chained `ColumnTransformer`s lose them, and once names are gone every
downstream explanation becomes guesswork about `feature_37`.

## Core data structures

| Type | Holds |
|---|---|
| `DatasetProfile` | per-column semantic types, quality issues, correlations, VIF, target candidates, leakage suspects |
| `BusinessContext` | what the *user* asserts — kept separate from everything measured |
| `Objective` | task type, target, features, horizon, k, rationale, and whether it came from the user or the data |
| `AnalysisPlan` | candidates with reasons, validation strategy, primary metric, warnings |
| `ExperimentResult` | scores, folds, timing, hyper-parameters, warnings, overfitting gap |
| `TournamentOutcome` | ranked models, four named choices, decisions, weighting |
| `Finding` | a claim, its evidence kind, its confidence and its caveats |
| `Recommendation` | an action, its reason, its evidence, and what it traces back to |
| `Decision` | one auditable entry: decided, reason, evidence, rejected alternatives |
| `RunManifest` | everything needed to reproduce the run, with a fingerprint |

## The decision engine

Scoring is additive over measured properties, starting at 0.5:

```
sample size        tiny → simple models up, ensembles down
dimensionality     wide → high-dimensional handlers up, distance methods down
missingness        native handling up
nonlinearity       Spearman-vs-Pearson gap → nonlinear models up, linear down
collinearity       VIF ≥ 10 → regularised and tree models up, plain linear down
outliers           robust models up, squared-loss models down
class imbalance    class_weight support up
target validity    Gamma/Poisson penalised when the target breaks the link function
interpretability   scaled by stated need
compute budget     cost above budget penalised
```

Then `_diversify` caps how many models come from one family, and always keeps a baseline in
the field. A tournament of five gradient-boosting variants tells you nothing about whether
the problem needs complexity at all.

## The tournament

```
composite = 0.50 · performance
          + 0.20 · generalisation   (train-to-held-out gap)
          + 0.15 · stability        (score variance across folds)
          + 0.10 · interpretability
          + 0.05 · efficiency       (training time)
```

Bounded metrics (R², AUC, F1) map to 0–1 directly. Unbounded ones (RMSE, MAE) are
normalised across the field, because they have no absolute scale — the best model in the
tournament scores 1, the worst 0. Getting this wrong produced NaN composites and silently
mis-ranked the whole table; there is a regression test for it.

## The self-check

Eleven questions, run before anything is presented. Each either passes, downgrades
confidence, or blocks:

| Check | Effect when it fails |
|---|---|
| Enough rows? | −1, or −2 below 30; blocking below 20 |
| Enough rows per variable? | −1 |
| Could this be leakage? | −2; blocking when a near-perfect score coincides with a suspect column |
| Is the target balanced enough? | −1 |
| Is it overfitting? | −1, or −2 above a 0.3 gap |
| Is it stable across folds? | −1 |
| Does it beat the baseline? | **blocking** |
| Could outliers be driving it? | −1 above 10% |
| Unresolved quality problems? | −1 |
| Causal or associational? | caveat, always |
| Was time respected? | caveat |

Downgrades cap at two levels. Three always lands on "speculative" regardless of where a
finding started, which destroys the gradient the levels exist for.

## Extension points

| To add | Register into |
|---|---|
| An algorithm | `dsai.registry.base.REGISTRY` |
| A preprocessing step | `dsai.preprocessing.steps.STEPS` |
| A chart | `dsai.viz.recommender` + `dsai.viz.plots.render` |
| A statistical test | `dsai.statistics.tests` |
| An export format | `dsai.reporting.exporters` |
| A model pack | the `dsai.models` entry point |

Adding an algorithm requires no change to the decision engine, the experiment engine, the
tournament, the report or the UI. That is the test of whether the architecture holds.
