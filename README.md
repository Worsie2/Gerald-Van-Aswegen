# DSAI — an AI data scientist, not a chatbot about one

A modular data-science platform that takes a dataset and runs the whole analysis:
profile it, work out what question it can answer, build preprocessing suited to each
candidate model, train and cross-validate a field of them, compare them on more than the
headline score, explain the winner, check its own conclusions, and say what to do — with
the evidence attached to every claim.

The AI is not an interface on top of an analytics library. It is the layer that decides
what to run, runs it, and judges the result.

```
UPLOAD → UNDERSTAND → CONTEXT → QUALITY → PREPROCESS → CHOOSE METHODS
      → RUN MODELS → VALIDATE → COMPARE → EXPLAIN → FINDINGS → RECOMMENDATIONS
```

---

## What it actually does

| | |
|---|---|
| **111 algorithms** | across regression, classification, clustering, dimensionality reduction, time series, anomaly detection and association mining — registered as metadata, not hard-coded |
| **41 preprocessing steps** | with leakage prevention built into the architecture, not bolted on |
| **15+ statistical tests** | each reporting assumptions, effect size and whether the difference matters in practice |
| **6 export formats** | Markdown, HTML, Excel, JSON, runnable Python, PDF |
| **3 ways to work** | AI automatic, AI assisted, manual/expert — one engine underneath |

```bash
pip install -e ".[full]"

dsai profile customers.csv
dsai analyse customers.csv --target annual_spend --export ./reports
dsai ask customers.csv "which customers are most valuable?"
dsai app                     # the workspace UI
```

```python
from dsai.core.schema import BusinessContext
from dsai.engines.orchestrator import AIDataScientist, RunSettings

run = AIDataScientist().analyse(
    frame,
    name="customers",
    context=BusinessContext(
        description="600 customers of a water instrumentation company",
        business_problem="grow revenue from the existing base",
        stated_objective="predict annual spend and understand what drives it",
        currency="ZAR",
    ),
    settings=RunSettings(max_models=8, interpretability_need="high"),
)

print(run.summary())
print(run.trace.render())
for recommendation in run.recommendations:
    print(recommendation.action, "—", recommendation.reason)
```

---

## The five commitments

These are the design decisions that shaped everything else. They are also what the test
suite checks hardest.

### 1. It never claims a model is best in general

The tournament reports the *best-performing model for this dataset under the validation
strategy used*, and gives four answers rather than one:

| | |
|---|---|
| **Recommended** | best once generalisation, fold stability, interpretability and training cost are weighed together |
| **Highest score** | best on the headline metric alone — often not the same model |
| **Simplest acceptable** | within 5% of the best, and far easier to defend |
| **Baseline** | ignores every predictor; the number any real model must beat |

A model that scores highest while overfitting badly does not win on that basis. The
weighting is stated on screen, not buried in a formula.

### 2. Preprocessing cannot leak

Steps are split into two kinds. Row operations (deduplication, row removal) change which
observations exist and run before splitting. Everything that *learns* a statistic —
means, scales, category-to-target maps, PCA loadings, feature selection — is compiled into
an unfitted scikit-learn pipeline and refitted on each training fold. Target encoding, the
easiest way to leak a target, is marked `leakage_safe=False` and can only be fitted inside
the fold.

`pipeline.leakage_report()` states exactly which steps are handled which way.

### 3. Assumption, measurement and inference stay apart

Every finding carries an `EvidenceKind`:

- **observed in the data** — measured
- **statistical test** — established, with the test named
- **model-derived** — inferred by a fitted model
- **AI interpretation** — the platform's reading
- **user assumption** — what you told it, never verified

The report groups findings under those headings. What you asserted in the Context page is
reported as your assumption, and is never presented as a measurement.

### 4. Correlation is never quietly upgraded to cause

Every model-derived recommendation carries the caveat explicitly, and the self-check
raises it on every observational dataset — which is all of them. Recommendations state
what an intervention would assume, and say that only a controlled test can settle it.

### 5. A weak result is reported as weak

If nothing beats the baseline, the platform says so and blocks the recommendation rather
than presenting the least-bad option. Eleven validation checks run before anything is
shown, each either lowering confidence or attaching a caveat:

sample size · rows per variable · leakage · class balance · overfitting · fold stability ·
does it beat the baseline · outlier influence · data quality · causality · temporal validity

---

## The engines

```
dsai/
├── core/          schema, dataset profiler
├── dataio/        CSV, Excel, JSON, Parquet, Feather, Stata, SPSS, SAS, SQL
├── registry/      the model registry — 111 algorithms as metadata
├── preprocessing/ 41 steps, editable pipelines, the recommender
├── engines/       objective · decision · experiment · selection · insight
│                  recommendation · self-check · orchestrator · natural language
├── explain/       importance, partial dependence, local explanation, diagnostics
├── statistics/    descriptive, hypothesis tests, time-series structure
├── viz/           chart selection and rendering
├── reporting/     report builder and exporters
├── repro/         run manifests, code generation, project persistence
└── app/           the 11-page workspace
```

**Data understanding** — semantic type inference (identifier, datetime, ordinal, binary,
text, high-cardinality), missingness, duplicates, outliers, skew, correlation, VIF-based
multicollinearity, target-candidate scoring, leakage suspects and a severity-ranked
quality report. Every inference is shown and can be corrected.

**Decision engine** — queries the registry with the constraints the data imposes, then
scores survivors on sample size, dimensionality, missingness, nonlinearity evidence,
collinearity, outliers, class imbalance, interpretability need, compute budget, and
whether the target actually satisfies each model's link function. The shortlist spans
families deliberately: if a simple model matches a complex one, that is itself the finding.

**Experiment engine** — trains, cross-validates (k-fold, stratified, repeated,
time-series, rolling-origin back-tests for forecasters), scores on a hold-out set,
optionally tunes, and flags overfitting, fold instability, useless clusterings,
implausible anomaly rates and forecasts that fail to beat naive.

**Explanation** — coefficients with inference where the model provides them, built-in and
permutation importance, partial dependence with the shape of the curve described, and
per-prediction attribution (SHAP where installed, an occlusion approximation otherwise —
the method used is always named, never implied).

---

## Reproducibility

Every run records its dataset fingerprint, context, objective, preprocessing steps,
validation strategy, random seed, every model evaluated, the one selected, its
hyper-parameters, metrics, the platform's decisions, your overrides and library versions.

The strongest guarantee is the exported Python: a standalone pandas and scikit-learn
script with the custom transformers inlined, which reproduces the analysis without this
platform installed. A test runs it in a subprocess and checks the output.

```bash
dsai analyse data.csv --target spend --export ./reports
# → reports/data_<id>_reproduce.py   runs anywhere pandas + sklearn are installed
```

---

## Extending it

Nothing hard-codes a model list. Adding an algorithm is one registration:

```python
from dsai.core.schema import TaskType
from dsai.registry._helpers import hp, lazy
from dsai.registry.base import Cost, Interpretability, ModelSpec, register

register(ModelSpec(
    key="my_regressor",
    name="My Regressor",
    category="regression",
    family="ensemble",
    task_types=[TaskType.REGRESSION],
    builder=lazy("my_package.models:MyRegressor"),
    module="my_package",
    requires_scaling=False,
    nonlinear=True,
    interpretability=Interpretability.MODERATE,
    cost=Cost.MEDIUM,
    assumptions=["..."],
    advantages=["..."],
    limitations=["..."],          # required — every model has some
    hyperparameters=[hp("depth", "int", 6, 2, 20, description="Tree depth.")],
    metrics=["r2", "rmse", "mae"],
))
```

The decision engine considers it from that point on, filtered by the same metadata as
everything else. A separate package can ship a whole pack through an entry point:

```toml
[project.entry-points."dsai.models"]
my_pack = "my_package.models:register_all"
```

Preprocessing steps extend the same way through `dsai.preprocessing.steps.STEPS`.

---

## The workspace

```bash
dsai app          # or: streamlit run dsai/app/main.py
```

**Data** → **Context** → **Preprocessing** → **Analysis** → **Models** → **Insights** →
**Recommendations**, plus **Ask** (plain-language questions that show the plan before
running), **Report** (both audiences, every export), **Projects** (saved workspaces and
run history) and **Model Library** (the full registry, browsable).

Six synthetic sample datasets with deliberately known structure are built in, so the
platform's answers can be checked against a ground truth you already know.

---

## Installation

```bash
pip install -e ".[full]"     # everything
pip install -e .             # core only: pandas, numpy, scikit-learn, scipy
```

Optional extras are degraded gracefully — a missing library removes its algorithms from
the registry and says so, rather than failing at import.

| Extra | Adds |
|---|---|
| `stats` | statsmodels — ARIMA/SARIMA, state-space, OLS inference, STL |
| `boosting` | XGBoost, LightGBM, CatBoost |
| `io` | Excel, Parquet, SQL |
| `mining` | mlxtend — Apriori, FP-Growth |
| `app` | Streamlit workspace |
| `viz` | Plotly charts |
| `manifold` | UMAP |
| `explain` | exact SHAP |

```bash
pytest                       # 95 tests
```

---

## What it will not do

- Claim a model is best in general
- Present a correlation as a cause
- Hide a result that fails to beat doing nothing
- Let preprocessing see the rows it will be scored on
- Mix your assumptions with its measurements
- Draw a chart that only restates the summary table

---

## Honest limits

- **Scale.** Designed for datasets that fit in memory. Above a few million rows, sample
  first or push the aggregation into your database.
- **Causal inference.** It detects and reports associations, and says so. There is no
  do-calculus, no instrumental variables, no difference-in-differences.
- **Deep learning.** Neural support is scikit-learn's MLP. No transformers, no images, no
  sequence models — the platform is built for structured tabular data.
- **The "AI" is deterministic.** Objective detection and model selection are rule-based
  over measured properties, not an LLM. That is a deliberate choice: the same dataset
  always produces the same plan, the reasoning is auditable, and it works with no API key.
  Where an LLM would help is narration; where it would hurt is deciding what is true.
- **Text.** Text columns get length, word-count and vocabulary features. For semantics,
  add TF-IDF plus SVD deliberately.

---

MIT licensed.
