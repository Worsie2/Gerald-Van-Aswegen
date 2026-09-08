# DSAI — what the application does

A complete functional specification, written to be handed to a reviewer. Every
number in it was read out of the running code rather than remembered.

**Purpose of this document.** To let someone who has not seen the app judge
whether the functionality is complete, whether the interface exposes it well,
and what is missing. Section 14 lists what the author already knows is absent —
a reviewer should treat that as a starting point, not a boundary.

**What it is.** A desktop data-science platform: a Streamlit application plus a
command-line interface over a Python package. It takes a dataset, works out what
question the data can answer, builds preprocessing suited to each candidate
model, trains and cross-validates a field of them, compares them on more than the
headline score, explains the winner, checks its own conclusions, says what to do,
and then lets the resulting model score new data.

Roughly 33,800 lines of Python across 94 modules, with 290 tests.

Stack: Python 3.11, pandas, scikit-learn, scipy, statsmodels, XGBoost, LightGBM,
mlxtend, Plotly, Streamlit. Runs entirely locally; no data leaves the machine and
no LLM is called at runtime.

---

## 1. The design premises

These shape every decision below. A reviewer disagreeing with one of these is
giving more useful feedback than one disagreeing with a button placement.

1. **The AI recommends; the user decides.** Every automated choice is visible and
   overrideable. Nothing is applied silently.
2. **Facts and assumptions are kept apart.** What was measured in the data, what
   a statistical test established, what a model inferred, what the platform
   interpreted, and what the user asserted are five distinct kinds of claim, are
   labelled as such, and are never blended.
3. **No model is ever called "best" in general.** The wording throughout is
   "best-performing on this dataset under the validation strategy used".
4. **Leakage is prevented structurally, not by discipline.** Operations that
   change which rows exist run before the split; operations that learn anything
   from the data are compiled into an unfitted scikit-learn pipeline and refitted
   inside every cross-validation fold. It is not possible to configure a leaking
   pipeline through the interface.
5. **Caveats travel with claims.** A finding and its limitation are one object;
   the limitation cannot be dropped when the finding is exported.
6. **No exposed chain of thought.** The platform reports decisions, reasons and
   the options it rejected — not a narration of its own reasoning.
7. **Charts are never the only way to read something.** Every chart in the app
   and in every export carries a written caption stating what it shows and a
   table of the numbers behind it.

---

## 2. Getting data in

**Formats.** CSV, TSV, TXT, Excel (.xlsx/.xlsm/.xls, with a sheet picker when a
workbook has several), JSON, JSONL, Parquet, Feather. Encoding and delimiter are
detected.

**Databases.** Any SQLAlchemy connection string. Read-only by construction: only
`SELECT`, `WITH`, `SHOW`, `PRAGMA` and `DESCRIBE` are accepted and anything else
raises. Connection strings are credential-stripped before being stored in any
record or manifest.

**Sample datasets.** Six synthetic datasets with deliberately planted structure,
so the platform's answers can be checked against a ground truth the user already
knows: water customers (regression, a +R60,000 premium-tier effect), customer
churn (classification), customer segments (clustering, three planted groups),
monthly sales (seasonal time series), product baskets (association rules), messy
survey (deliberately dirty — duplicates, missing values, mixed types, an
identifier column, a leaking column).

**Large files.** A size warning before reading. Above 250,000 rows the platform
offers a 100,000-row random sample (random rather than the first N: files are
often sorted, and the first hundred thousand rows of a date-sorted file are a
different population). `MemoryError` produces a plain message and a suggested
next step, not a stack trace.

---

## 3. Understanding the data (the "dataset intelligence report")

Runs automatically on load.

**Per column.** Semantic type inferred from the values across eleven categories —
`numeric_continuous`, `numeric_discrete`, `binary`, `categorical_nominal`,
`categorical_ordinal`, `high_cardinality_categorical`, `datetime`, `text`,
`identifier`, `constant`, `unknown`. Plus dtype, missing count and share,
distinct count, dominant value, mean, median, standard deviation, min, max,
skewness, kurtosis and outlier share.

**Type corrections.** The user can override any inferred type. The correction is
stored on the workspace, not only on the profile, and is re-applied whenever the
dataset is re-profiled — so re-profiling cannot silently overrule the user.

**Per dataset.** Row and column counts, duplicate rows, total missing cells,
correlation pairs, multicollinearity (VIF), and a 0–100 quality score.

**Quality issues.** Fourteen detectors: `small_sample`, `wide_data`,
`duplicate_rows`, `severe_missingness`, `missing_values`, `constant_columns`,
`near_constant_columns`, `outliers`, `skewed_distributions`, `high_correlation`,
`multicollinearity`, `high_cardinality`, `identifier_columns`,
`possible_leakage`. Each carries a severity, the columns involved, and a
suggested action.

The single quality score is also broken into six dimensions a reader can act on —
completeness, uniqueness, consistency, distribution, independence, sufficiency —
each with the specific issue driving it.

**Target candidates.** Columns that look like an outcome, ranked by how strongly
they resemble one, with the reasons. Suggestion only.

**Leakage suspects.** Columns that would let a model see the answer, each with the
reason it was flagged.

**Variable inspector.** One column at a time: its facts, its distribution as a
chart, and what preprocessing would do about it and why (a column 43% missing, a
skew of 3.34, an identifier that one-hot encoding would turn into a column per
row).

---

## 4. Business context (kept separate from measurement)

A form for what the user knows and the data cannot say: description, data source,
business problem, research question, desired outcome, industry, time period,
geography, currency (defaults to ZAR), stated objective, per-variable meanings,
variables to exclude, assumptions, and known limitations.

Everything entered here is marked `user_assumption` wherever it appears, is
reported separately from measurements, and is listed in the report as something
every conclusion inherits.

---

## 5. Objective detection

The platform proposes what the data can answer, ranked. Two sources feed in and
are kept apart: what the user said (authoritative on intent, often vague) and
what the data supports (measured, blind to intent). Where they disagree, the user
wins on intent and the data wins on feasibility — and the disagreement is
surfaced rather than resolved silently.

Ten task types: regression, binary classification, multiclass classification,
clustering, dimensionality reduction, time-series forecasting, anomaly detection,
association rules, hypothesis testing, exploratory.

The user can pick any suggestion or define their own — task type, target,
predictors, time column, cluster count, forecast horizon, and basket layout for
association mining.

---

## 6. Preprocessing

**42 steps across nine categories.**

| Category | Steps |
|---|---|
| Cleaning | drop duplicates, drop rows by missingness, drop columns, drop constant columns, coerce to numeric |
| Imputation | mean, median, most frequent, constant, **within a group**, KNN, iterative (MICE), forward/backward fill |
| Outliers | winsorize, clip to IQR, remove outlier rows |
| Scaling | standard, min-max, robust, max-abs, row normalisation |
| Transformation | log, Yeo-Johnson, Box-Cox, quantile-to-normal, binning, binarise |
| Encoding | one-hot, ordinal, frequency, target, group rare categories |
| Feature engineering | date/time features (incl. cyclical), text features, polynomial, interactions, aggregations |
| Feature selection | k-best (regression), k-best (classification), mutual information, drop correlated |
| Dimensionality | PCA (components named `component_1…n`, not `pca0`) |

**Leakage control.** Each step is classified `row` scope (changes which rows
exist, applied before the split) or `column` scope (learns something, compiled
into an unfitted pipeline and refitted per fold). The interface states which
applies to every step and shows the split as a report.

**Missing-values workbench.** Every column with gaps gets its own decision, not
one strategy applied to a multiselect. Beside each choice is the value it would
actually insert — "the single value 41.45 in every gap", or a table of the
per-group values with how many rows each rests on and a warning where a group has
fewer than five. Identical choices merge into one pipeline step.

**Group-wise imputation.** Fills a gap with the statistic of the row's own group —
the mean spend of its *region* rather than of everyone. Learns per-group
statistics from training rows only and falls back to the overall training
statistic for a group seen only in the test fold.

**Multiple pipelines.** Several can be held at once, duplicated, deleted and
compared side by side (correlation heatmaps, distributions, VIF, column counts).

**Other views.** A recommended pipeline built for a chosen model or general
purpose at three depths; a drawn pipeline graph (raw data → steps → model, each
node labelled with where it is fitted); a step editor with reorder, disable,
remove, undo and redo; a step-by-step preview of the effect; a before/after
comparison; and export of the cleaned dataset as CSV or Parquet.

---

## 7. Model selection and the tournament

**Registry: 112 algorithms, 109 available in a standard install.**

| Category | Count |
|---|---|
| Regression | 31 |
| Classification | 28 |
| Time series | 17 |
| Clustering | 14 |
| Dimensionality reduction | 11 |
| Anomaly detection | 8 |
| Association | 2 |

Nothing hard-codes a list of models. Each algorithm is described by metadata —
task types, family, interpretability, cost, minimum rows, whether it handles
missing values, whether it needs scaling, whether it captures non-linearity,
advantages, limitations, assumptions, what it is good for. The decision engine
queries that metadata against the measured characteristics of the dataset. A
third-party model pack can register itself through the `dsai.models` entry point
without touching the codebase.

**Planning.** Before anything runs, the platform shows: the shortlist with a fit
score and the reason each was chosen, the metric the comparison will be ranked
on, the validation strategy and why that one, the number of folds, and any
warnings. Presented as a ranked list (recommended / strong alternative /
baseline / has a caveat) and as a table.

**Validation.** Strategy chosen from the data — k-fold, stratified k-fold,
time-series split, or a hold-out where cross-validation is not defensible — with
the reason stated. Every model gets the same split, the same folds and the same
seed.

**Composite ranking.** Not the raw score. Each model's figures are normalised
against the field, then combined: performance 50%, generalisation 20%, stability
15%, interpretability 10%, training cost 5%. A model that wins on the headline
metric can rank below one marginally worse and far steadier.

**Four named answers, not one verdict.** Recommended, highest-performance,
simplest-acceptable, and baseline — because "best" depends on whether the reader
needs accuracy, explainability, or simply not to be beaten by doing nothing.

**Three modes** over one engine: AI automatic, AI assisted, manual/expert. Only
how much the user decides changes.

---

## 8. Explanation and diagnostics

- Global feature importance — model-native, impurity-based, permutation, or
   coefficient magnitude depending on the model — with the method named on the
   output and its caveats attached.
- Coefficients with a plain-language reading of each.
- Partial dependence curves.
- Per-row explanation: why *this* prediction, for a chosen row. Uses exact
  Shapley attribution where `shap` is installed, and a simpler
  contribution decomposition where it is not, saying which was used.
- Regression diagnostics: predicted-against-actual, residuals, residual structure.
- Classification diagnostics: confusion matrix, per-class metrics, ROC,
  precision-recall, probability calibration.
- Learning curve: would more data help?
- Cluster profiling: what distinguishes each segment, segment sizes, a PCA
  projection, and a k-sweep comparing elbow, silhouette, Calinski-Harabasz and
  Davies-Bouldin — reporting agreement or disagreement rather than picking one.
- Time-series analysis: trend, seasonality (detrended first, so a strong trend
  cannot mask it), stationarity, decomposition, structural breaks, gaps,
  autocorrelation.

---

## 9. Self-check, findings and recommendations

**Eleven validation questions** run against every analysis, each of which can
downgrade the confidence of the conclusions or block them:

1. Is the dataset large enough to support a conclusion?
2. Are there enough rows per variable?
3. Could the model be seeing the answer (leakage)?
4. Is the target balanced enough to model?
5. Is the model overfitting?
6. Is the result stable across folds?
7. Does the model beat a naive baseline?
8. Could a handful of extreme values be driving this?
9. Are there unresolved data-quality problems?
10. Is the relationship causal, or only associational?
11. Was time respected in the validation split?

**Findings** are grouped by the kind of evidence behind them — observed in the
data, established by a statistical test, derived from a model, platform
interpretation, user assumption — each with a confidence level (high, moderate,
low, speculative), its evidence, and its caveats. Presented as a numbered report
or as cards.

**Recommendations** are grouped as risks to address first, data quality, actions
to take, and what to investigate next. Each names the analytical output it rests
on, its expected impact, and its confidence. A blanket limitation is attached to
all of them: this is observational data, every relationship is an association,
and acting on one assumes it survives intervention.

**Decision log.** Every automated decision with its reason, its evidence, the
options rejected and why, its confidence, and whether the user overrode it.

---

## 10. Statistics (run directly, outside the modelling flow)

Descriptive: numeric and categorical summaries, percentiles, group summaries,
outlier tables, correlation matrix and pairs, covariance, VIF.

Tests: normality, homoscedasticity, t-test, Mann-Whitney, Wilcoxon signed-rank
(paired), ANOVA, Kruskal-Wallis, chi-square, Fisher exact, correlation test, and
multiple-comparison correction.

Every result reports the assumptions it checked, an effect size with an
interpretation, and whether the difference is large enough to matter in practice
— separately from whether it is statistically significant. The interface makes
the reader choose between independent-groups and paired designs explicitly,
because running the wrong one is the classic way to get a confident wrong answer.

Time series: trend, stationarity (ADF and KPSS), seasonality, decomposition,
structural breaks, gaps, autocorrelation.

---

## 10a. Challenging the result

Everything in this section exists to attack the analysis rather than produce it.

**Sensitivity analysis.** Re-runs the analysis across up to 20 alternative
specifications — a different model family, minimal or thorough preprocessing,
mean or KNN imputation instead of median, winsorised or untouched outliers, a
different seed, a different fold count, and without whichever predictor is most
questionable — and reports whether the same variable still comes out strongest.
A conclusion that survives all of them is worth acting on; one that flips when
the outlier treatment changes is a finding about the outlier treatment, and the
report says which lever moved it.

**Subgroup discovery.** Finds where prediction error is materially different,
across single conditions and pairs of conditions. Two safeguards: nothing under
30 rows is reported, and a group is flagged only when it stands beyond three
standard errors — not merely above average, which half of all groups are. The
wording is always a measured difference, never bias or a fairness failure.

**Prediction intervals.** Split conformal prediction fitted on held-out
residuals, with a stated coverage level that is empirically achieved. The
interval is the same width for every row, and that limitation is stated wherever
one is shown. Below 20 calibration rows, none is offered.

**Abstention.** The model can refuse. Severe missingness, an unseen category, a
value outside the training range, or low model confidence each mark a row as
*use caution* or *refused*. Abstained rows are kept in the output and flagged —
never dropped, never silently replaced with a guess.

**Decision threshold.** For classification: state the cost of a false alarm, the
cost of a miss, the cost of acting and the value of a catch, and the cheapest
cut-off is computed. Always described as conditional on those costs, never as
universally optimal.

**Calibration.** Brier score, log loss, a skill score against always predicting
the base rate, a reliability table, and a plain reading — *when this model says
80%, the event happens about N% of the time*.

**Dataset versions.** Each distinct state of the data is fingerprinted over its
values and summarised; the diff names rows and columns added or removed, type
changes, missingness shifts, new or vanished categories, and distributions that
moved — with a warning on each that would invalidate a model.

**Reviewer mode.** Seven areas with a status, what to check and where to look.
Deliberately not a verdict.

---

## 11. Output

**Natural language.** A question in plain English is parsed into an intent, the
plan is shown before anything expensive runs, and the answer comes back with its
evidence. Example prompts are clickable rather than merely listed.

**Report.** The same analysis written for two audiences (management, technical,
or both), with charts placed in the section they illustrate. Optional
methodology sections:

- *Methodology* — the question, the validation design and why that one, exactly
  how the rows were divided, the random seed, every metric's definition **and
  formula**, and where each preprocessing step was fitted.
- *The calculations* — every model's score on every fold beside the mean those
  folds produce; training against held-out in both raw units and the relative
  gap the ranking uses; hyper-parameters; the composite weights; importances and
  coefficients as numbers; the confusion matrix as counts.
- *Did it run cleanly* — every stage with status and timing, every model that
  failed and why, all eleven self-checks with what each found, library versions.
- *What the preprocessing learned* — the median it would use, the levels one-hot
  encoding found, the limits winsorising clips to.

**Export formats.** Markdown, HTML (self-contained, interactive charts, opens
offline), Excel (multi-sheet with a charts sheet), JSON, PDF, runnable standalone
Python that reproduces the analysis without the platform, and chart PNGs.

**Reproducibility.** A manifest fingerprinting the dataset, objective,
preprocessing, validation strategy, seed, model and library versions. Two runs
with the same fingerprint should produce the same numbers.

**Projects.** Save the dataset, context, pipelines, runs and fitted models to
plain files on disk and reopen them.

**Run comparison.** Diffs two runs — setup, scores, pipeline steps, findings —
with a *Changed* column. Runs answering different questions or predicting
different targets are refused by name rather than given a meaningless score
difference. A difference smaller than the fold-to-fold spread within a single run
is called a tie.

**Scoring new data.** Point a trained model at a new file. Before predicting: a
schema check separating a required column that is absent (fatal) from an extra
one (ignored), a column that has changed kind, category values the model never
saw, and a sharp rise in missingness. After: predictions with class
probabilities, a drift report on every comparable column, and a downloadable CSV.
Rows empty across every needed column are skipped and counted rather than
guessed. Also available as `dsai predict`.

---

## 12. The interface

**Structure.** A collapsible navigation rail grouped into Workflow (Data,
Context, Preprocessing, Analysis, Models, Insights, Recommendations, Score),
Investigate (Ask, Statistics), Output (Report, Projects) and Reference (Model
library). A top strip carrying project identity and system state. A seven-stage
progress stepper reflecting what has actually happened, not where the user is
standing.

**Pages and their tabs.**

| Page | Tabs |
|---|---|
| Data | Upload / Database / Samples; then Preview, Detected types, Inspect a variable, Data quality, Possible targets, Statistics |
| Robustness | What could change this conclusion?, Where does the model work poorly? |
| Context | one form |
| Preprocessing | Build, Missing values, The pipeline, Edit steps, Preview effect, What it changed, Compare pipelines, Saved pipelines |
| Analysis | Candidate models, As a table, Why these choices |
| Models | Performance, Model card, Explanation, Diagnostics, Decision threshold, Why this row?, More data?, Hyper-parameters, Decision log, Charts |
| Insights | As a report, As cards |
| Recommendations | grouped by category |
| Score new data | Upload / Re-score; then Predictions, Reliability, Has the data changed?, What the predictions look like |
| Statistics | Describe, Compare groups, Relationships, Assumptions |
| Report | Read, Charts, Methodology & workings, Decision log, Reproducibility, Export |
| Projects | Save, Open, Run history, Compare two runs, Review this analysis, Dataset versions |
| Model library | Algorithms, Preprocessing steps, Extending it |

**Visual language.** Four kinds of statement are deliberately distinguishable at
a glance: a measurement gets no decoration at all; an inference gets a thin
accent rule and a label naming the method; a decision the user must make is the
only filled, raised surface in the app; a limitation is a quiet margin note.
Filled alert styling is reserved for things that are genuinely wrong.

**Theme.** Dark by default with a light mode, from one set of tokens shared with
the charts. A five-step elevation ladder, a four-step spacing scale, a narrow
type scale, and motion in the 150–300ms band.

**Colour.** A colour-vision-deficiency-validated categorical palette in fixed
order, never cycled. There is no separate "AI" accent colour: every violet
candidate measured under 4 ΔE from the existing series violet, meaning a reader
could not separate AI chrome from a data series, so AI identity is carried by a
mark and by motion instead.

**Accessibility.** Text size to 1.5×, high contrast, reduced motion,
always-visible chart data tables, underlined links. A skip link, visible focus
rings clearing both surfaces, screen-reader labels on the stepper and metric
groups, decorative glyphs hidden from screen readers. Every chart goes through
one component that extracts its own data table from the figure, so no chart can
be added without a text alternative. Confidence and severity are never conveyed
by colour alone.

**Error and empty states.** Errors name the cause and the next move rather than
saying something failed. Empty states offer the action that resolves them.

**Progress.** Long operations show named stages — "Profiling variables and data
quality", "Training and validating 5 models", "Comparing models" — not a spinner.

---

## 13. Command line

```
dsai profile <file>                     # profile and report quality
dsai analyse <file> --target <col>      # full analysis, optionally --export
dsai predict <train> <new> --target <col>
dsai ask <file> "question"
dsai models / dsai steps                # browse the registry
dsai app                                # the workspace
dsai doctor                             # what is wrong with this install, and the fix
dsai version
```

`dsai doctor` checks the Python version, virtual environment, whether the install
is editable (a non-editable install means `git pull` does not update the running
code), every optional package and what each adds, the Streamlit config, whether
an older copy is already serving port 8501, uncommitted local changes, and
whether static chart export can find a browser — each failure with the command
that fixes it.

---

## 14. Known gaps — the author's own list

A reviewer should extend this, not be bounded by it.

1. **No scenario / what-if tool.** The platform shows what drives an outcome but
   cannot answer "what happens if I raise this by 10%".
1a. **No analysis sandbox.** Experiments overwrite nothing, but there is no
   explicit "try this without touching the main run" mode.
2. **No scheduled or repeated runs.** Every analysis is manual.
3. **Dark mode is CSS layered over a light base.** Streamlit has no runtime
   theming API. It works, but a hard refresh can flash light first.
4. **No multi-user or collaboration features.** Single user, single machine.
5. **No model monitoring over time.** Drift is checked at scoring time against
   the training data, and dataset versions are tracked within a session, but
   nothing persists a drift or performance history across sessions.
6. **No time-series cross-validation beyond a single forward split.**
7. **No causal inference.** The platform is careful to say every relationship is
   associational, but offers no tools (matching, instrumental variables,
   difference-in-differences) to go further.
8. **No text or image data support** beyond basic text-length features.
9. **Association rules are limited to Apriori and FP-Growth**, and the basket
   layout detection is heuristic.
10. **No undo across the whole workspace** — the pipeline has undo/redo, nothing
    else does.
11. **An `llm` optional extra is declared and never used.** It installs the
    Anthropic SDK; nothing in the codebase imports it. Either something was
    intended there or the extra should go.

---

## 15. Questions worth a reviewer's attention

- Is the seven-step workflow the right spine, or does it force a linear path
  through work that is actually iterative?
- Is the four-kinds-of-statement visual language legible, or is the distinction
  too subtle to carry the weight placed on it?
- Does the volume of caveats build trust or erode it? Is there a point at which
  honesty becomes noise?
- Is 111 algorithms a strength or a liability? Does the model library make them
  findable, or is the number itself the problem?
- Are the three modes (automatic / assisted / manual) genuinely distinct in use,
  or does one dominate?
- What would a first-time user get stuck on, and does the app notice?
