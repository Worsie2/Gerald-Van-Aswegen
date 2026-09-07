# Response to the upgrade specification

What was already there, what was built, what was deliberately deferred, and where
the specification conflicts with the existing design.

Written against the spec's own instruction (§52, §55): map first, implement
Phase 1, do not attempt everything at once.

---

## Already present before this round

Roughly a third of the specification described things the application already
did. Listing them so review effort goes where it is needed:

| Spec section | Status |
|---|---|
| §2 Existing strengths | All preserved. 214 tests pass. |
| §8 Progressive disclosure | Tabs, expanders and cards throughout; critical warnings never collapsed |
| §23 Experiment tracking | Run comparison exists (setup, scores, pipeline, findings, with a Changed column) |
| §26 Density levels | Report has management / analyst / technical audiences over one analysis |
| §32 Ask upgrade | Plan shown before expensive work, with clickable examples |
| §34 Explain this chart | Every chart carries a written caption and a data table |
| §37 Design system | Tokens for spacing, type, radius, elevation, motion; dark and light |
| §38 Streamlit rules | Already followed — stable keys, centralised state, no iframes |
| §44 Registry UX | Search, categories, compact cards; the full table behind an expander |
| §46 Tournament UX | Ranked list with recommended / strong alternative / baseline |
| §47 Visual comparison | One `leaderboard` and `rank_item` pattern across models, runs, pipelines |
| §48 Error states | Every error names cause and next move |
| §49 Accessibility | Text scale, contrast, motion, tables, focus, skip link, no colour-only meaning |
| §16 Calibration (partial) | Calibration curve and reliability already in Models → Diagnostics |

---

## Built this round — Phase 1

### §9, §10, §40 — Evidence ledger and lineage

`dsai/core/evidence.py`. Every finding and recommendation now carries a stable
identifier (`F-9F8B`, `R-0740`) derived from its own content rather than a
counter — so the same claim keeps the same ID across runs and exports, which is
what makes a diff between two runs readable.

Each supporting number becomes an `Evidence` record naming its kind (measured,
tested, model-derived, interpretation, your assumption), its source, and the run,
model, pipeline and dataset it came from. `ledger.chain(recommendation_id)`
returns the full lineage. Rendered on Insights and Recommendations as an
expandable lineage view.

The user's own assertions are recorded as evidence too, of kind
`user_assumption` — the only kind that was never checked. That is what stops them
being mistaken for measurements further down the chain.

A recommendation naming no finding is linked to none, not to a guess.

### §11, §27, §28 — Why trust this, known unknowns, assumption debt

`dsai/engines/trust.py`. Three outputs, never shown apart:

- **Evidence strength**, 0–100, a weighted count of the eleven self-checks. The
  weights are explicit and documented: beating the baseline is worth 20, a note
  about causality is worth 0 because it is a permanent property of observational
  data rather than a fault. Every surface that shows the number carries the
  sentence "not a statistical confidence level — it does not mean there is an
  N% chance the conclusion is right".
- **Known unknowns** — what the design cannot settle at all: whether anything
  causes the outcome, whether an unmeasured variable explains it, whether the
  sample represents the population, whether it holds into the future. These do
  not move the score, because they are not deficiencies.
- **Assumption debt**, 0–100 — what would have to be true and has not been shown:
  unverified user assumptions, imputation steps, unresolved leakage suspects,
  missing business context.

### §5, §15 — Analysis brief and data contract

`dsai/engines/brief.py`. Both written *before* the expensive work and stored on
the run.

The **brief** states the question in plain English (not a task-type enum), the
dataset, the proposed method, the validation design, the constraints, and a
three-valued verdict: suitable / proceed with caution / cannot defensibly answer.
The middle value is the useful one — an analysis that will produce numbers that
look fine and rest on something the reader should know first.

The **contract** records what the analysis requires of its data: the target,
required columns, types, the categories seen in training, missingness tolerances,
minimum rows, time ordering. Stored with the run and re-checked when new rows
arrive, because a model is valid only on data meeting the contract it was trained
under.

### §22 — Model cards

`dsai/reporting/model_card.py`. Generated from the run, so the card cannot drift
from what the model is. Purpose, data, validation, metrics with their source,
hyper-parameters, preprocessing, assumptions, limitations, appropriate uses,
**inappropriate uses**, subgroup performance, calibration, drift sensitivity,
fingerprint, library versions. Exportable as Markdown.

Inappropriate uses is the section that earns the card its keep. Every card format
lists what a model is good at; models fail where nobody thought to look.

### §20 — Subgroup performance

Included in the model card: error by categorical group, with a minimum of 30 rows
per group. A group of nine with a bad score is noise, and presenting it as a
finding invites someone to act on nothing. The wording is a measured difference,
never "bias" or "fairness failure" — investigating why is the reader's job.

### §4, §6, §7 — Dashboard, status bar, state-aware navigation

The landing page now leads with **the answer**, then **why trust this**, then what
drives it, what should happen next, and what could invalidate it.

A seven-cell **status bar** reads the workspace, not the current page: Data,
Question, Prepare, Model, Validate, Explain, Decide. State is carried by a
border, a glyph *and* a word, so it survives a colour-blind reader and a
black-and-white print.

The **navigation rail** carries status where it has something to say —
"evidence strength 93/100", "2 leakage suspect(s)", "no pipeline built",
"8 action(s)". Deliberately sparse: a marker beside every item is wallpaper.

### §31 — Audit package

`export_audit_package()` writes one directory containing both reports, the
methodology, the model card, the data contract, the evidence ledger, the trust
assessment, findings, recommendations, the decision log, the manifest, the
charts, runnable Python, and a README naming what each file is for and how to
read them in order. Every file carries the same fingerprint. Files that could not
be written are named in the README rather than being quietly absent.

---

## One bug the specification surfaced

`REGISTRY.find()` returned nothing when the bundled algorithm packs had not been
imported — so any code path that did not call `load_builtin_models()` first got
"no algorithm suits your data", which is indistinguishable from a real verdict
and sends the reader looking for a problem in their dataset that is not there.
`find()` now loads the built-ins when the registry is empty.

---

## Built this round — Phase 2

### §12, §13 — Sensitivity and robustness

`dsai/engines/sensitivity.py`, surfaced on a new **Robustness** page.

The analysis is re-run across up to 20 alternative specifications — a different
model family, minimal or thorough preprocessing, mean or KNN imputation instead
of median, winsorised or untouched outliers, a different seed, a different fold
count, and without whichever predictor is most questionable. Every variation is
one a competent analyst might have chosen; searching over unreasonable ones and
reporting that the result "survived" would be theatre.

Agreement is measured on **what a reader acts on** — whether the same variable
comes out strongest — not on the third decimal place of a metric. Two things
that mattered in testing: the baseline model is excluded (it ignores every
predictor by design, so including it manufactures a disagreement that means
nothing), and a categorical driver named before and after one-hot encoding
counts as the same variable, or every dataset with a categorical driver would
report a false negative.

On the water-customers sample: 15 specifications in 62 seconds, verdict
**stable**, the planted premium-tier effect strongest in 13 of 15.

### §16 — Prediction intervals

Split conformal prediction, fitted on held-out residuals. It assumes only that
new rows resemble the calibration rows — no distributional assumption, no
assumption that the model is correct. Empirical coverage on fresh data measures
0.912 against a 0.90 target.

The cost is that the interval is the same width everywhere. That is stated
plainly wherever one is shown: this method knows how wrong the model usually is,
not where it is less sure, and widening some intervals and not others would be
inventing a confidence the model does not have. Below 20 calibration rows it
refuses rather than producing a guess about a guess.

### §17 — Abstention

Four tests, all about the **row** rather than the model's output — a model is
equally confident about a row it understands and one it has never seen anything
like, so its own confidence cannot be the only test: severe missingness, a
category never seen in training, a value outside the training range, and low
model confidence where the model has one worth reading.

Abstained rows are kept in the output and flagged, never dropped and never
silently replaced with a guess. In testing, rows with a predictor set far outside
its training range predicted **R1.1 billion** — refused, correctly.

### §18, §19 — Calibration and decision threshold

Calibration gained Brier score, log loss, a Brier skill score against always
predicting the base rate, and a plain reading: *when this model says 80%, the
event happens about N% of the time*. Systematic over- and under-confidence are
named as such.

The threshold tool takes the cost of a false alarm, the cost of a miss, the cost
of acting and the value of a catch, and recommends the cheapest cut-off. Verified:
costly misses push the threshold down, costly false alarms push it up. Every
statement of the answer says it depends entirely on the costs supplied and is not
universally optimal.

### §20, §21 — Subgroup performance and discovery

Single conditions and pairs of conditions — "Region C *and* high spend", which is
usually where a real weakness hides. Two safeguards, both deliberately
conservative: nothing under 30 rows is reported, and a group is only flagged when
it stands beyond three standard errors, not merely above average, which half of
all groups are.

Tested against a planted weakness: found *region = C and spend high* as the top
result at 3.8× the overall error. The wording is always a measured difference —
naming it bias or a fairness failure would be a conclusion this evidence cannot
support, and would let the reader skip the investigation that matters.

### §24 — Dataset versioning

A version records shape, types, missingness, categories and distributions — never
the rows. The fingerprint is over the values, so two loads of the same file are
the same version and any real change is a new one. The diff names rows added and
removed, columns added and removed, type changes, missingness shifts, categories
that appeared or vanished, and distributions that moved by more than 0.2 standard
deviations — with a warning attached to each that would invalidate a model.

### §30 — Reviewer mode

Seven areas — question, data, preprocessing, validation, model, evidence,
recommendations — each with a status, a summary, what to check, and where to look.
Exportable.

Deliberately **not a verdict**. Whether the analysis is acceptable is the
judgement the reviewer was brought in to make, and a tool that made it for them
would be answering their question.

---

## Deferred, with reasons

### Phase 2 — remaining

**§14 Analysis sandbox** is the only Phase 2 item left. The run object model is
now clean enough to fork, so this is mostly UI: a second run that never
overwrites the first, with the existing run comparison pointed at the pair.

### Phase 3 and 4 — architectural groundwork only

Phase 2's dataset versioning is the foundation Phase 3's monitoring would build
on: a version is already fingerprinted and diffable, and the scoring page already
computes drift. What is missing is somewhere to *keep* those across sessions.

Monitoring, scheduled runs, deployment and API scoring need a persistent service.
This is a local desktop application by design; adding a service changes what the
product is, and that is a decision rather than a feature.

Causal inference (§42) is deliberately kept separate. The platform's care about
not presenting association as causation would be undermined by a half-built
causal module, and a real one requires the user to state assumptions the
application cannot check.

---

## Where the specification conflicts with the existing design

Stated explicitly, as §55 requires.

**1. §35 Command palette.** Streamlit cannot reliably capture `Ctrl/Cmd+K`
globally. Any implementation depends on JavaScript reaching out of the component
iframe into the parent document, which breaks on Streamlit upgrades. The spec
allows "the closest robust alternative"; a search-driven page is that, and it is
not built yet. The keyboard shortcut is not going to work.

**2. §3 Navigation restructure.** The proposed Project / Workspace / Investigate
/ Output / Operations grouping is better organised than the current Workflow /
Investigate / Output / Reference. It was not adopted this round because renaming
the pages breaks every deep link, every `st.page_link` target and the tests that
assert on page titles — a change worth making deliberately rather than as a side
effect of adding features.

**3. §26 Three density levels as a global setting.** The report already has three
audiences. Making density a global UI mode multiplies the surface every new
component must handle, and the spec elsewhere (§53) warns against a feature
supermarket. The current answer — progressive disclosure per page — costs less
and does most of the same work. Worth arguing about.

**4. §45 Compute budget with an estimated runtime.** An honest estimate needs
either a timing model or a calibration run, and a wrong estimate is worse than
none. The existing time-budget modes (fast / balanced / thorough) already control
the cost. Estimated memory in particular would be a guess presented as a figure.

**5. §25 Model monitoring "🔴 population shift".** A single traffic light for
population shift implies a threshold nobody has justified. The scoring page
already reports per-column drift with a reading in words. Reducing that to one
light would lose the part a reader can act on.

---

## Not adopted, and why

**Anything that increases feature count without increasing trust.** §53 sets the
test and it is the right one. Several items in the spec — a DAG builder, forecast
reconciliation, synthetic controls — would each need to be a small product of
their own to be honest, and a shallow version would be worse than none.

---

## Maintenance done alongside

`use_container_width` was deprecated by Streamlit with a removal date that has
now passed; 27 call sites were migrated to `width="stretch"` / `width="content"`
before the next Streamlit upgrade removed them.

---

## What a reviewer should look at next

1. Whether **evidence strength** is a good idea at all. It is the one number in
   the application that summarises quality, and the strongest argument against it
   is that any such number gets quoted without its list.
2. Whether the **assumption-debt** score adds anything the reducing-confidence
   list does not already say.
3. Whether **stable finding IDs** are useful in practice or just noise on screen.
4. Whether the dashboard's answer-first ordering survives a run where the top
   finding is uninteresting.
