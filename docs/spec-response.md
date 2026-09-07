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

## Deferred, with reasons

### Phase 2 — not started

**§13 Sensitivity analysis** is the single most valuable item left. Re-running an
analysis across 20 reasonable specifications and reporting whether the direction
survives is a genuinely different claim from anything the platform makes today.
It is also the most expensive: 20× the compute, and it needs a specification
space that is defensible rather than arbitrary.

**§14 Analysis sandbox** depends on the run object model being clean enough to
fork. It largely is now.

**§16 Prediction intervals** — conformal prediction would be the honest
implementation. The spec's own caution applies: do not manufacture uncertainty
where the model cannot support it, and most models here cannot.

**§17 Model abstention** — the scoring engine already reports unseen categories,
schema mismatches and drift. Turning those into a per-row 🟢🟡🔴 status and an
abstention count is a small extension of existing work.

**§19 Threshold/cost analysis**, **§21 subgroup discovery**, **§24 data
versioning**, **§30 reviewer mode** — all tractable, none started.

### Phase 3 and 4 — architectural groundwork only

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

## What a reviewer should look at next

1. Whether **evidence strength** is a good idea at all. It is the one number in
   the application that summarises quality, and the strongest argument against it
   is that any such number gets quoted without its list.
2. Whether the **assumption-debt** score adds anything the reducing-confidence
   list does not already say.
3. Whether **stable finding IDs** are useful in practice or just noise on screen.
4. Whether the dashboard's answer-first ordering survives a run where the top
   finding is uninteresting.
