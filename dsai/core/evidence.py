"""Evidence as a first-class object, with identifiers that survive an export.

A finding that says "premium customers spend more" and lists three numbers
underneath is not traceable: a reader cannot tell which run produced it, which
model, which pipeline, or which version of the data. Six months later nobody
can answer "where did this come from", and the claim quietly becomes folklore.

So every finding and every recommendation carries an ID, and every number
supporting it is an :class:`Evidence` record naming its own kind, its source and
the run it came from. The chain is:

    Recommendation → Finding → Evidence → Run → Model → Pipeline → Dataset

IDs are derived from content, not from a counter. Two identical findings in two
runs of the same analysis get the same ID, which is what makes a diff between
runs meaningful — and it means an ID stays stable across a re-export.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from dsai.core.schema import Confidence, EvidenceKind, JsonMixin

__all__ = ["Evidence", "EvidenceLedger", "finding_id", "recommendation_id", "evidence_id"]


def _short_hash(*parts: Any, prefix: str, length: int = 4) -> str:
    """A stable short ID derived from content.

    Content-derived rather than sequential so the same finding keeps the same
    identifier across runs and across exports. A counter would renumber
    everything the moment one finding disappeared, which makes a comparison
    between two runs unreadable.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length].upper()}"


def finding_id(title: str) -> str:
    return _short_hash(title, prefix="F")


def recommendation_id(action: str) -> str:
    return _short_hash(action, prefix="R")


def evidence_id(statement: str, kind: str) -> str:
    return _short_hash(statement, kind, prefix="E")


@dataclass
class Evidence(JsonMixin):
    """One piece of support for a claim, with its provenance attached.

    ``kind`` is the same five-way distinction the whole platform runs on: what
    was observed, what a test established, what a model derived, what the
    platform interpreted, and what the user asserted. Keeping it on the evidence
    rather than only on the finding means a single finding can rest on several
    kinds of support at once — which is usually the honest situation — and a
    reader can see which parts are measurement and which are inference.
    """

    id: str = ""
    statement: str = ""
    kind: EvidenceKind = EvidenceKind.OBSERVED
    #: Where the number came from: a column, a test name, a model, a fold.
    source: str = ""
    value: float | None = None
    #: The run that produced it, so a claim can be traced to a fingerprint.
    run_id: str = ""
    model: str = ""
    pipeline: str = ""
    dataset: str = ""
    caveats: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = evidence_id(self.statement, self.kind.value)

    @property
    def is_measurement(self) -> bool:
        """Whether this is something counted, as opposed to concluded."""
        return self.kind in (EvidenceKind.OBSERVED, EvidenceKind.STATISTICAL)

    def render(self) -> str:
        label = {
            EvidenceKind.OBSERVED: "measured",
            EvidenceKind.STATISTICAL: "tested",
            EvidenceKind.MODEL: "model-derived",
            EvidenceKind.INTERPRETATION: "interpretation",
            EvidenceKind.USER_ASSUMPTION: "your assumption — not verified",
        }[self.kind]
        source = f" · {self.source}" if self.source else ""
        return f"[{self.id}] {self.statement}  ({label}{source})"


@dataclass
class EvidenceLedger(JsonMixin):
    """Every piece of evidence in one run, and what rests on what.

    Built after the analysis rather than during it, from the findings and
    recommendations the engines already produce. That keeps the engines
    unchanged and means the ledger cannot disagree with what the report says —
    it is assembled from the same objects.
    """

    run_id: str = ""
    dataset: str = ""
    model: str = ""
    pipeline: str = ""
    fingerprint: str = ""
    items: dict[str, Evidence] = field(default_factory=dict)
    #: finding id -> the evidence ids supporting it
    supports: dict[str, list[str]] = field(default_factory=dict)
    #: recommendation id -> the finding ids it rests on
    rests_on: dict[str, list[str]] = field(default_factory=dict)
    #: id -> the title or action it names, so a chain can be rendered in words
    labels: dict[str, str] = field(default_factory=dict)

    def add(self, evidence: Evidence) -> Evidence:
        evidence.run_id = evidence.run_id or self.run_id
        evidence.model = evidence.model or self.model
        evidence.pipeline = evidence.pipeline or self.pipeline
        evidence.dataset = evidence.dataset or self.dataset
        self.items.setdefault(evidence.id, evidence)
        return self.items[evidence.id]

    def evidence_for(self, finding: str) -> list[Evidence]:
        return [self.items[i] for i in self.supports.get(finding, []) if i in self.items]

    def chain(self, recommendation: str) -> list[dict[str, Any]]:
        """The full lineage of one recommendation, as rows a UI can render.

        Deliberately a list of plain dicts rather than a graph object: the point
        is to be readable and exportable, not to be traversed.
        """
        rows: list[dict[str, Any]] = [{
            "level": "Recommendation", "id": recommendation,
            "label": self.labels.get(recommendation, ""), "kind": "",
        }]
        for found in self.rests_on.get(recommendation, []):
            rows.append({"level": "Finding", "id": found,
                         "label": self.labels.get(found, ""), "kind": ""})
            for item in self.evidence_for(found):
                rows.append({"level": "Evidence", "id": item.id, "label": item.statement,
                             "kind": item.kind.value.replace("_", " ")})
        rows += [
            {"level": "Run", "id": self.run_id, "label": self.fingerprint, "kind": ""},
            {"level": "Model", "id": "", "label": self.model, "kind": ""},
            {"level": "Pipeline", "id": "", "label": self.pipeline, "kind": ""},
            {"level": "Dataset", "id": "", "label": self.dataset, "kind": ""},
        ]
        return rows

    def counts(self) -> dict[str, int]:
        by_kind: dict[str, int] = {}
        for item in self.items.values():
            by_kind[item.kind.value] = by_kind.get(item.kind.value, 0) + 1
        return by_kind

    @property
    def measured_share(self) -> float:
        """How much of the evidence is measurement rather than inference."""
        if not self.items:
            return 0.0
        measured = sum(1 for item in self.items.values() if item.is_measurement)
        return measured / len(self.items)


def build_ledger(run: Any) -> EvidenceLedger:
    """Assemble the ledger for a completed run.

    Findings and recommendations already carry their evidence as free text; this
    turns each line into a record with an identity and a provenance, and links
    recommendations to the findings they name. Where a recommendation does not
    name a finding, it is linked to none rather than to a guess — an invented
    link is worse than an absent one.
    """
    from dsai.repro.provenance import build_manifest

    ledger = EvidenceLedger(
        run_id=run.id,
        dataset=run.dataset_name,
        model=run.best.model_name if run.best else "",
        pipeline=(run.pipeline.name if run.pipeline is not None else "none"),
    )
    try:
        ledger.fingerprint = build_manifest(run).fingerprint()
    except Exception:
        ledger.fingerprint = ""

    for finding in run.findings:
        key = finding_id(finding.title)
        finding.id = key
        ledger.labels[key] = finding.title
        ids: list[str] = []
        for statement in finding.evidence:
            item = ledger.add(Evidence(
                statement=statement, kind=finding.kind,
                source=", ".join(finding.columns[:3]),
                caveats=list(finding.caveats),
            ))
            ids.append(item.id)
        ledger.supports[key] = ids

    titles = {f.title: finding_id(f.title) for f in run.findings}
    for recommendation in run.recommendations:
        key = recommendation_id(recommendation.action)
        recommendation.id = key
        ledger.labels[key] = recommendation.action
        linked: list[str] = []
        for title, found_id in titles.items():
            haystack = " ".join([recommendation.reason, *recommendation.evidence]).lower()
            if title.lower() in haystack:
                linked.append(found_id)
        ledger.rests_on[key] = linked

    # The user's own assertions are evidence too, and the only kind that was
    # never checked. Recording them here is what stops them being mistaken for
    # measurements further down the chain.
    for assumption in getattr(run.context, "assumptions", []) or []:
        ledger.add(Evidence(statement=assumption, kind=EvidenceKind.USER_ASSUMPTION,
                            source="you told the platform this"))
    return ledger
