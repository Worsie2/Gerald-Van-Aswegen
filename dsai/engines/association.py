"""Market-basket analysis: turn transaction data into readable rules.

Association mining needs the data in a very specific shape — one row per
transaction, one boolean column per item. Real datasets almost never arrive that
way, so most of the work here is reshaping long transaction tables, wide
indicator tables and one-item-per-row logs into that form, and then translating
support / confidence / lift into sentences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import Confidence, EvidenceKind, Finding, JsonMixin, Recommendation


@dataclass
class BasketShape(JsonMixin):
    """How the transaction table was reconstructed, so the user can check it."""

    layout: str = ""              # long | wide_binary | delimited
    transaction_column: str | None = None
    item_column: str | None = None
    n_transactions: int = 0
    n_items: int = 0
    mean_basket_size: float = 0.0
    note: str = ""


@dataclass
class AssociationResult(JsonMixin):
    algorithm: str = ""
    shape: BasketShape = field(default_factory=BasketShape)
    frequent_itemsets: list[dict[str, Any]] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def rules_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rules) if self.rules else pd.DataFrame()


def detect_basket_layout(
    frame: pd.DataFrame,
    transaction_column: str | None = None,
    item_column: str | None = None,
) -> BasketShape:
    """Work out how the transaction data is arranged before trying to mine it."""
    shape = BasketShape()
    if transaction_column and item_column:
        shape.layout = "long"
        shape.transaction_column = transaction_column
        shape.item_column = item_column
        shape.n_transactions = int(frame[transaction_column].nunique())
        shape.n_items = int(frame[item_column].nunique())
        shape.mean_basket_size = round(len(frame) / max(shape.n_transactions, 1), 2)
        shape.note = "One row per item per transaction."
        return shape

    boolean_like = [
        c for c in frame.columns
        if frame[c].dropna().nunique() <= 2
        and set(pd.unique(frame[c].dropna())).issubset({0, 1, True, False, "0", "1", "Y", "N", "yes", "no"})
    ]
    if len(boolean_like) >= 3:
        shape.layout = "wide_binary"
        shape.n_transactions = len(frame)
        shape.n_items = len(boolean_like)
        matrix = frame[boolean_like].astype(bool)
        shape.mean_basket_size = round(float(matrix.sum(axis=1).mean()), 2)
        shape.note = f"One row per transaction with {len(boolean_like)} yes/no item columns."
        return shape

    for column in frame.columns:
        series = frame[column].dropna().astype(str)
        if series.empty:
            continue
        if series.str.contains(r"[,;|]").mean() > 0.5:
            shape.layout = "delimited"
            shape.item_column = column
            shape.n_transactions = len(frame)
            items = series.str.split(r"[,;|]")
            shape.n_items = int(items.explode().str.strip().nunique())
            shape.mean_basket_size = round(float(items.str.len().mean()), 2)
            shape.note = f"Items listed in a single delimited column ('{column}')."
            return shape

    shape.layout = "unknown"
    shape.note = (
        "The data does not look like a transaction table. Basket analysis needs either one row per "
        "item per transaction (give the transaction and item columns), one row per transaction with "
        "yes/no item columns, or a column listing items separated by commas."
    )
    return shape


def to_basket_matrix(
    frame: pd.DataFrame,
    shape: BasketShape,
    min_item_frequency: int = 1,
    max_items: int = 500,
) -> pd.DataFrame:
    """Produce the boolean transaction × item matrix the miners expect."""
    if shape.layout == "long":
        matrix = (
            frame.assign(__present__=1)
            .pivot_table(
                index=shape.transaction_column, columns=shape.item_column,
                values="__present__", aggfunc="max", fill_value=0,
            )
            .astype(bool)
        )
    elif shape.layout == "wide_binary":
        columns = [
            c for c in frame.columns
            if frame[c].dropna().nunique() <= 2
            and set(pd.unique(frame[c].dropna())).issubset({0, 1, True, False, "0", "1", "Y", "N", "yes", "no"})
        ]
        matrix = frame[columns].replace({"Y": 1, "N": 0, "yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(bool)
    elif shape.layout == "delimited":
        exploded = (
            frame[shape.item_column].dropna().astype(str).str.split(r"[,;|]").explode().str.strip()
        )
        matrix = pd.crosstab(exploded.index, exploded).astype(bool)
    else:
        raise ValueError(shape.note)

    counts = matrix.sum(axis=0)
    keep = counts[counts >= min_item_frequency].sort_values(ascending=False).head(max_items).index
    return matrix[list(keep)]


def mine_associations(
    frame: pd.DataFrame,
    transaction_column: str | None = None,
    item_column: str | None = None,
    algorithm: str = "fpgrowth",
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_len: int = 3,
    top_n: int = 50,
) -> AssociationResult:
    """Find frequent itemsets and association rules, and explain what they mean."""
    result = AssociationResult(algorithm=algorithm)
    result.parameters = {
        "min_support": min_support, "min_confidence": min_confidence,
        "min_lift": min_lift, "max_len": max_len,
    }
    shape = detect_basket_layout(frame, transaction_column, item_column)
    result.shape = shape
    if shape.layout == "unknown":
        result.warnings.append(shape.note)
        return result

    try:
        from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    except ImportError:
        result.warnings.append(
            "Association mining needs mlxtend, which is not installed. Run: pip install mlxtend"
        )
        return result

    matrix = to_basket_matrix(frame, shape)
    if matrix.empty or matrix.shape[1] < 2:
        result.warnings.append("Fewer than two distinct items were found — there is nothing to associate.")
        return result

    miner = fpgrowth if algorithm == "fpgrowth" else apriori
    itemsets = miner(matrix, min_support=min_support, use_colnames=True, max_len=max_len)
    if itemsets.empty:
        result.warnings.append(
            f"No itemset appears in at least {min_support:.1%} of transactions. Lower the minimum "
            "support, or accept that purchases here are not co-occurring in stable patterns."
        )
        return result

    result.frequent_itemsets = [
        {
            "items": sorted(str(i) for i in row.itemsets),
            "size": len(row.itemsets),
            "support": round(float(row.support), 5),
            "n_transactions": int(round(row.support * len(matrix))),
        }
        for row in itemsets.sort_values("support", ascending=False).head(top_n).itertuples()
    ]

    try:
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    except (ValueError, KeyError):
        rules = pd.DataFrame()
    if rules.empty:
        result.warnings.append(
            f"Itemsets were found, but no rule reaches {min_confidence:.0%} confidence. The items "
            "co-occur, but not reliably enough to predict one from another."
        )
        return result

    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False).head(top_n)
    for row in rules.itertuples():
        antecedents = sorted(str(i) for i in row.antecedents)
        consequents = sorted(str(i) for i in row.consequents)
        result.rules.append(
            {
                "if": antecedents,
                "then": consequents,
                "support": round(float(row.support), 5),
                "confidence": round(float(row.confidence), 4),
                "lift": round(float(row.lift), 4),
                "leverage": round(float(row.leverage), 5),
                "conviction": round(float(row.conviction), 4) if np.isfinite(row.conviction) else None,
                "n_transactions": int(round(row.support * len(matrix))),
                "statement": _rule_sentence(antecedents, consequents, row),
            }
        )

    result.findings = _association_findings(result, len(matrix))
    result.recommendations = _association_recommendations(result)
    result.warnings.extend(_association_warnings(result, len(matrix)))
    return result


def _rule_sentence(antecedents: list[str], consequents: list[str], row) -> str:
    left = " and ".join(antecedents)
    right = " and ".join(consequents)
    lift = float(row.lift)
    strength = (
        f"{lift:.1f}× more likely than chance" if lift >= 1.1
        else "no more likely than chance" if lift < 1.05
        else f"{lift:.2f}× more likely than chance"
    )
    return (
        f"Transactions containing {left} also contain {right} {float(row.confidence):.0%} of the time "
        f"— {strength}. This pattern appears in {float(row.support):.1%} of all transactions."
    )


def _association_findings(result: AssociationResult, n_transactions: int) -> list[Finding]:
    findings: list[Finding] = []
    if not result.rules:
        return findings
    strongest = result.rules[0]
    findings.append(
        Finding(
            title=f"Strongest association: {' + '.join(strongest['if'])} → {' + '.join(strongest['then'])}",
            detail=strongest["statement"],
            kind=EvidenceKind.OBSERVED,
            evidence=[
                f"Support {strongest['support']:.1%} ({strongest['n_transactions']:,} of {n_transactions:,} transactions)",
                f"Confidence {strongest['confidence']:.1%}",
                f"Lift {strongest['lift']:.2f}",
            ],
            confidence=Confidence.HIGH if strongest["n_transactions"] >= 50 else Confidence.MODERATE,
            caveats=[
                "Association is co-occurrence, not cause. These items may simply be bought by the "
                "same kind of customer rather than one driving the other.",
            ],
        )
    )
    high_lift = [r for r in result.rules if r["lift"] >= 2.0]
    if high_lift:
        findings.append(
            Finding(
                title=f"{len(high_lift)} rule(s) show a strong lift above 2.0",
                detail=(
                    "These pairings occur at least twice as often as they would if the items were "
                    "unrelated — the clearest cross-sell candidates in the data."
                ),
                kind=EvidenceKind.OBSERVED,
                evidence=[r["statement"] for r in high_lift[:5]],
                confidence=Confidence.MODERATE,
            )
        )
    if result.frequent_itemsets:
        top = result.frequent_itemsets[0]
        findings.append(
            Finding(
                title=f"Most common combination: {' + '.join(top['items'])}",
                detail=f"Appears in {top['support']:.1%} of transactions ({top['n_transactions']:,} baskets).",
                kind=EvidenceKind.OBSERVED,
                evidence=[f"{' + '.join(i['items'])}: {i['support']:.1%}" for i in result.frequent_itemsets[:5]],
                confidence=Confidence.HIGH,
            )
        )
    return findings


def _association_recommendations(result: AssociationResult) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    actionable = [r for r in result.rules if r["lift"] >= 1.5 and r["confidence"] >= 0.4][:5]
    for rule in actionable:
        left, right = " + ".join(rule["if"]), " + ".join(rule["then"])
        recommendations.append(
            Recommendation(
                action=f"Offer {right} to customers buying {left}",
                reason=(
                    f"{rule['confidence']:.0%} of baskets containing {left} also contain {right}, "
                    f"which is {rule['lift']:.1f}× the base rate."
                ),
                evidence=[
                    f"Observed in {rule['n_transactions']:,} transactions",
                    f"Support {rule['support']:.1%}, confidence {rule['confidence']:.1%}, lift {rule['lift']:.2f}",
                ],
                confidence=Confidence.MODERATE if rule["n_transactions"] >= 50 else Confidence.LOW,
                expected_impact=(
                    f"If the association holds when prompted, roughly {rule['confidence']:.0%} of "
                    f"{left} buyers are already candidates for {right}."
                ),
                caveats=[
                    "This is observed co-occurrence. Whether prompting actually changes behaviour "
                    "can only be established by a test — run it on a holdout group before rolling out.",
                ],
                category="action",
                traceable_to=[f"association_rule:{left}->{right}"],
            )
        )
    return recommendations


def _association_warnings(result: AssociationResult, n_transactions: int) -> list[str]:
    warnings: list[str] = []
    if n_transactions < 100:
        warnings.append(
            f"Only {n_transactions} transactions. Support and confidence estimates will be very noisy "
            "at this size; treat any rule as a hypothesis to test rather than a finding."
        )
    if len(result.rules) > 200:
        warnings.append(
            f"{len(result.rules)} rules were generated. At this volume some will look strong purely "
            "by chance — raise the minimum support or lift and focus on the top handful."
        )
    trivial = [r for r in result.rules if r["lift"] < 1.1]
    if trivial and len(trivial) > len(result.rules) / 2:
        warnings.append(
            "Most rules have a lift near 1.0, meaning the items co-occur about as often as chance "
            "would predict. Popularity, not association, is driving these."
        )
    return warnings
