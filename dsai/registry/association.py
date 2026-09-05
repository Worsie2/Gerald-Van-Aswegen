"""Association-rule / pattern-mining pack.

These do not follow the estimator API — they are functions over a transaction
table — so the analysis engine calls them through
:mod:`dsai.engines.association_runner` rather than the experiment loop.
"""

from __future__ import annotations

from dsai.core.schema import TaskType
from dsai.registry._helpers import TRANSPARENT, hp, lazy, spec
from dsai.registry.base import Cost, register

AR = [TaskType.ASSOCIATION_RULES]
METRICS = ["support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"]

SPECS = [
    spec(
        key="apriori",
        name="Apriori",
        category="association",
        family="pattern_mining",
        task_types=AR,
        builder=lazy("mlxtend.frequent_patterns:apriori"),
        module="mlxtend",
        requires_numeric_only=False,
        interpretability=TRANSPARENT,
        cost=Cost.HIGH,
        min_rows=50,
        assumptions=["Data is a basket/transaction table: one row per transaction, one column per item"],
        advantages=["Rules read as plain sentences", "Support, confidence and lift are all directly interpretable"],
        limitations=["Slow with many distinct items", "A low support threshold produces an unmanageable number of rules"],
        good_for=["market basket analysis", "cross-sell discovery", "co-occurrence patterns"],
        hyperparameters=[
            hp("min_support", "float", 0.05, 0.001, 0.5, description="Minimum share of transactions containing the itemset."),
            hp("max_len", "int", 3, 2, 5, description="Largest itemset size to consider."),
        ],
        metrics=METRICS,
        tags=["rules", "basket"],
    ),
    spec(
        key="fpgrowth",
        name="FP-Growth",
        category="association",
        family="pattern_mining",
        task_types=AR,
        builder=lazy("mlxtend.frequent_patterns:fpgrowth"),
        module="mlxtend",
        requires_numeric_only=False,
        interpretability=TRANSPARENT,
        cost=Cost.MEDIUM,
        min_rows=50,
        assumptions=["Same transaction-table shape as Apriori"],
        advantages=["Far faster than Apriori on large baskets", "Same rules, same interpretation"],
        limitations=["Higher memory use", "Still explodes at very low support"],
        good_for=["large transaction datasets"],
        hyperparameters=[
            hp("min_support", "float", 0.05, 0.001, 0.5),
            hp("max_len", "int", 3, 2, 5),
        ],
        metrics=METRICS,
        tags=["rules", "basket", "fast"],
    ),
]

for _spec in SPECS:
    register(_spec, replace=True)

_LOADED = True
