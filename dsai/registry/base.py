"""The Model Registry — the plugin architecture the whole platform hangs off.

Nothing in this codebase hard-codes a list of models. Algorithms are described
by :class:`ModelSpec` metadata and registered into a :class:`ModelRegistry`.
The decision engine *queries* the registry for candidates that suit the current
dataset profile; adding a new algorithm is a matter of appending one spec.
"""

from __future__ import annotations

import importlib
import enum
from dataclasses import dataclass, field
from typing import Any, Callable

from dsai.core.schema import JsonMixin, TaskType


class Interpretability(str, enum.Enum):
    """How readable the model's mechanism is to a human."""

    TRANSPARENT = "transparent"     # coefficients / rules a person can read directly
    MODERATE = "moderate"           # importances and partial dependence explain most of it
    OPAQUE = "opaque"               # needs post-hoc explanation to say anything

    @property
    def rank(self) -> int:
        return {"transparent": 3, "moderate": 2, "opaque": 1}[self.value]


class Cost(str, enum.Enum):
    """Rough training cost, used to keep the tournament inside a time budget."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @property
    def rank(self) -> int:
        return {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}[self.value]


@dataclass
class HyperParam(JsonMixin):
    """One tunable knob, with enough metadata to build a UI and a search space."""

    name: str
    kind: str                      # int | float | categorical | bool
    default: Any = None
    low: float | None = None
    high: float | None = None
    choices: list[Any] = field(default_factory=list)
    log_scale: bool = False
    description: str = ""
    tune: bool = True              # include in automatic hyper-parameter search

    def search_values(self, n: int = 5) -> list[Any]:
        """A small, sensible grid for random/grid search."""
        import numpy as np

        if self.kind == "categorical" or self.choices:
            return list(self.choices) if self.choices else [self.default]
        if self.kind == "bool":
            return [True, False]
        if self.low is None or self.high is None:
            return [self.default]
        if self.log_scale:
            values = np.logspace(np.log10(max(self.low, 1e-9)), np.log10(self.high), n)
        else:
            values = np.linspace(self.low, self.high, n)
        if self.kind == "int":
            return sorted({int(round(v)) for v in values})
        return [round(float(v), 6) for v in values]


@dataclass
class ModelSpec(JsonMixin):
    """Everything the platform needs to know about one algorithm.

    ``builder`` is a callable returning an unfitted estimator; it is only invoked
    when the model is actually selected, so specs for uninstalled libraries can
    sit in the registry harmlessly and simply report as unavailable.
    """

    key: str
    name: str
    category: str
    task_types: list[TaskType]
    builder: Callable[..., Any] | None = None
    module: str = "sklearn"            # import path used for the availability check
    version: str = "1.0"
    family: str = "other"              # linear | tree | ensemble | kernel | neural | probabilistic | distance | statistical
    # data requirements
    requires_scaling: bool = False
    requires_numeric_only: bool = True
    handles_missing: bool = False
    handles_categorical: bool = False
    supports_multiclass: bool = True
    supports_predict_proba: bool = False
    supports_feature_importance: bool = False
    supports_coefficients: bool = False
    supports_sample_weight: bool = False
    min_rows: int = 20
    max_recommended_rows: int | None = None
    max_recommended_features: int | None = None
    # judgement metadata
    interpretability: Interpretability = Interpretability.MODERATE
    cost: Cost = Cost.MEDIUM
    complexity: str = "O(n)"
    assumptions: list[str] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    good_for: list[str] = field(default_factory=list)
    hyperparameters: list[HyperParam] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    docs_url: str = ""
    tags: list[str] = field(default_factory=list)
    # scoring hints used by the decision engine
    nonlinear: bool = False
    robust_to_outliers: bool = False
    handles_high_dimensionality: bool = False
    baseline: bool = False

    def default_params(self) -> dict[str, Any]:
        return {h.name: h.default for h in self.hyperparameters if h.default is not None}

    def build(self, **params: Any) -> Any:
        if self.builder is None:
            raise RuntimeError(f"Model '{self.key}' has no builder attached.")
        merged = {**self.default_params(), **params}
        return self.builder(**merged)

    def is_available(self) -> bool:
        """True when the underlying library is importable in this environment."""
        try:
            importlib.import_module(self.module.split(".")[0])
            return True
        except Exception:
            return False

    def search_space(self, n: int = 4) -> dict[str, list[Any]]:
        return {h.name: h.search_values(n) for h in self.hyperparameters if h.tune and h.kind != "fixed"}

    def summary(self) -> str:
        return (
            f"{self.name} ({self.category}) — {self.interpretability.value} interpretability, "
            f"{self.cost.value} cost. {'; '.join(self.advantages[:2])}"
        )


class ModelRegistry:
    """A searchable catalogue of :class:`ModelSpec` objects."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    # -- registration ---------------------------------------------------------
    def register(self, spec: ModelSpec, replace: bool = False) -> ModelSpec:
        if spec.key in self._specs and not replace:
            raise ValueError(f"Model key '{spec.key}' is already registered.")
        self._specs[spec.key] = spec
        return spec

    def register_many(self, specs: list[ModelSpec], replace: bool = False) -> None:
        for spec in specs:
            self.register(spec, replace=replace)

    def unregister(self, key: str) -> None:
        self._specs.pop(key, None)

    # -- lookup ---------------------------------------------------------------
    def __contains__(self, key: object) -> bool:
        return key in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, key: str) -> ModelSpec:
        if key not in self._specs:
            raise KeyError(f"No model registered under '{key}'.")
        return self._specs[key]

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def keys(self) -> list[str]:
        return list(self._specs)

    def categories(self) -> list[str]:
        return sorted({s.category for s in self._specs.values()})

    def families(self) -> list[str]:
        return sorted({s.family for s in self._specs.values()})

    def available(self) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.is_available()]

    def find(
        self,
        task_type: TaskType | None = None,
        category: str | None = None,
        family: str | None = None,
        max_cost: Cost | None = None,
        min_interpretability: Interpretability | None = None,
        n_rows: int | None = None,
        n_features: int | None = None,
        require_proba: bool = False,
        require_importance: bool = False,
        require_multiclass: bool = False,
        handles_missing: bool | None = None,
        only_available: bool = True,
        tags: list[str] | None = None,
    ) -> list[ModelSpec]:
        """Filter the registry down to the specs that fit the situation."""
        out: list[ModelSpec] = []
        for spec in self._specs.values():
            if task_type is not None and task_type not in spec.task_types:
                continue
            if category is not None and spec.category != category:
                continue
            if family is not None and spec.family != family:
                continue
            if max_cost is not None and spec.cost.rank > max_cost.rank:
                continue
            if min_interpretability is not None and spec.interpretability.rank < min_interpretability.rank:
                continue
            if n_rows is not None and n_rows < spec.min_rows:
                continue
            if n_rows is not None and spec.max_recommended_rows and n_rows > spec.max_recommended_rows:
                continue
            if n_features is not None and spec.max_recommended_features and n_features > spec.max_recommended_features:
                continue
            if require_proba and not spec.supports_predict_proba:
                continue
            if require_importance and not (spec.supports_feature_importance or spec.supports_coefficients):
                continue
            if require_multiclass and not spec.supports_multiclass:
                continue
            if handles_missing is not None and spec.handles_missing != handles_missing:
                continue
            if tags and not set(tags).issubset(set(spec.tags)):
                continue
            if only_available and not spec.is_available():
                continue
            out.append(spec)
        return out

    def stats(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_task: dict[str, int] = {}
        for spec in self._specs.values():
            by_category[spec.category] = by_category.get(spec.category, 0) + 1
            for t in spec.task_types:
                by_task[t.value] = by_task.get(t.value, 0) + 1
        return {
            "total": len(self._specs),
            "available": len(self.available()),
            "by_category": dict(sorted(by_category.items())),
            "by_task": dict(sorted(by_task.items())),
        }

    def to_catalogue(self) -> list[dict[str, Any]]:
        """A flat table of the whole registry, for display or export."""
        return [
            {
                "key": s.key,
                "name": s.name,
                "category": s.category,
                "family": s.family,
                "tasks": ", ".join(t.value for t in s.task_types),
                "interpretability": s.interpretability.value,
                "cost": s.cost.value,
                "handles_missing": s.handles_missing,
                "nonlinear": s.nonlinear,
                "available": s.is_available(),
                "n_hyperparameters": len(s.hyperparameters),
                "advantages": "; ".join(s.advantages),
                "limitations": "; ".join(s.limitations),
            }
            for s in sorted(self._specs.values(), key=lambda x: (x.category, x.name))
        ]


#: The process-wide registry. Plugin modules import this and append to it.
REGISTRY = ModelRegistry()


def register(spec: ModelSpec, replace: bool = False) -> ModelSpec:
    """Module-level convenience so plugins read as ``register(ModelSpec(...))``."""
    return REGISTRY.register(spec, replace=replace)


def load_builtin_models() -> ModelRegistry:
    """Import every bundled algorithm pack. Safe to call repeatedly."""
    packs = [
        "dsai.registry.regression",
        "dsai.registry.classification",
        "dsai.registry.clustering",
        "dsai.registry.dimensionality",
        "dsai.registry.timeseries",
        "dsai.registry.anomaly",
        "dsai.registry.association",
    ]
    for pack in packs:
        try:
            module = importlib.import_module(pack)
            importlib.reload(module) if getattr(module, "_LOADED", False) else None
        except Exception as exc:  # pragma: no cover - a broken pack must not kill the app
            import logging

            logging.getLogger(__name__).warning("Could not load model pack %s: %s", pack, exc)
    return REGISTRY


def load_plugins(entry_point_group: str = "dsai.models") -> list[str]:
    """Load third-party model packs advertised through Python entry points.

    A separate package can ship extra algorithms by declaring::

        [project.entry-points."dsai.models"]
        my_pack = "my_package.models:register_all"
    """
    loaded: list[str] = []
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=entry_point_group):
            try:
                ep.load()()
                loaded.append(ep.name)
            except Exception as exc:  # pragma: no cover
                import logging

                logging.getLogger(__name__).warning("Plugin %s failed to load: %s", ep.name, exc)
    except Exception:
        pass
    return loaded
