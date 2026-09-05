"""Small helpers that keep the algorithm packs readable."""

from __future__ import annotations

from typing import Any, Callable

from dsai.registry.base import Cost, HyperParam, Interpretability, ModelSpec

# Short aliases used heavily inside the packs.
TRANSPARENT = Interpretability.TRANSPARENT
MODERATE = Interpretability.MODERATE
OPAQUE = Interpretability.OPAQUE


def hp(
    name: str,
    kind: str,
    default: Any = None,
    low: float | None = None,
    high: float | None = None,
    choices: list[Any] | None = None,
    log: bool = False,
    description: str = "",
    tune: bool = True,
) -> HyperParam:
    return HyperParam(
        name=name,
        kind=kind,
        default=default,
        low=low,
        high=high,
        choices=choices or [],
        log_scale=log,
        description=description,
        tune=tune,
    )


def lazy(import_path: str) -> Callable[..., Any]:
    """Build an estimator only when it is actually needed.

    ``lazy("sklearn.ensemble:RandomForestRegressor")`` returns a callable that
    imports on first use, so specs for optional libraries (XGBoost, LightGBM,
    statsmodels, …) can live in the registry whether or not they are installed.
    """
    module_path, _, attr = import_path.partition(":")

    def _build(**params: Any) -> Any:
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        return cls(**params)

    _build.__name__ = f"build_{attr}"
    _build.import_path = import_path  # type: ignore[attr-defined]
    return _build


def spec(**kwargs: Any) -> ModelSpec:
    """Create a ModelSpec, deriving the availability module from the builder."""
    builder = kwargs.get("builder")
    if builder is not None and "module" not in kwargs:
        path = getattr(builder, "import_path", "")
        if path:
            kwargs["module"] = path.split(":")[0].split(".")[0]
    return ModelSpec(**kwargs)


# Reusable hyper-parameter blocks -------------------------------------------------

def tree_params(max_depth_default: Any = None) -> list[HyperParam]:
    return [
        hp("max_depth", "int", max_depth_default, 2, 30, description="Deeper trees fit more detail but overfit sooner."),
        hp("min_samples_split", "int", 2, 2, 40, description="Minimum rows required before a node may split."),
        hp("min_samples_leaf", "int", 1, 1, 20, description="Minimum rows in a final leaf; raise it to smooth the model."),
    ]


def forest_params() -> list[HyperParam]:
    return [
        hp("n_estimators", "int", 300, 50, 1000, description="Number of trees. More is steadier but slower."),
        hp("max_depth", "int", None, 3, 30, description="Depth cap per tree."),
        hp("min_samples_leaf", "int", 1, 1, 20, description="Minimum rows per leaf."),
        hp("max_features", "categorical", "sqrt", choices=["sqrt", "log2", None, 0.5],
           description="How many features each split may consider."),
        hp("n_jobs", "fixed", -1, tune=False),
        hp("random_state", "fixed", 42, tune=False),
    ]


def boosting_params() -> list[HyperParam]:
    return [
        hp("n_estimators", "int", 300, 50, 1500, description="Number of boosting rounds."),
        hp("learning_rate", "float", 0.1, 0.01, 0.3, log=True,
           description="Step size. Lower needs more rounds but generalises better."),
        hp("max_depth", "int", 3, 2, 10, description="Depth of each weak learner."),
        hp("subsample", "float", 1.0, 0.5, 1.0, description="Row sample per round; below 1 adds regularisation."),
        hp("random_state", "fixed", 42, tune=False),
    ]
