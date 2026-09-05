"""The Experiment Engine: actually run the analysis.

This is what separates the platform from a chatbot that recommends models. It
builds the preprocessing pipeline, fits the estimator, cross-validates it,
scores it on a held-out set, records everything needed to reproduce the run, and
flags overfitting, instability and leakage as it goes.

Every experiment is immutable once complete and carries its own provenance, so a
tournament is a set of comparable, replayable records rather than a printout.
"""

from __future__ import annotations

import platform
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import Confidence, DatasetProfile, JsonMixin, Objective, TaskType
from dsai.engines import metrics as M
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.registry.base import REGISTRY, ModelRegistry, ModelSpec


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class ValidationResult(JsonMixin):
    strategy: str = ""
    n_splits: int = 0
    fold_scores: dict[str, list[float]] = field(default_factory=dict)
    mean_scores: dict[str, float] = field(default_factory=dict)
    std_scores: dict[str, float] = field(default_factory=dict)

    def stability(self, metric: str) -> float:
        """Coefficient of variation across folds — how much the score itself wobbles."""
        mean = abs(self.mean_scores.get(metric, 0.0))
        std = self.std_scores.get(metric, 0.0)
        return float(std / mean) if mean > 1e-12 else float("inf")


@dataclass
class ExperimentResult(JsonMixin):
    """One trained-and-validated model, with everything needed to judge it."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    model_key: str = ""
    model_name: str = ""
    task_type: TaskType = TaskType.EXPLORATORY
    target: str | None = None
    features: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)

    train_scores: dict[str, float] = field(default_factory=dict)
    test_scores: dict[str, float] = field(default_factory=dict)
    validation: ValidationResult = field(default_factory=ValidationResult)

    training_time_s: float = 0.0
    n_train: int = 0
    n_test: int = 0
    n_features_out: int = 0

    status: str = "success"            # success | failed | skipped
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    overfitting_gap: float | None = None
    notes: list[str] = field(default_factory=list)

    interpretability: str = ""
    cost: str = ""
    family: str = ""
    random_seed: int = 42
    extras: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def primary(self, metric: str) -> float | None:
        """Preferred score for ranking: cross-validated mean, else test, else train."""
        for source in (self.validation.mean_scores, self.test_scores, self.train_scores):
            if metric in source and np.isfinite(source[metric]):
                return float(source[metric])
        return None

    def score_source(self, metric: str) -> str:
        if metric in self.validation.mean_scores:
            return "cross-validated"
        if metric in self.test_scores:
            return "hold-out test"
        return "training data"


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

@dataclass
class ExperimentConfig(JsonMixin):
    test_size: float = 0.2
    random_state: int = 42
    cross_validate: bool = True
    validation: dict[str, Any] = field(default_factory=dict)
    scoring_metrics: list[str] = field(default_factory=list)
    tune: bool = False
    tuning_method: str = "random"       # grid | random | halving
    tuning_iterations: int = 20
    max_seconds_per_model: float = 300.0
    verbose: bool = False


class ExperimentEngine:
    """Trains, validates and records candidate models for one objective."""

    def __init__(
        self,
        frame: pd.DataFrame,
        objective: Objective,
        profile: DatasetProfile | None = None,
        registry: ModelRegistry | None = None,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.frame = frame
        self.objective = objective
        self.profile = profile
        self.registry = registry or REGISTRY
        self.config = config or ExperimentConfig()
        self.results: list[ExperimentResult] = []
        self._fitted: dict[str, Any] = {}
        self._split_cache: tuple | None = None

    # -- data preparation ----------------------------------------------------
    def _features(self) -> list[str]:
        if self.objective.features:
            return [c for c in self.objective.features if c in self.frame.columns]
        skip = {self.objective.target, self.objective.time_column}
        return [c for c in self.frame.columns if c not in skip]

    def prepare(self, pipeline: PreprocessingPipeline | None) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
        """Apply row-level operations, then split off the target."""
        working = self.frame
        row_log: list[str] = []
        if pipeline is not None:
            working, row_log = pipeline.apply_row_steps(working)

        target = None
        if self.objective.target and self.objective.target in working.columns:
            target = working[self.objective.target]
            before = len(working)
            keep = target.notna()
            if not keep.all():
                working, target = working[keep], target[keep]
                row_log.append(f"Dropped {before - len(working)} row(s) with a missing target value.")
        features = [c for c in self._features() if c in working.columns]
        return working[features], target, row_log

    def _split(self, X: pd.DataFrame, y: pd.Series | None):
        from sklearn.model_selection import train_test_split

        if self._split_cache is not None:
            return self._split_cache
        if y is None:
            self._split_cache = (X, X.iloc[:0], None, None)
            return self._split_cache

        stratify = None
        if self.objective.task_type.is_classification:
            counts = y.value_counts()
            if counts.min() >= 2:
                stratify = y
        if self.objective.task_type is TaskType.TIME_SERIES_FORECAST:
            cut = int(len(X) * (1 - self.config.test_size))
            self._split_cache = (X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:])
            return self._split_cache

        self._split_cache = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=stratify,
        )
        return self._split_cache

    # -- the main call -------------------------------------------------------
    def run_model(
        self,
        model_key: str,
        pipeline: PreprocessingPipeline | None = None,
        hyperparameters: dict[str, Any] | None = None,
        tune: bool | None = None,
    ) -> ExperimentResult:
        spec = self.registry.get(model_key)
        result = ExperimentResult(
            model_key=model_key,
            model_name=spec.name,
            task_type=self.objective.task_type,
            target=self.objective.target,
            interpretability=spec.interpretability.value,
            cost=spec.cost.value,
            family=spec.family,
            random_seed=self.config.random_state,
            preprocessing=[s.describe() for s in (pipeline.active_steps if pipeline else [])],
        )
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._execute(spec, result, pipeline, hyperparameters or {}, tune)
        except Exception as exc:
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            result.notes.append(traceback.format_exc(limit=3))
        result.training_time_s = round(time.perf_counter() - started, 4)
        self.results.append(result)
        return result

    def _execute(self, spec, result, pipeline, hyperparameters, tune) -> None:
        task = self.objective.task_type
        if task is TaskType.TIME_SERIES_FORECAST:
            self._run_forecast(spec, result, hyperparameters)
        elif task is TaskType.CLUSTERING:
            self._run_clustering(spec, result, pipeline, hyperparameters)
        elif task is TaskType.ANOMALY_DETECTION:
            self._run_anomaly(spec, result, pipeline, hyperparameters)
        elif task.is_supervised:
            self._run_supervised(spec, result, pipeline, hyperparameters, tune)
        else:
            result.status = "skipped"
            result.error = f"The experiment engine does not run {task.value} tasks directly."

    # -- supervised ----------------------------------------------------------
    def _run_supervised(self, spec: ModelSpec, result: ExperimentResult,
                        pipeline, hyperparameters, tune) -> None:
        from sklearn.pipeline import Pipeline

        X, y, row_log = self.prepare(pipeline)
        result.notes.extend(row_log)
        if y is None:
            raise ValueError(f"'{self.objective.target}' is not present, so there is nothing to predict.")
        if len(X) < 10:
            raise ValueError(f"Only {len(X)} usable rows remain after cleaning — too few to fit anything.")

        result.features = list(X.columns)
        X_train, X_test, y_train, y_test = self._split(X, y)
        result.n_train, result.n_test = len(X_train), len(X_test)

        estimator = spec.build(**hyperparameters)
        pre = pipeline.build_sklearn_pipeline() if pipeline else None
        model = Pipeline(list(pre.steps) + [("model", estimator)]) if pre else Pipeline([("model", estimator)])

        should_tune = self.config.tune if tune is None else tune
        if should_tune and spec.search_space():
            model, best_params = self._tune(model, spec, X_train, y_train)
            result.hyperparameters = best_params
            result.notes.append(f"Hyper-parameters tuned by {self.config.tuning_method} search.")
        else:
            result.hyperparameters = {**spec.default_params(), **hyperparameters}

        model.fit(X_train, y_train)
        self._fitted[result.id] = model

        try:
            transformed = model[:-1].transform(X_train) if len(model.steps) > 1 else X_train
            result.n_features_out = int(np.asarray(transformed).shape[1])
        except Exception:
            result.n_features_out = X_train.shape[1]

        labels = sorted(pd.unique(y)) if self.objective.task_type.is_classification else None
        result.train_scores = self._score(model, X_train, y_train, labels)
        if len(X_test):
            result.test_scores = self._score(model, X_test, y_test, labels)

        if self.config.cross_validate:
            result.validation = self._cross_validate(model, X, y)

        self._flag_overfitting(result)
        self._flag_instability(result)

    def _score(self, model, X, y, labels) -> dict[str, float]:
        task = self.objective.task_type
        predictions = model.predict(X)
        if task.is_classification:
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                except Exception:
                    proba = None
            return M.classification_metrics(y, predictions, proba, labels)
        return M.regression_metrics(y, predictions)

    def _cross_validate(self, model, X, y) -> ValidationResult:
        from sklearn.base import clone
        from sklearn.model_selection import (
            KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit,
        )

        settings = self.config.validation or {}
        strategy = settings.get("strategy", "kfold")
        n_splits = int(settings.get("n_splits", 5))
        n_splits = max(2, min(n_splits, len(X) // 2))
        seed = int(settings.get("random_state", self.config.random_state))

        if self.objective.task_type.is_classification:
            minimum_class = int(pd.Series(y).value_counts().min())
            n_splits = max(2, min(n_splits, minimum_class))

        splitter = {
            "kfold": lambda: KFold(n_splits=n_splits, shuffle=True, random_state=seed),
            "stratified_kfold": lambda: StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            "repeated_kfold": lambda: RepeatedKFold(n_splits=n_splits, n_repeats=int(settings.get("n_repeats", 3)), random_state=seed),
            "repeated_stratified_kfold": lambda: RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=int(settings.get("n_repeats", 3)), random_state=seed),
            "time_series_split": lambda: TimeSeriesSplit(n_splits=n_splits),
        }.get(strategy, lambda: KFold(n_splits=n_splits, shuffle=True, random_state=seed))()

        labels = sorted(pd.unique(y)) if self.objective.task_type.is_classification else None
        fold_scores: dict[str, list[float]] = {}
        n_folds = 0
        for train_idx, test_idx in splitter.split(X, y):
            fold_model = clone(model)
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            try:
                # Every fitted transformation is refitted here on the training
                # fold only — this is the leakage guarantee in practice.
                fold_model.fit(X_tr, y_tr)
                scores = self._score(fold_model, X_te, y_te, labels)
            except Exception:
                continue
            for name, value in scores.items():
                fold_scores.setdefault(name, []).append(value)
            n_folds += 1

        validation = ValidationResult(strategy=strategy, n_splits=n_folds, fold_scores=fold_scores)
        for name, values in fold_scores.items():
            finite = [v for v in values if np.isfinite(v)]
            if finite:
                validation.mean_scores[name] = float(np.mean(finite))
                validation.std_scores[name] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        return validation

    def _tune(self, model, spec: ModelSpec, X, y):
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        space = {f"model__{k}": v for k, v in spec.search_space().items()}
        if not space:
            return model, spec.default_params()
        scoring = M.sklearn_scoring_name(self.config.scoring_metrics[0]) if self.config.scoring_metrics else None
        common = {"cv": 3, "n_jobs": -1, "scoring": scoring, "error_score": "raise"}
        if self.config.tuning_method == "grid":
            search = GridSearchCV(model, space, **common)
        else:
            search = RandomizedSearchCV(
                model, space, n_iter=self.config.tuning_iterations,
                random_state=self.config.random_state, **common,
            )
        search.fit(X, y)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        return search.best_estimator_, best

    # -- clustering ----------------------------------------------------------
    def _run_clustering(self, spec: ModelSpec, result: ExperimentResult, pipeline, hyperparameters) -> None:
        X, _, row_log = self.prepare(pipeline)
        result.notes.extend(row_log)
        result.features = list(X.columns)
        if len(X) < 10:
            raise ValueError(f"Only {len(X)} rows available — too few to cluster meaningfully.")

        pre = pipeline.build_sklearn_pipeline() if pipeline else None
        matrix = pre.fit_transform(X) if pre else X
        matrix = np.asarray(matrix, dtype=float)
        result.n_train = len(matrix)
        result.n_features_out = matrix.shape[1]

        estimator = spec.build(**hyperparameters)
        labels = estimator.fit_predict(matrix) if hasattr(estimator, "fit_predict") else estimator.fit(matrix).labels_
        result.hyperparameters = {**spec.default_params(), **hyperparameters}
        result.test_scores = M.clustering_metrics(matrix, labels)
        result.extras["labels"] = np.asarray(labels).tolist()
        result.extras["row_index"] = list(X.index)
        self._fitted[result.id] = (pre, estimator)

        n_clusters = result.test_scores.get("n_clusters", 0)
        if n_clusters < 2:
            result.warnings.append(
                "Fewer than two clusters were found — with these settings the algorithm sees one "
                "undifferentiated group, or classified almost everything as noise."
            )
        if result.test_scores.get("noise_share", 0) > 0.5:
            result.warnings.append(
                f"{result.test_scores['noise_share']:.0%} of rows were labelled noise. "
                "The density threshold is probably too strict for this data."
            )
        imbalance = result.test_scores.get("size_imbalance_ratio", 1)
        if imbalance > 20:
            result.warnings.append(
                f"The largest cluster is {imbalance:.0f}× the smallest. Tiny clusters are often "
                "outlier pockets rather than usable segments."
            )

    # -- anomaly detection ---------------------------------------------------
    def _run_anomaly(self, spec: ModelSpec, result: ExperimentResult, pipeline, hyperparameters) -> None:
        X, _, row_log = self.prepare(pipeline)
        result.notes.extend(row_log)
        result.features = list(X.columns)
        pre = pipeline.build_sklearn_pipeline() if pipeline else None
        matrix = np.asarray(pre.fit_transform(X) if pre else X, dtype=float)
        result.n_train = len(matrix)
        result.n_features_out = matrix.shape[1]

        estimator = spec.build(**hyperparameters)
        flags = estimator.fit_predict(matrix) if hasattr(estimator, "fit_predict") else estimator.fit(matrix).predict(matrix)
        flags = np.asarray(flags)
        scores = None
        for method in ("decision_function", "score_samples"):
            if hasattr(estimator, method):
                try:
                    scores = np.asarray(getattr(estimator, method)(matrix))
                    break
                except Exception:
                    continue
        if scores is None and hasattr(estimator, "negative_outlier_factor_"):
            scores = np.asarray(estimator.negative_outlier_factor_)

        anomalies = int((flags == -1).sum())
        result.hyperparameters = {**spec.default_params(), **hyperparameters}
        result.test_scores = {
            "n_anomalies": float(anomalies),
            "anomaly_rate": float(anomalies / max(len(flags), 1)),
        }
        if scores is not None:
            result.test_scores["mean_anomaly_score"] = float(np.mean(scores))
        result.extras["flags"] = flags.tolist()
        result.extras["scores"] = scores.tolist() if scores is not None else None
        result.extras["row_index"] = list(X.index)
        self._fitted[result.id] = (pre, estimator)

        rate = result.test_scores["anomaly_rate"]
        if rate > 0.25:
            result.warnings.append(
                f"{rate:.0%} of rows were flagged. That is too many to review, and usually means the "
                "contamination setting is too high or the data is genuinely multi-modal."
            )
        if anomalies == 0:
            result.warnings.append("No anomalies were flagged at these settings — loosen the threshold to see borderline cases.")

    # -- forecasting ---------------------------------------------------------
    def _run_forecast(self, spec: ModelSpec, result: ExperimentResult, hyperparameters) -> None:
        target, time_column = self.objective.target, self.objective.time_column
        if not target:
            raise ValueError("Forecasting needs a numeric column to project forward.")

        series = self._build_series(target, time_column)
        result.features = [time_column] if time_column else []
        if len(series) < max(spec.min_rows, 10):
            raise ValueError(
                f"{spec.name} needs at least {spec.min_rows} observations; this series has {len(series)}."
            )

        horizon = self.objective.horizon or max(1, int(len(series) * self.config.test_size))
        horizon = min(horizon, max(1, len(series) // 3))
        cut = len(series) - horizon
        train, test = series.iloc[:cut], series.iloc[cut:]
        result.n_train, result.n_test = len(train), len(test)

        seasonal_period = self.objective.extras.get("seasonal_period") or _infer_season(series)
        params = dict(hyperparameters)
        if seasonal_period:
            for key in ("seasonal_period", "seasonal_periods", "period"):
                if any(h.name == key for h in spec.hyperparameters) and key not in params:
                    params[key] = seasonal_period

        forecaster = spec.build(**params)
        forecaster.fit(train)
        predictions = forecaster.predict(len(test))
        result.hyperparameters = {**spec.default_params(), **params}
        result.test_scores = M.forecast_metrics(test.to_numpy(), predictions, train.to_numpy(), seasonal_period or 1)

        # Rolling-origin back-test: the honest way to validate a forecaster.
        if self.config.cross_validate and len(series) >= 4 * horizon:
            result.validation = self._backtest(spec, params, series, horizon, seasonal_period)

        full = spec.build(**params)
        full.fit(series)
        result.extras["forecast"] = [float(v) for v in np.asarray(full.predict(horizon))]
        result.extras["horizon"] = horizon
        result.extras["seasonal_period"] = seasonal_period
        result.extras["history_index"] = [str(i) for i in series.index[-60:]]
        result.extras["history_values"] = [float(v) for v in series.to_numpy()[-60:]]
        interval = full.predict_interval(horizon) if hasattr(full, "predict_interval") else None
        if interval is not None:
            result.extras["forecast_lower"] = [float(v) for v in interval[0]]
            result.extras["forecast_upper"] = [float(v) for v in interval[1]]
        self._fitted[result.id] = full

        mase = result.test_scores.get("mase")
        if mase is not None and mase >= 1.0:
            result.warnings.append(
                f"MASE is {mase:.2f}. This model is no better than simply repeating the last observed "
                "value — do not deploy it over the naive forecast."
            )

    def _build_series(self, target: str, time_column: str | None) -> pd.Series:
        frame = self.frame
        if time_column and time_column in frame.columns:
            working = frame[[time_column, target]].dropna()
            working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
            working = working.dropna().sort_values(time_column)
            series = working.set_index(time_column)[target].astype(float)
            if series.index.has_duplicates:
                series = series.groupby(level=0).mean()
            return series
        return frame[target].dropna().astype(float).reset_index(drop=True)

    def _backtest(self, spec, params, series, horizon, seasonal_period) -> ValidationResult:
        fold_scores: dict[str, list[float]] = {}
        n_folds = 0
        max_folds = 3
        for i in range(max_folds, 0, -1):
            cut = len(series) - i * horizon
            if cut < max(spec.min_rows, 10):
                continue
            train = series.iloc[:cut]
            test = series.iloc[cut:cut + horizon]
            if len(test) < 1:
                continue
            try:
                model = spec.build(**params)
                model.fit(train)
                predictions = model.predict(len(test))
                scores = M.forecast_metrics(test.to_numpy(), predictions, train.to_numpy(), seasonal_period or 1)
            except Exception:
                continue
            for name, value in scores.items():
                fold_scores.setdefault(name, []).append(value)
            n_folds += 1

        validation = ValidationResult(strategy="rolling_origin_backtest", n_splits=n_folds, fold_scores=fold_scores)
        for name, values in fold_scores.items():
            finite = [v for v in values if np.isfinite(v)]
            if finite:
                validation.mean_scores[name] = float(np.mean(finite))
                validation.std_scores[name] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        return validation

    # -- diagnostics ---------------------------------------------------------
    def _flag_overfitting(self, result: ExperimentResult) -> None:
        metric = "r2" if self.objective.task_type is TaskType.REGRESSION else "f1"
        train = result.train_scores.get(metric)
        held_out = result.validation.mean_scores.get(metric, result.test_scores.get(metric))
        if train is None or held_out is None or not np.isfinite(train) or not np.isfinite(held_out):
            return
        gap = float(train - held_out)
        result.overfitting_gap = round(gap, 4)
        scale = max(abs(train), 1e-9)
        if gap / scale > 0.25 and gap > 0.1:
            result.warnings.append(
                f"Overfitting: {M.METRIC_LABELS.get(metric, metric)} is {train:.3f} on training data but "
                f"{held_out:.3f} on unseen data. The model has memorised patterns that do not generalise."
            )
        elif held_out - train > 0.1:
            result.notes.append(
                "Held-out performance exceeds training performance — usually a sign of heavy "
                "regularisation, or an unlucky split. Worth a second look."
            )

    def _flag_instability(self, result: ExperimentResult) -> None:
        for metric in ("r2", "f1", "roc_auc", "rmse", "mase"):
            if metric not in result.validation.mean_scores:
                continue
            mean = result.validation.mean_scores[metric]
            # A near-zero mean makes the coefficient of variation explode without
            # meaning anything, so bounded metrics get an absolute floor instead.
            if metric in {"r2", "f1", "roc_auc"} and abs(mean) < 0.05:
                break
            cv = result.validation.stability(metric)
            if np.isfinite(cv) and cv > 0.3:
                result.warnings.append(
                    f"Unstable: {M.METRIC_LABELS.get(metric, metric)} varies by "
                    f"{cv:.0%} across folds ({result.validation.mean_scores[metric]:.3f} "
                    f"± {result.validation.std_scores[metric]:.3f}). A single reported score would be misleading."
                )
            break

    # -- access --------------------------------------------------------------
    def fitted_model(self, result_id: str):
        return self._fitted.get(result_id)

    def run_many(
        self,
        model_keys: list[str],
        pipelines: dict[str, PreprocessingPipeline] | PreprocessingPipeline | None = None,
        on_progress=None,
    ) -> list[ExperimentResult]:
        """Run a whole tournament, reporting progress as it goes."""
        out: list[ExperimentResult] = []
        for i, key in enumerate(model_keys):
            pipeline = pipelines.get(key) if isinstance(pipelines, dict) else pipelines
            if on_progress:
                on_progress(i, len(model_keys), key, "running")
            result = self.run_model(key, pipeline)
            out.append(result)
            if on_progress:
                on_progress(i + 1, len(model_keys), key, result.status)
        return out

    def environment(self) -> dict[str, str]:
        """Library versions, recorded so a result can be reproduced later."""
        import sklearn

        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        }


def _infer_season(series: pd.Series) -> int | None:
    """Guess the seasonal period from the index frequency."""
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 4:
        return None
    try:
        median_days = float(pd.Series(index).diff().dropna().median() / pd.Timedelta(days=1))
    except Exception:
        return None
    if median_days <= 0:
        return None
    if median_days < 1.5:
        return 7
    if median_days < 9:
        return 52 if len(series) >= 104 else None
    if median_days < 45:
        return 12
    if median_days < 135:
        return 4
    return None
