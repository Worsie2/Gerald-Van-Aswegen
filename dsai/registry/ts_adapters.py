"""Forecasting models behind one small, consistent interface.

Every forecaster implements::

    fit(y: pd.Series, exog: pd.DataFrame | None = None) -> self
    predict(horizon: int, exog: ... ) -> np.ndarray
    predict_interval(horizon, alpha) -> (lower, upper) | None
    fitted_values() -> np.ndarray | None

so the experiment engine can back-test any of them the same way, whether they
come from statsmodels or are machine-learning models fitted on lag features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster:
    """Minimal forecaster contract."""

    requires_stationarity: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.fitted_ = False

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self.params)

    def set_params(self, **params: Any) -> "BaseForecaster":
        self.params.update(params)
        return self

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        raise NotImplementedError

    def predict_interval(self, horizon: int, alpha: float = 0.05,
                         exog: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        return None

    def fitted_values(self) -> np.ndarray | None:
        return None


class NaiveForecaster(BaseForecaster):
    """Repeat the last value (optionally the value one season ago).

    This is the number every other forecast has to beat. A model that cannot
    beat the naive baseline is not adding value, however sophisticated it looks.
    """

    def __init__(self, seasonal_period: int | None = None, **kw: Any) -> None:
        super().__init__(seasonal_period=seasonal_period, **kw)
        self.seasonal_period = seasonal_period

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "NaiveForecaster":
        self.y_ = np.asarray(y, dtype=float)
        self.fitted_ = True
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        m = self.seasonal_period
        if m and len(self.y_) >= m:
            season = self.y_[-m:]
            return np.array([season[i % m] for i in range(horizon)])
        return np.repeat(self.y_[-1], horizon)

    def fitted_values(self) -> np.ndarray:
        m = self.seasonal_period or 1
        out = np.full(len(self.y_), np.nan)
        out[m:] = self.y_[:-m]
        return out


class DriftForecaster(BaseForecaster):
    """Straight line through the first and last observation."""

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "DriftForecaster":
        self.y_ = np.asarray(y, dtype=float)
        n = len(self.y_)
        self.slope_ = (self.y_[-1] - self.y_[0]) / max(n - 1, 1)
        self.fitted_ = True
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        return self.y_[-1] + self.slope_ * np.arange(1, horizon + 1)


class MovingAverageForecaster(BaseForecaster):
    """Flat forecast at the mean of the last `window` observations."""

    def __init__(self, window: int = 3, **kw: Any) -> None:
        super().__init__(window=window, **kw)
        self.window = window

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "MovingAverageForecaster":
        self.y_ = np.asarray(y, dtype=float)
        self.level_ = float(self.y_[-min(self.window, len(self.y_)):].mean())
        self.fitted_ = True
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        return np.repeat(self.level_, horizon)

    def fitted_values(self) -> np.ndarray:
        return pd.Series(self.y_).rolling(self.window).mean().to_numpy()


class _StatsmodelsForecaster(BaseForecaster):
    """Shared plumbing for statsmodels results objects."""

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        kwargs = {"exog": exog} if exog is not None else {}
        return np.asarray(self.results_.forecast(horizon, **kwargs), dtype=float)

    def fitted_values(self) -> np.ndarray | None:
        try:
            return np.asarray(self.results_.fittedvalues, dtype=float)
        except Exception:
            return None

    def predict_interval(self, horizon: int, alpha: float = 0.05,
                         exog: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            kwargs = {"exog": exog} if exog is not None else {}
            frame = self.results_.get_forecast(horizon, **kwargs).conf_int(alpha=alpha)
            values = np.asarray(frame, dtype=float)
            return values[:, 0], values[:, 1]
        except Exception:
            return None

    def summary_text(self) -> str:
        try:
            return str(self.results_.summary())
        except Exception:
            return ""


class ExponentialSmoothingForecaster(_StatsmodelsForecaster):
    """Holt / Holt-Winters exponential smoothing."""

    def __init__(self, trend: str | None = "add", seasonal: str | None = None,
                 seasonal_periods: int | None = None, damped_trend: bool = False, **kw: Any) -> None:
        super().__init__(trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
                         damped_trend=damped_trend, **kw)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ExponentialSmoothingForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        p = self.params
        seasonal = p.get("seasonal")
        periods = p.get("seasonal_periods")
        # Holt-Winters needs at least two full seasons to estimate seasonality.
        if seasonal and (not periods or len(y) < 2 * periods):
            seasonal, periods = None, None
        model = ExponentialSmoothing(
            np.asarray(y, dtype=float),
            trend=p.get("trend"),
            seasonal=seasonal,
            seasonal_periods=periods,
            damped_trend=p.get("damped_trend", False),
            initialization_method="estimated",
        )
        self.results_ = model.fit(optimized=True)
        self.fitted_ = True
        return self


class SimpleExponentialSmoothingForecaster(ExponentialSmoothingForecaster):
    def __init__(self, **kw: Any) -> None:
        super().__init__(trend=None, seasonal=None, **kw)


class ARIMAForecaster(_StatsmodelsForecaster):
    """ARIMA / SARIMA via statsmodels, with an optional automatic order search."""

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: str | None = None, auto: bool = False, **kw: Any) -> None:
        super().__init__(order=order, seasonal_order=seasonal_order, trend=trend, auto=auto, **kw)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        values = np.asarray(y, dtype=float)
        order = tuple(self.params.get("order", (1, 1, 1)))
        seasonal = tuple(self.params.get("seasonal_order", (0, 0, 0, 0)))
        if self.params.get("auto"):
            order, seasonal = auto_arima_order(values, seasonal_period=seasonal[3] if seasonal[3] else None)
            self.selected_order_ = (order, seasonal)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results_ = ARIMA(
                values, order=order, seasonal_order=seasonal,
                trend=self.params.get("trend"), exog=exog,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit()
        self.order_ = order
        self.seasonal_order_ = seasonal
        self.fitted_ = True
        return self


class SARIMAXForecaster(ARIMAForecaster):
    """SARIMA with exogenous regressors."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), **kw: Any) -> None:
        super().__init__(order=order, seasonal_order=seasonal_order, **kw)


class VARForecaster(BaseForecaster):
    """Vector Autoregression for several interdependent series at once."""

    def __init__(self, maxlags: int = 5, target_index: int = 0, **kw: Any) -> None:
        super().__init__(maxlags=maxlags, target_index=target_index, **kw)

    def fit(self, y: pd.DataFrame, exog: pd.DataFrame | None = None) -> "VARForecaster":
        from statsmodels.tsa.api import VAR

        frame = y if isinstance(y, pd.DataFrame) else pd.DataFrame({"y": y})
        if frame.shape[1] < 2:
            raise ValueError("VAR needs at least two series; give it multiple numeric columns.")
        self.columns_ = list(frame.columns)
        self.data_ = frame.astype(float)
        maxlags = min(self.params["maxlags"], max(1, len(frame) // (frame.shape[1] * 3)))
        self.results_ = VAR(self.data_.to_numpy()).fit(maxlags=maxlags, ic="aic")
        self.fitted_ = True
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        forecast = self.results_.forecast(self.data_.to_numpy()[-self.results_.k_ar:], steps=horizon)
        return np.asarray(forecast)[:, self.params.get("target_index", 0)]

    def forecast_all(self, horizon: int) -> pd.DataFrame:
        forecast = self.results_.forecast(self.data_.to_numpy()[-self.results_.k_ar:], steps=horizon)
        return pd.DataFrame(forecast, columns=self.columns_)


class UnobservedComponentsForecaster(_StatsmodelsForecaster):
    """State-space structural model: explicit level, trend and seasonal components."""

    def __init__(self, level: str = "local linear trend", seasonal: int | None = None, **kw: Any) -> None:
        super().__init__(level=level, seasonal=seasonal, **kw)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "UnobservedComponentsForecaster":
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        import warnings

        seasonal = self.params.get("seasonal")
        if seasonal and len(y) < 2 * seasonal:
            seasonal = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results_ = UnobservedComponents(
                np.asarray(y, dtype=float),
                level=self.params.get("level", "local linear trend"),
                seasonal=seasonal,
                exog=exog,
            ).fit(disp=False)
        self.fitted_ = True
        return self

    def components(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in ("level", "trend", "seasonal"):
            try:
                series = getattr(self.results_, name, None)
                if series is not None:
                    out[name] = np.asarray(series.smoothed)
            except Exception:
                continue
        return out


class ThetaForecaster(_StatsmodelsForecaster):
    """The Theta method — a strong, simple benchmark that won the M3 competition."""

    def __init__(self, period: int | None = None, deseasonalize: bool = True, **kw: Any) -> None:
        super().__init__(period=period, deseasonalize=deseasonalize, **kw)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ThetaForecaster":
        from statsmodels.tsa.forecasting.theta import ThetaModel
        import warnings

        period = self.params.get("period")
        values = pd.Series(np.asarray(y, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deseason = bool(self.params.get("deseasonalize", True)) and bool(period) and len(values) >= 2 * period
            self.model_ = ThetaModel(values, period=period or 1, deseasonalize=deseason)
            self.results_ = self.model_.fit()
        self.fitted_ = True
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        return np.asarray(self.results_.forecast(horizon), dtype=float)

    def predict_interval(self, horizon: int, alpha: float = 0.05, exog=None):
        try:
            frame = self.results_.prediction_intervals(horizon, alpha=alpha)
            values = np.asarray(frame, dtype=float)
            return values[:, 0], values[:, 1]
        except Exception:
            return None

    def fitted_values(self) -> np.ndarray | None:
        return None


class MLForecaster(BaseForecaster):
    """Turn forecasting into supervised learning on lag and calendar features.

    This is how gradient boosting, random forests and linear models are used for
    forecasting here: build lags of the target, fit a regressor, then roll the
    prediction forward one step at a time.
    """

    def __init__(self, estimator: Any = None, n_lags: int = 12, seasonal_period: int | None = None,
                 add_time_features: bool = True, **kw: Any) -> None:
        super().__init__(n_lags=n_lags, seasonal_period=seasonal_period,
                         add_time_features=add_time_features, **kw)
        self.estimator = estimator
        self.n_lags = n_lags
        self.seasonal_period = seasonal_period
        self.add_time_features = add_time_features

    def _lag_names(self) -> list[int]:
        lags = list(range(1, self.n_lags + 1))
        if self.seasonal_period and self.seasonal_period not in lags:
            lags.append(self.seasonal_period)
        return lags

    def _design(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lags = self._lag_names()
        max_lag = max(lags)
        rows, targets = [], []
        for t in range(max_lag, len(values)):
            row = [values[t - lag] for lag in lags]
            if self.add_time_features:
                row.extend(self._time_features(t))
            rows.append(row)
            targets.append(values[t])
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def _time_features(self, t: int) -> list[float]:
        feats = [float(t)]
        m = self.seasonal_period
        if m:
            feats.extend([np.sin(2 * np.pi * (t % m) / m), np.cos(2 * np.pi * (t % m) / m)])
        return feats

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "MLForecaster":
        from sklearn.ensemble import RandomForestRegressor

        values = np.asarray(y, dtype=float)
        max_lag = max(self._lag_names())
        if len(values) <= max_lag + 5:
            raise ValueError(
                f"Need more than {max_lag + 5} observations to fit {max_lag} lags; got {len(values)}. "
                "Reduce n_lags or use a simpler forecaster."
            )
        X, target = self._design(values)
        self.model_ = self.estimator if self.estimator is not None else RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
        self.model_.fit(X, target)
        self.history_ = values.copy()
        self.n_train_ = len(values)
        self.fitted_ = True
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        lags = self._lag_names()
        history = list(self.history_)
        out = []
        for step in range(horizon):
            t = self.n_train_ + step
            row = [history[-lag] for lag in lags]
            if self.add_time_features:
                row.extend(self._time_features(t))
            value = float(self.model_.predict(np.asarray([row], dtype=float))[0])
            out.append(value)
            history.append(value)
        return np.asarray(out)

    def feature_importance(self) -> dict[str, float] | None:
        importances = getattr(self.model_, "feature_importances_", None)
        if importances is None:
            return None
        names = [f"lag_{lag}" for lag in self._lag_names()]
        if self.add_time_features:
            names.append("time_index")
            if self.seasonal_period:
                names.extend(["season_sin", "season_cos"])
        return dict(zip(names, [float(v) for v in importances]))


def make_ml_forecaster(estimator_path: str, n_lags: int = 12, seasonal_period: int | None = None,
                       **estimator_params: Any) -> MLForecaster:
    """Build an :class:`MLForecaster` around any sklearn-style regressor."""
    import importlib

    module_path, _, attr = estimator_path.partition(":")
    cls = getattr(importlib.import_module(module_path), attr)
    return MLForecaster(estimator=cls(**estimator_params), n_lags=n_lags, seasonal_period=seasonal_period)


# --------------------------------------------------------------------------
# order selection and diagnostics
# --------------------------------------------------------------------------

def auto_arima_order(
    values: np.ndarray,
    seasonal_period: int | None = None,
    max_p: int = 3,
    max_q: int = 3,
    max_d: int = 2,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Small AIC grid search over ARIMA orders.

    Deliberately modest in range: an exhaustive search on a short series finds
    noise, not signal.
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    d = _differencing_order(values, max_d=max_d)
    best = (np.inf, (1, d, 1), (0, 0, 0, 0))
    seasonal_orders = [(0, 0, 0, 0)]
    if seasonal_period and len(values) >= 3 * seasonal_period:
        seasonal_orders += [(1, 0, 0, seasonal_period), (0, 1, 1, seasonal_period), (1, 1, 1, seasonal_period)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                for seasonal in seasonal_orders:
                    try:
                        res = ARIMA(values, order=(p, d, q), seasonal_order=seasonal,
                                    enforce_stationarity=False, enforce_invertibility=False).fit()
                        if np.isfinite(res.aic) and res.aic < best[0]:
                            best = (res.aic, (p, d, q), seasonal)
                    except Exception:
                        continue
    return best[1], best[2]


def _differencing_order(values: np.ndarray, max_d: int = 2, alpha: float = 0.05) -> int:
    """How many differences are needed before the ADF test rejects a unit root."""
    from dsai.statistics.compat import adf_test

    series = np.asarray(values, dtype=float)
    for d in range(max_d + 1):
        if len(series) < 10:
            return d
        try:
            pvalue = adf_test(series)["p_value"]
        except Exception:
            return d
        if not np.isfinite(pvalue):
            return d
        if pvalue < alpha:
            return d
        series = np.diff(series)
    return max_d
