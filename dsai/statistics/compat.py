"""Thin compatibility shims over library APIs that are mid-migration."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def adf_test(values: Any, autolag: str = "AIC") -> dict[str, float]:
    """Augmented Dickey-Fuller test, stable across statsmodels versions.

    statsmodels is changing ``adfuller`` from a tuple to a result object; this
    returns a plain dict either way.
    """
    from statsmodels.tsa.stattools import adfuller

    series = np.asarray(values, dtype=float)
    series = series[~np.isnan(series)]
    if len(series) < 10:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_lags": 0, "usable": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(series, autolag=autolag)
    stat = float(result[0])
    pvalue = float(result[1])
    lags = int(result[2])
    return {"statistic": stat, "p_value": pvalue, "n_lags": lags, "usable": True}


def kpss_test(values: Any, regression: str = "c") -> dict[str, float]:
    """KPSS test — null hypothesis is *stationarity*, the opposite of ADF."""
    from statsmodels.tsa.stattools import kpss

    series = np.asarray(values, dtype=float)
    series = series[~np.isnan(series)]
    if len(series) < 12:
        return {"statistic": float("nan"), "p_value": float("nan"), "usable": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series, regression=regression, nlags="auto")
    return {"statistic": float(result[0]), "p_value": float(result[1]), "usable": True}
