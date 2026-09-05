"""Time-series structure: trend, seasonality, stationarity and breaks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.statistics.compat import adf_test, kpss_test


def analyse_series(series: pd.Series, seasonal_period: int | None = None) -> dict[str, Any]:
    """Everything worth knowing about a series before forecasting it."""
    values = pd.Series(series).dropna().astype(float)
    n = len(values)
    out: dict[str, Any] = {
        "n_observations": n,
        "start": str(values.index[0]) if n else None,
        "end": str(values.index[-1]) if n else None,
        "mean": float(values.mean()) if n else None,
        "std": float(values.std(ddof=1)) if n > 1 else None,
        "issues": [],
        "findings": [],
    }
    if n < 10:
        out["issues"].append(f"Only {n} observations — far too few to identify trend or seasonality.")
        return out

    out["trend"] = detect_trend(values)
    out["stationarity"] = test_stationarity(values)
    period = seasonal_period or infer_seasonal_period(values)
    out["seasonal_period"] = period
    if period and n >= 2 * period:
        out["seasonality"] = detect_seasonality(values, period)
        out["decomposition"] = decompose(values, period)
    else:
        out["seasonality"] = {
            "detected": False,
            "reason": (
                f"Need at least two full cycles to test for seasonality; the series has {n} "
                f"observations{f' and an inferred period of {period}' if period else ' and no inferred period'}."
            ),
        }
    out["structural_breaks"] = detect_structural_breaks(values)
    out["gaps"] = detect_gaps(values)
    out["autocorrelation"] = autocorrelation_summary(values)
    _summarise(out)
    return out


def detect_trend(values: pd.Series) -> dict[str, Any]:
    """Mann-Kendall style test via Spearman rank correlation against time."""
    from scipy import stats

    index = np.arange(len(values))
    correlation, p_value = stats.spearmanr(index, values.to_numpy())
    slope, intercept = np.polyfit(index, values.to_numpy(), 1)
    total_change = float(slope * (len(values) - 1))
    mean = float(values.mean())
    return {
        "direction": "increasing" if correlation > 0 else "decreasing" if correlation < 0 else "flat",
        "rank_correlation": float(correlation),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "slope_per_period": float(slope),
        "total_change": total_change,
        "pct_change_over_series": round(100 * total_change / mean, 2) if abs(mean) > 1e-9 else None,
        "interpretation": (
            f"The series {'rises' if correlation > 0 else 'falls'} over time by roughly "
            f"{abs(slope):,.4g} per period ({abs(total_change):,.4g} in total)."
            if p_value < 0.05 else "No statistically significant trend over the observed period."
        ),
    }


def test_stationarity(values: pd.Series) -> dict[str, Any]:
    """ADF and KPSS together — they test opposite nulls, so agreement matters."""
    adf = adf_test(values)
    kpss_result = kpss_test(values)
    adf_stationary = bool(adf.get("usable") and adf["p_value"] < 0.05)
    kpss_stationary = bool(kpss_result.get("usable") and kpss_result["p_value"] >= 0.05)

    if adf_stationary and kpss_stationary:
        verdict, action = "stationary", "No differencing needed before ARIMA-style models."
    elif not adf_stationary and not kpss_stationary:
        verdict, action = "non-stationary", "Difference the series once, then re-test."
    elif adf_stationary and not kpss_stationary:
        verdict, action = "trend-stationary", "Remove the deterministic trend rather than differencing."
    else:
        verdict, action = "difference-stationary", "Differencing is the right treatment here."

    return {
        "adf": adf,
        "kpss": kpss_result,
        "verdict": verdict,
        "recommended_action": action,
        "interpretation": (
            f"ADF {'rejects' if adf_stationary else 'does not reject'} a unit root; KPSS "
            f"{'supports' if kpss_stationary else 'rejects'} stationarity. Taken together the series "
            f"looks {verdict}. {action}"
        ),
    }


def infer_seasonal_period(values: pd.Series) -> int | None:
    """Guess the season length from the index, falling back to autocorrelation."""
    index = values.index
    if isinstance(index, pd.DatetimeIndex) and len(index) > 3:
        median_days = float(pd.Series(index).diff().dropna().median() / pd.Timedelta(days=1))
        if median_days > 0:
            if median_days < 1.5:
                return 7
            if median_days < 9:
                return 52 if len(values) >= 104 else None
            if median_days < 45:
                return 12
            if median_days < 135:
                return 4
            return None
    # No usable index: look for the strongest autocorrelation peak.
    max_lag = min(len(values) // 3, 60)
    if max_lag < 4:
        return None
    series = values - values.mean()
    correlations = [
        float(np.corrcoef(series[:-lag], series[lag:])[0, 1]) if lag < len(series) else 0.0
        for lag in range(2, max_lag)
    ]
    if not correlations:
        return None
    best = int(np.argmax(correlations)) + 2
    return best if correlations[best - 2] > 0.3 else None


def detect_seasonality(values: pd.Series, period: int) -> dict[str, Any]:
    """How strong is the repeating pattern, and which position peaks?

    The series is detrended first. A trend of any size dominates the variance
    and will hide a real seasonal pattern from a raw comparison of positions —
    which is exactly the case where knowing about the seasonality matters most.
    """
    raw = values.to_numpy(dtype=float)
    index = np.arange(len(raw))
    slope, intercept = np.polyfit(index, raw, 1)
    detrended = raw - (slope * index + intercept)

    positions = index % period
    frame = pd.DataFrame({"value": detrended, "position": positions})
    means = frame.groupby("position")["value"].mean()
    overall_variance = float(np.var(detrended, ddof=1))
    seasonal_variance = float(means.var(ddof=1)) if len(means) > 1 else 0.0
    strength = float(seasonal_variance / overall_variance) if overall_variance > 0 else 0.0

    from scipy import stats

    groups = [g["value"].to_numpy() for _, g in frame.groupby("position")]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        statistic, p_value = stats.f_oneway(*groups)
    else:
        statistic, p_value = float("nan"), float("nan")

    return {
        "detected": bool(np.isfinite(p_value) and p_value < 0.05 and strength > 0.05),
        "period": period,
        "strength": round(strength, 4),
        "f_statistic": float(statistic),
        "p_value": float(p_value),
        "peak_position": int(means.idxmax()),
        "trough_position": int(means.idxmin()),
        "seasonal_means": {int(k): float(v) for k, v in means.items()},
        "amplitude": float(means.max() - means.min()),
        "detrended": True,
        "interpretation": (
            f"After removing the trend, a repeating pattern of length {period} explains about "
            f"{strength:.1%} of the remaining variation, "
            f"peaking at position {means.idxmax()} and bottoming at position {means.idxmin()} "
            f"(swing of {means.max() - means.min():,.4g})."
            if np.isfinite(p_value) and p_value < 0.05
            else f"No statistically reliable seasonal pattern at period {period}."
        ),
    }


def decompose(values: pd.Series, period: int, model: str = "additive") -> dict[str, Any]:
    """Split the series into trend, seasonal and residual components."""
    try:
        from statsmodels.tsa.seasonal import STL, seasonal_decompose
    except ImportError:
        return {"supported": False, "reason": "statsmodels is not installed."}
    if len(values) < 2 * period:
        return {"supported": False, "reason": f"Need at least {2 * period} observations for period {period}."}
    try:
        if len(values) >= 3 * period:
            result = STL(values.to_numpy(), period=period, robust=True).fit()
            method = "STL (robust)"
            trend, seasonal, residual = result.trend, result.seasonal, result.resid
        else:
            result = seasonal_decompose(values.to_numpy(), period=period, model=model, extrapolate_trend="freq")
            method = f"classical ({model})"
            trend, seasonal, residual = result.trend, result.seasonal, result.resid
    except Exception as exc:
        return {"supported": False, "reason": str(exc)}

    residual_clean = np.asarray(residual)[~np.isnan(residual)]
    total_variance = float(np.var(values.to_numpy()))
    return {
        "supported": True,
        "method": method,
        "period": period,
        "trend": [float(v) for v in np.asarray(trend)],
        "seasonal": [float(v) for v in np.asarray(seasonal)],
        "residual": [float(v) for v in np.asarray(residual)],
        "trend_strength": round(float(1 - np.var(residual_clean) / max(np.var(np.asarray(trend)[~np.isnan(trend)] + residual_clean), 1e-12)), 4),
        "seasonal_share": round(float(np.var(np.asarray(seasonal)) / total_variance), 4) if total_variance > 0 else 0.0,
        "residual_share": round(float(np.var(residual_clean) / total_variance), 4) if total_variance > 0 else 0.0,
    }


def detect_structural_breaks(values: pd.Series, min_segment: int = 10) -> dict[str, Any]:
    """Find the single point where the mean shifts most (a CUSUM-style scan)."""
    array = values.to_numpy()
    n = len(array)
    if n < 3 * min_segment:
        return {"detected": False, "reason": f"Need at least {3 * min_segment} observations to look for a break."}

    best_position, best_statistic = None, 0.0
    for cut in range(min_segment, n - min_segment):
        left, right = array[:cut], array[cut:]
        pooled = np.sqrt(
            ((len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(right, ddof=1))
            / max(n - 2, 1)
        )
        if pooled <= 0:
            continue
        statistic = abs(left.mean() - right.mean()) / pooled
        if statistic > best_statistic:
            best_statistic, best_position = statistic, cut

    if best_position is None or best_statistic < 1.0:
        return {"detected": False, "strongest_statistic": round(best_statistic, 3),
                "interpretation": "The mean level looks stable across the series."}

    label = str(values.index[best_position]) if hasattr(values.index, "__getitem__") else str(best_position)
    return {
        "detected": True,
        "position": int(best_position),
        "label": label,
        "statistic": round(best_statistic, 3),
        "mean_before": float(array[:best_position].mean()),
        "mean_after": float(array[best_position:].mean()),
        "interpretation": (
            f"The level shifts around {label}: the mean moves from "
            f"{array[:best_position].mean():,.4g} to {array[best_position:].mean():,.4g}. "
            "Training a forecast across a break like this mixes two different regimes — consider "
            "modelling only the period after it, or adding an indicator for the change."
        ),
    }


def detect_gaps(values: pd.Series) -> dict[str, Any]:
    index = values.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return {"checked": False, "reason": "No datetime index, so gaps cannot be identified."}
    deltas = pd.Series(index).diff().dropna()
    median = deltas.median()
    if median <= pd.Timedelta(0):
        return {"checked": False, "reason": "Timestamps are not strictly increasing."}
    large = deltas[deltas > median * 1.5]
    expected = int((index[-1] - index[0]) / median) + 1
    return {
        "checked": True,
        "expected_periods": expected,
        "actual_periods": len(index),
        "missing_periods": max(0, expected - len(index)),
        "n_gaps": int(len(large)),
        "largest_gap": str(large.max()) if len(large) else None,
        "interpretation": (
            f"{max(0, expected - len(index))} period(s) appear to be missing across {len(large)} gap(s). "
            "Fill or resample before forecasting — gaps distort seasonality estimates."
            if expected - len(index) > 0 else "The series is evenly spaced with no missing periods."
        ),
    }


def autocorrelation_summary(values: pd.Series, max_lag: int = 24) -> dict[str, Any]:
    """Which lags carry information — a direct guide to AR order and lag features."""
    array = values.to_numpy() - values.mean()
    n = len(array)
    max_lag = min(max_lag, n // 3)
    if max_lag < 2:
        return {"supported": False}
    threshold = 1.96 / np.sqrt(n)  # approximate 95% band
    correlations = {}
    for lag in range(1, max_lag + 1):
        denominator = np.sqrt(np.sum(array[:-lag] ** 2) * np.sum(array[lag:] ** 2))
        correlations[lag] = float(np.sum(array[:-lag] * array[lag:]) / denominator) if denominator > 0 else 0.0
    significant = [lag for lag, r in correlations.items() if abs(r) > threshold]
    return {
        "supported": True,
        "acf": correlations,
        "threshold": float(threshold),
        "significant_lags": significant,
        "interpretation": (
            f"Lags {significant[:6]} carry information beyond noise — useful lag features, and an "
            f"indication of the AR order to try."
            if significant else "No lag shows autocorrelation beyond the noise band; the series looks close to white noise."
        ),
    }


def _summarise(out: dict[str, Any]) -> None:
    findings, issues = out["findings"], out["issues"]
    trend = out.get("trend", {})
    if trend.get("significant"):
        findings.append(trend["interpretation"])
    seasonality = out.get("seasonality", {})
    if seasonality.get("detected"):
        findings.append(seasonality["interpretation"])
    stationarity = out.get("stationarity", {})
    if stationarity:
        findings.append(stationarity["interpretation"])
    breaks = out.get("structural_breaks", {})
    if breaks.get("detected"):
        issues.append(breaks["interpretation"])
    gaps = out.get("gaps", {})
    if gaps.get("checked") and gaps.get("missing_periods", 0) > 0:
        issues.append(gaps["interpretation"])
    if out["n_observations"] < 24:
        issues.append(
            f"Only {out['n_observations']} observations. Seasonality cannot be estimated reliably and "
            "forecast intervals will be very wide."
        )
