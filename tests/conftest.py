"""Shared fixtures. Every dataset here has a structure the tests can assert against."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.registry.base import load_builtin_models


@pytest.fixture(scope="session", autouse=True)
def registry():
    return load_builtin_models()


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    """Spend is genuinely driven by meters, employees and tier, plus noise."""
    rng = np.random.default_rng(42)
    n = 400
    tier = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    frame = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "region": rng.choice(["Gauteng", "Western Cape", "KwaZulu-Natal"], n),
        "service_tier": tier,
        "employees": rng.lognormal(3, 0.8, n).round(),
        "meters_installed": rng.poisson(12, n),
    })
    frame["annual_spend"] = (
        18_000 + 900 * frame.meters_installed + 40 * frame.employees
        + pd.Series(tier).map({"basic": 0, "standard": 15_000, "premium": 60_000}).to_numpy()
        + rng.normal(0, 8_000, n)
    ).clip(1_000)
    return frame


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    """Roughly 20% churn, driven by tenure, charge and support calls."""
    rng = np.random.default_rng(43)
    n = 500
    frame = pd.DataFrame({
        "tenure_months": rng.exponential(20, n).round(1),
        "monthly_charge": rng.normal(450, 120, n).clip(80),
        "support_calls": rng.poisson(1.5, n),
        "plan": rng.choice(["prepaid", "contract"], n),
    })
    logit = -2.0 + 0.004 * frame.monthly_charge - 0.05 * frame.tenure_months + 0.3 * frame.support_calls
    frame["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


@pytest.fixture
def cluster_frame() -> pd.DataFrame:
    """Three genuinely separate groups — the correct answer is k = 3."""
    rng = np.random.default_rng(44)
    blocks = [
        pd.DataFrame({
            "spend": rng.normal(centre, centre * 0.15, size),
            "visits": rng.poisson(visits, size),
            "age": rng.normal(age, 6, size),
        })
        for size, centre, visits, age in [(150, 4_000, 4, 30), (150, 15_000, 18, 45), (100, 45_000, 34, 58)]
    ]
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def series_frame() -> pd.DataFrame:
    """Trend plus a twelve-month seasonal cycle."""
    rng = np.random.default_rng(45)
    periods = 120
    t = np.arange(periods)
    return pd.DataFrame({
        "month": pd.date_range("2016-01-31", periods=periods, freq="ME"),
        "sales": (250_000 + 1_800 * t + 42_000 * np.sin(2 * np.pi * t / 12)
                  + rng.normal(0, 12_000, periods)).round(2),
    })


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """Deliberately broken: duplicates, missingness, a constant, an id and a leak."""
    rng = np.random.default_rng(46)
    n = 200
    frame = pd.DataFrame({
        "row_id": range(n),
        "constant_column": "same",
        "mostly_missing": [np.nan] * 180 + list(rng.normal(size=20)),
        "value": rng.lognormal(3, 1, n),
        "category": rng.choice(["a", "b", "c"], n),
    })
    frame["value_leak"] = frame.value * 1.0001
    frame.loc[frame.sample(30, random_state=1).index, "value"] = np.nan
    return pd.concat([frame, frame.head(15)], ignore_index=True)


@pytest.fixture(scope="session")
def regression_run():
    """One completed regression analysis, plus the typed frame it ran on.

    Session-scoped: it trains real models, and every test that needs a finished
    run needs the same one.
    """
    from dsai.app.samples import build_sample
    from dsai.core.schema import BusinessContext
    from dsai.engines.orchestrator import AIDataScientist, RunSettings

    frame, source = build_sample("water_customers")
    engine = AIDataScientist()
    run = engine.analyse(
        frame, "water_customers",
        BusinessContext(description="water instrumentation customers", currency="ZAR"),
        RunSettings(max_models=3, time_budget="fast"), source,
    )
    # The fitted model belongs to the engine that trained it, so the engine has
    # to travel with the run for anything that wants to score new rows.
    run._engine = engine
    return run, engine._typed_frame
