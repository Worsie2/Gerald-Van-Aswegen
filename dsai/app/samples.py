"""Synthetic sample datasets, each with a deliberately known structure.

They exist so the platform can be tried without hunting for a file, and so the
results can be checked against a ground truth the user already knows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dsai.dataio.loaders import DataSource

SAMPLES: dict[str, dict[str, str]] = {
    "water_customers": {
        "label": "Water instrumentation customers (regression)",
        "description": (
            "600 South African business customers with annual spend. Spend is genuinely driven by "
            "service tier, meters installed and headcount, plus noise — so you can check whether the "
            "platform recovers the real drivers."
        ),
    },
    "customer_churn": {
        "label": "Customer churn (imbalanced classification)",
        "description": (
            "1 200 subscribers, roughly 15% of whom churned. Deliberately imbalanced, so accuracy "
            "will look impressive while being useless — a good test of whether the metric choice holds up."
        ),
    },
    "customer_segments": {
        "label": "Customer segments (clustering)",
        "description": (
            "900 customers drawn from three genuinely distinct groups. The correct answer is three "
            "segments; see whether the cluster-count measures agree."
        ),
    },
    "monthly_sales": {
        "label": "Monthly sales (time series)",
        "description": (
            "Ten years of monthly sales with a trend, a twelve-month seasonal cycle, and a step "
            "change partway through. The structural break is deliberate."
        ),
    },
    "product_baskets": {
        "label": "Product baskets (association rules)",
        "description": (
            "2 000 transactions with two planted associations: flow meters pull calibration kits, "
            "controllers pull cabling."
        ),
    },
    "messy_survey": {
        "label": "Messy survey data (data quality)",
        "description": (
            "400 survey responses with missing values, duplicates, a constant column, an identifier, "
            "free text and a leakage column. Built to exercise the data-quality checks."
        ),
    },
}


def build_sample(key: str) -> tuple[pd.DataFrame, DataSource]:
    builder = {
        "water_customers": _water_customers,
        "customer_churn": _customer_churn,
        "customer_segments": _customer_segments,
        "monthly_sales": _monthly_sales,
        "product_baskets": _product_baskets,
        "messy_survey": _messy_survey,
    }[key]
    frame = builder()
    source = DataSource(
        kind="dataframe", name=key, n_rows=len(frame), n_columns=int(frame.shape[1]),
        options={"generated": "synthetic sample"},
    )
    return frame, source


def _water_customers() -> pd.DataFrame:
    rng = np.random.default_rng(20)
    n = 600
    tier = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    frame = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "signup_date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "region": rng.choice(["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape"],
                             n, p=[0.4, 0.3, 0.2, 0.1]),
        "sector": rng.choice(["municipal", "mining", "agriculture", "industrial"], n),
        "service_tier": tier,
        "employees": rng.lognormal(3, 1, n).round(),
        "meters_installed": rng.poisson(12, n),
        "support_tickets": rng.poisson(3, n),
    })
    frame["annual_spend"] = (
        18_000
        + 900 * frame.meters_installed
        + 40 * frame.employees
        + pd.Series(tier).map({"basic": 0, "standard": 15_000, "premium": 60_000}).to_numpy()
        + rng.normal(0, 12_000, n)
    ).clip(1_000).round(2)
    frame.loc[frame.sample(45, random_state=1).index, "employees"] = np.nan
    return frame


def _customer_churn() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 1_200
    frame = pd.DataFrame({
        "subscriber_id": range(1, n + 1),
        "tenure_months": rng.exponential(20, n).round(1),
        "monthly_charge": rng.normal(450, 120, n).clip(80).round(2),
        "support_calls": rng.poisson(1.5, n),
        "data_usage_gb": rng.gamma(3, 8, n).round(1),
        "plan": rng.choice(["prepaid", "contract", "business"], n, p=[0.5, 0.35, 0.15]),
        "province": rng.choice(["GP", "WC", "KZN", "EC", "FS"], n),
        "autopay": rng.choice([True, False], n, p=[0.6, 0.4]),
    })
    logit = (
        -2.6
        + 0.004 * frame.monthly_charge
        - 0.045 * frame.tenure_months
        + 0.32 * frame.support_calls
        - 0.7 * frame.autopay.astype(int)
    )
    frame["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


def _customer_segments() -> pd.DataFrame:
    rng = np.random.default_rng(22)
    blocks = []
    for size, spend, visits, age, tenure in [
        (400, 4_000, 4, 31, 8), (350, 15_000, 18, 46, 30), (150, 46_000, 34, 57, 62),
    ]:
        blocks.append(pd.DataFrame({
            "annual_spend": rng.normal(spend, spend * 0.16, size).clip(200).round(2),
            "visits_per_year": rng.poisson(visits, size),
            "age": rng.normal(age, 7, size).clip(18, 85).round(),
            "tenure_months": rng.normal(tenure, 9, size).clip(1).round(),
            "products_held": rng.poisson(max(1, visits // 6), size) + 1,
        }))
    frame = pd.concat(blocks, ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
    frame.insert(0, "customer_id", [f"S{i:05d}" for i in range(1, len(frame) + 1)])
    frame["region"] = rng.choice(["Gauteng", "Western Cape", "KwaZulu-Natal"], len(frame))
    return frame


def _monthly_sales() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    periods = 120
    t = np.arange(periods)
    sales = (
        250_000 + 1_800 * t
        + 42_000 * np.sin(2 * np.pi * t / 12)
        + rng.normal(0, 14_000, periods)
    )
    sales[78:] += 90_000  # a deliberate step change, for the break detector to find
    return pd.DataFrame({
        "month": pd.date_range("2016-01-31", periods=periods, freq="ME"),
        "sales": sales.round(2),
        "marketing_spend": (sales * 0.06 + rng.normal(0, 4_000, periods)).clip(0).round(2),
        "active_reps": rng.poisson(14, periods) + 5,
    })


def _product_baskets() -> pd.DataFrame:
    rng = np.random.default_rng(24)
    items = ["flow meter", "pressure sensor", "calibration kit", "cabling",
             "controller", "data logger", "valve", "filter"]
    rows = []
    for transaction in range(1, 2_001):
        basket = set(rng.choice(items, rng.integers(1, 4), replace=False))
        if "flow meter" in basket and rng.random() < 0.72:
            basket.add("calibration kit")
        if "controller" in basket and rng.random() < 0.65:
            basket.add("cabling")
        for item in basket:
            rows.append({
                "transaction_id": transaction,
                "item": item,
                "quantity": int(rng.integers(1, 5)),
                "store": rng.choice(["Johannesburg", "Cape Town", "Durban"]),
            })
    return pd.DataFrame(rows)


def _messy_survey() -> pd.DataFrame:
    rng = np.random.default_rng(25)
    n = 400
    frame = pd.DataFrame({
        "response_id": [f"R{i:04d}" for i in range(1, n + 1)],
        "submitted_at": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "satisfaction": rng.choice(["low", "medium", "high"], n, p=[0.25, 0.45, 0.3]),
        "nps_score": rng.integers(0, 11, n),
        "age": rng.normal(41, 14, n).clip(18, 90).round(),
        "monthly_income": rng.lognormal(9.6, 0.7, n).round(2),
        "comments": rng.choice([
            "The service has been reliable and the support team responds quickly to queries",
            "Too expensive for what is offered, considering other options at the moment",
            "Generally satisfied although delivery times could be improved considerably",
            "Excellent product quality and the installation team was highly professional",
        ], n),
        "survey_version": "v2.1",
        "region": rng.choice(["Gauteng", "Western Cape", "KwaZulu-Natal", None], n, p=[.4, .3, .2, .1]),
    })
    frame.loc[frame.sample(70, random_state=4).index, "monthly_income"] = np.nan
    frame.loc[frame.sample(120, random_state=5).index, "age"] = np.nan
    # A planted leakage column: it is the target restated.
    frame["satisfaction_score_computed"] = frame.satisfaction.map({"low": 1, "medium": 2, "high": 3})
    return pd.concat([frame, frame.head(18)], ignore_index=True)
