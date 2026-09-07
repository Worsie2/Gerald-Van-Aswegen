"""Hypothesis tests, with assumptions checked and effect sizes reported.

Every test returns a :class:`TestResult` that separates three things people
routinely conflate:

* the **p-value** — how surprising the data would be if nothing were going on,
* the **effect size** — how big the difference actually is, and
* the **practical reading** — whether that size matters.

A significant p-value on 50 000 rows can describe a difference nobody would ever
act on. The result object says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import JsonMixin

ALPHA = 0.05


@dataclass
class TestResult(JsonMixin):
    test: str
    statistic: float
    p_value: float
    significant: bool = False
    alpha: float = ALPHA
    n: int = 0
    groups: list[str] = field(default_factory=list)
    effect_size: float | None = None
    effect_size_name: str = ""
    effect_interpretation: str = ""
    confidence_interval: tuple[float, float] | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)
    assumption_warnings: list[str] = field(default_factory=list)
    null_hypothesis: str = ""
    conclusion: str = ""
    practical_note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def summarise(self) -> str:
        verdict = "reject" if self.significant else "cannot reject"
        parts = [
            f"{self.test}: statistic = {self.statistic:.4f}, p = {_format_p(self.p_value)}; "
            f"at α = {self.alpha} we {verdict} the null hypothesis."
        ]
        if self.effect_size is not None:
            parts.append(
                f"Effect size ({self.effect_size_name}) = {self.effect_size:.3f} — {self.effect_interpretation}."
            )
        if self.practical_note:
            parts.append(self.practical_note)
        if self.assumption_warnings:
            parts.append("Assumption check: " + "; ".join(self.assumption_warnings))
        return " ".join(parts)


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def _practical_note(n: int, significant: bool, effect: float | None, small_threshold: float) -> str:
    if significant and effect is not None and abs(effect) < small_threshold:
        return (
            f"Statistically significant but practically small: with n = {n:,}, even a difference too "
            "small to act on will register as significant. Judge this on the effect size, not the p-value."
        )
    if not significant and n < 30:
        return (
            f"With only n = {n}, this test has little power. 'Not significant' here means "
            "'not established', not 'no effect'."
        )
    if significant:
        return "The effect is large enough to be worth acting on, subject to the assumption checks."
    return "No effect of a detectable size was found at this sample size."


# --------------------------------------------------------------------------
# assumption checks
# --------------------------------------------------------------------------

def normality_test(values: Any, name: str = "variable") -> TestResult:
    """Shapiro-Wilk for small samples, D'Agostino for larger ones."""
    from scipy import stats

    series = pd.Series(values).dropna().astype(float)
    n = len(series)
    if n < 8:
        return TestResult(
            test="Normality", statistic=float("nan"), p_value=float("nan"), n=n,
            null_hypothesis=f"'{name}' is normally distributed",
            conclusion="Too few observations (fewer than 8) to test normality meaningfully.",
        )
    if n <= 5000:
        statistic, p_value = stats.shapiro(series)
        test_name = "Shapiro-Wilk normality test"
    else:
        statistic, p_value = stats.normaltest(series)
        test_name = "D'Agostino-Pearson normality test"

    skew, kurt = float(series.skew()), float(series.kurt())
    result = TestResult(
        test=test_name, statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < ALPHA), n=n,
        null_hypothesis=f"'{name}' is drawn from a normal distribution",
        detail={"skewness": skew, "kurtosis": kurt},
    )
    if result.significant:
        result.conclusion = (
            f"'{name}' departs from normality (skew {skew:.2f}, excess kurtosis {kurt:.2f}). "
            "Prefer non-parametric tests, or transform the variable first."
        )
        if n > 5000:
            result.practical_note = (
                "At this sample size normality tests reject almost any real data. Look at the "
                "histogram and Q-Q plot rather than the p-value."
            )
    else:
        result.conclusion = f"'{name}' is consistent with a normal distribution; parametric tests are reasonable."
    return result


def homoscedasticity_test(*groups: Any, names: list[str] | None = None) -> TestResult:
    """Levene's test for equal variances — robust to non-normality."""
    from scipy import stats

    cleaned = [pd.Series(g).dropna().astype(float).to_numpy() for g in groups]
    cleaned = [g for g in cleaned if len(g) > 1]
    if len(cleaned) < 2:
        return TestResult(test="Levene", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Need at least two groups with variation to compare variances.")
    statistic, p_value = stats.levene(*cleaned, center="median")
    variances = [float(np.var(g, ddof=1)) for g in cleaned]
    result = TestResult(
        test="Levene's test for equal variances", statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < ALPHA), n=sum(len(g) for g in cleaned),
        groups=names or [f"group_{i + 1}" for i in range(len(cleaned))],
        null_hypothesis="All groups have the same variance",
        detail={"variances": variances, "ratio": max(variances) / max(min(variances), 1e-12)},
    )
    result.conclusion = (
        "Variances differ significantly between groups — use a Welch-corrected test rather than the "
        "standard one." if result.significant else
        "Variances are comparable across groups, so equal-variance tests are appropriate."
    )
    return result


# --------------------------------------------------------------------------
# comparing two groups
# --------------------------------------------------------------------------

def t_test(
    group_a: Any,
    group_b: Any,
    paired: bool = False,
    equal_variance: bool | None = None,
    names: tuple[str, str] = ("group A", "group B"),
    alpha: float = ALPHA,
) -> TestResult:
    """Compare two means, checking assumptions and reporting Cohen's d."""
    from scipy import stats

    a = pd.Series(group_a).dropna().astype(float)
    b = pd.Series(group_b).dropna().astype(float)
    if paired:
        joined = pd.concat([pd.Series(group_a), pd.Series(group_b)], axis=1).dropna()
        a, b = joined.iloc[:, 0].astype(float), joined.iloc[:, 1].astype(float)
    if len(a) < 2 or len(b) < 2:
        return TestResult(test="t-test", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Each group needs at least two observations.")

    normal_a = normality_test(a, names[0])
    normal_b = normality_test(b, names[1])
    equal_var_test = homoscedasticity_test(a, b, names=list(names)) if not paired else None
    if equal_variance is None:
        equal_variance = not (equal_var_test.significant if equal_var_test else False)

    if paired:
        statistic, p_value = stats.ttest_rel(a, b)
        test_name = "Paired t-test"
        differences = a.to_numpy() - b.to_numpy()
        effect = float(np.mean(differences) / np.std(differences, ddof=1)) if np.std(differences, ddof=1) > 0 else 0.0
    else:
        statistic, p_value = stats.ttest_ind(a, b, equal_var=equal_variance)
        test_name = "Independent t-test" + ("" if equal_variance else " (Welch)")
        effect = cohens_d(a, b)

    result = TestResult(
        test=test_name, statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=len(a) + len(b),
        groups=list(names), effect_size=effect, effect_size_name="Cohen's d",
        effect_interpretation=interpret_cohens_d(effect),
        null_hypothesis=f"The mean of {names[0]} equals the mean of {names[1]}",
        confidence_interval=mean_difference_ci(a, b, paired=paired),
        detail={
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "difference": float(a.mean() - b.mean()),
            "n_a": len(a), "n_b": len(b),
            "std_a": float(a.std(ddof=1)), "std_b": float(b.std(ddof=1)),
        },
    )
    result.assumptions = {
        "normality_a": normal_a.to_dict(), "normality_b": normal_b.to_dict(),
        "equal_variance": equal_var_test.to_dict() if equal_var_test else None,
    }
    if normal_a.significant or normal_b.significant:
        if min(len(a), len(b)) < 30:
            result.assumption_warnings.append(
                "the data is not normal and the groups are small, so the p-value is unreliable — "
                "use Mann-Whitney U instead"
            )
        else:
            result.assumption_warnings.append(
                "the data is not normal, but with these sample sizes the t-test is robust enough "
                "(central limit theorem)"
            )
    if equal_var_test and equal_var_test.significant and equal_variance:
        result.assumption_warnings.append("variances differ — a Welch correction would be safer")

    difference = result.detail["difference"]
    result.conclusion = (
        f"{names[0]} averages {result.detail['mean_a']:,.4g} against {result.detail['mean_b']:,.4g} for "
        f"{names[1]}, a difference of {difference:,.4g}. "
        + ("This difference is statistically significant." if result.significant
           else "This difference is not statistically significant.")
    )
    result.practical_note = _practical_note(result.n, result.significant, effect, 0.2)
    return result


def mann_whitney(group_a: Any, group_b: Any, names: tuple[str, str] = ("group A", "group B"),
                 alpha: float = ALPHA) -> TestResult:
    """Non-parametric alternative to the t-test: compares distributions, not means."""
    from scipy import stats

    a = pd.Series(group_a).dropna().astype(float)
    b = pd.Series(group_b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return TestResult(test="Mann-Whitney U", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Each group needs at least two observations.")
    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial correlation: a readable effect size for this test.
    effect = float(1 - (2 * statistic) / (len(a) * len(b)))
    result = TestResult(
        test="Mann-Whitney U test", statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=len(a) + len(b), groups=list(names),
        effect_size=abs(effect), effect_size_name="rank-biserial correlation",
        effect_interpretation=interpret_rank_biserial(abs(effect)),
        null_hypothesis=f"{names[0]} and {names[1]} are drawn from the same distribution",
        detail={"median_a": float(a.median()), "median_b": float(b.median()),
                "n_a": len(a), "n_b": len(b)},
    )
    result.conclusion = (
        f"Median of {names[0]} is {result.detail['median_a']:,.4g} against "
        f"{result.detail['median_b']:,.4g} for {names[1]}. "
        + ("The distributions differ significantly." if result.significant
           else "No significant difference between the distributions.")
    )
    result.practical_note = _practical_note(result.n, result.significant, abs(effect), 0.1)
    return result


def wilcoxon_signed_rank(before: Any, after: Any, alpha: float = ALPHA) -> TestResult:
    """Paired non-parametric test — the Mann-Whitney of before/after comparisons."""
    from scipy import stats

    joined = pd.concat([pd.Series(before), pd.Series(after)], axis=1).dropna()
    if len(joined) < 5:
        return TestResult(test="Wilcoxon signed-rank", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Need at least five complete pairs.")
    a, b = joined.iloc[:, 0].astype(float), joined.iloc[:, 1].astype(float)
    statistic, p_value = stats.wilcoxon(a, b)
    # after minus before, which is what "change" means to anyone reading it.
    # Computed the other way round, a fall reads as a rise — the p-value is the
    # same either way, so the mistake is invisible in everything but the wording.
    differences = b - a
    effect = float(abs(np.mean(differences)) / max(np.std(differences, ddof=1), 1e-12))
    median = float(differences.median())
    result = TestResult(
        test="Wilcoxon signed-rank test", statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=len(joined),
        effect_size=effect, effect_size_name="standardised median shift",
        effect_interpretation=interpret_cohens_d(effect),
        null_hypothesis="The paired differences are centred on zero",
        detail={"median_difference": median, "n_pairs": len(joined),
                "direction": "increase" if median > 0 else "decrease" if median < 0 else "no change"},
    )
    direction = ("a rise of" if median > 0 else "a fall of" if median < 0 else "a change of")
    result.conclusion = (
        f"Median paired change: {direction} {abs(median):,.4g} from the first measure to the "
        f"second, across {len(joined):,} pairs. "
        + ("Significant." if result.significant else "Not significant.")
    )
    result.practical_note = _practical_note(result.n, result.significant, effect, 0.2)
    return result


# --------------------------------------------------------------------------
# comparing several groups
# --------------------------------------------------------------------------

def anova(frame: pd.DataFrame, value_column: str, group_column: str, alpha: float = ALPHA) -> TestResult:
    """One-way ANOVA with eta-squared and an assumption check."""
    from scipy import stats

    working = frame[[value_column, group_column]].dropna()
    groups = [g[value_column].astype(float).to_numpy() for _, g in working.groupby(group_column, observed=True)]
    names = [str(k) for k, _ in working.groupby(group_column, observed=True)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return TestResult(test="One-way ANOVA", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Need at least two groups with more than one observation each.")

    statistic, p_value = stats.f_oneway(*groups)
    grand_mean = float(np.concatenate(groups).mean())
    ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
    ss_total = float(((np.concatenate(groups) - grand_mean) ** 2).sum())
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    levene = homoscedasticity_test(*groups, names=names)
    result = TestResult(
        test="One-way ANOVA", statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=int(sum(len(g) for g in groups)),
        groups=names, effect_size=eta_squared, effect_size_name="eta squared (η²)",
        effect_interpretation=interpret_eta_squared(eta_squared),
        null_hypothesis=f"All groups of '{group_column}' have the same mean '{value_column}'",
        assumptions={"equal_variance": levene.to_dict()},
        detail={
            "group_means": {n: float(g.mean()) for n, g in zip(names, groups)},
            "group_sizes": {n: int(len(g)) for n, g in zip(names, groups)},
            "df_between": len(groups) - 1,
            "df_within": int(sum(len(g) for g in groups) - len(groups)),
        },
    )
    if levene.significant:
        result.assumption_warnings.append(
            "group variances differ — consider Welch's ANOVA or the Kruskal-Wallis test instead"
        )
    if result.significant:
        means = result.detail["group_means"]
        highest, lowest = max(means, key=means.get), min(means, key=means.get)
        result.conclusion = (
            f"'{value_column}' differs significantly across '{group_column}'. "
            f"'{highest}' is highest at {means[highest]:,.4g} and '{lowest}' lowest at {means[lowest]:,.4g}. "
            f"Group membership explains {eta_squared:.1%} of the variation. "
            "ANOVA says the groups are not all the same; it does not say which pairs differ — "
            "run pairwise tests with a multiple-comparison correction for that."
        )
    else:
        result.conclusion = f"No significant difference in '{value_column}' across '{group_column}'."
    result.practical_note = _practical_note(result.n, result.significant, eta_squared, 0.01)
    return result


def kruskal_wallis(frame: pd.DataFrame, value_column: str, group_column: str, alpha: float = ALPHA) -> TestResult:
    """Non-parametric ANOVA: compares distributions across several groups."""
    from scipy import stats

    working = frame[[value_column, group_column]].dropna()
    grouped = list(working.groupby(group_column, observed=True))
    groups = [g[value_column].astype(float).to_numpy() for _, g in grouped]
    names = [str(k) for k, _ in grouped]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return TestResult(test="Kruskal-Wallis", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Need at least two groups.")
    statistic, p_value = stats.kruskal(*groups)
    n = int(sum(len(g) for g in groups))
    k = len(groups)
    epsilon_squared = float((statistic - k + 1) / (n - k)) if n > k else 0.0
    result = TestResult(
        test="Kruskal-Wallis H test", statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=n, groups=names,
        effect_size=max(0.0, epsilon_squared), effect_size_name="epsilon squared (ε²)",
        effect_interpretation=interpret_eta_squared(max(0.0, epsilon_squared)),
        null_hypothesis=f"All groups of '{group_column}' have the same distribution of '{value_column}'",
        detail={"group_medians": {n_: float(np.median(g)) for n_, g in zip(names, groups)}},
    )
    result.conclusion = (
        f"'{value_column}' differs significantly across '{group_column}' (distribution-based test)."
        if result.significant else f"No significant difference in '{value_column}' across '{group_column}'."
    )
    result.practical_note = _practical_note(n, result.significant, result.effect_size, 0.01)
    return result


# --------------------------------------------------------------------------
# categorical association
# --------------------------------------------------------------------------

def chi_square(frame: pd.DataFrame, column_a: str, column_b: str, alpha: float = ALPHA) -> TestResult:
    """Chi-square test of independence with Cramér's V, falling back to Fisher's exact."""
    from scipy import stats

    table = pd.crosstab(frame[column_a], frame[column_b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return TestResult(test="Chi-square", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Both variables need at least two categories.")

    statistic, p_value, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    minimum_dimension = min(table.shape) - 1
    cramers_v = float(np.sqrt(statistic / (n * minimum_dimension))) if n and minimum_dimension else 0.0
    small_expected = int((expected < 5).sum())

    result = TestResult(
        test="Chi-square test of independence", statistic=float(statistic), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=n,
        groups=[column_a, column_b], effect_size=cramers_v, effect_size_name="Cramér's V",
        effect_interpretation=interpret_cramers_v(cramers_v),
        null_hypothesis=f"'{column_a}' and '{column_b}' are independent",
        detail={"dof": int(dof), "contingency_table": table.to_dict(),
                "cells_with_expected_below_5": small_expected},
    )
    if small_expected:
        result.assumption_warnings.append(
            f"{small_expected} cell(s) have an expected count below 5, which makes the chi-square "
            "approximation unreliable — Fisher's exact test is reported below where the table is 2×2"
        )
        if table.shape == (2, 2):
            odds, fisher_p = stats.fisher_exact(table)
            result.detail["fisher_exact_p"] = float(fisher_p)
            result.detail["odds_ratio"] = float(odds)
            result.p_value = float(fisher_p)
            result.significant = bool(fisher_p < alpha)
            result.test = "Fisher's exact test (chi-square assumptions not met)"
    result.conclusion = (
        f"'{column_a}' and '{column_b}' are associated (Cramér's V = {cramers_v:.3f}, "
        f"{result.effect_interpretation})." if result.significant
        else f"No significant association between '{column_a}' and '{column_b}'."
    )
    result.practical_note = _practical_note(n, result.significant, cramers_v, 0.1)
    return result


def fisher_exact(frame: pd.DataFrame, column_a: str, column_b: str, alpha: float = ALPHA) -> TestResult:
    from scipy import stats

    table = pd.crosstab(frame[column_a], frame[column_b])
    if table.shape != (2, 2):
        return TestResult(test="Fisher's exact test", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Fisher's exact test needs a 2×2 table.")
    odds, p_value = stats.fisher_exact(table)
    return TestResult(
        test="Fisher's exact test", statistic=float(odds), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=int(table.to_numpy().sum()),
        groups=[column_a, column_b], effect_size=float(odds), effect_size_name="odds ratio",
        effect_interpretation=(
            f"the odds are {odds:.2f}× higher in the first group"
            if odds > 1 else f"the odds are {1 / max(odds, 1e-9):.2f}× lower in the first group"
        ),
        null_hypothesis=f"'{column_a}' and '{column_b}' are independent",
        conclusion=("Significant association." if p_value < alpha else "No significant association."),
    )


# --------------------------------------------------------------------------
# effect sizes and intervals
# --------------------------------------------------------------------------

def cohens_d(a: Any, b: Any) -> float:
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def interpret_cohens_d(d: float) -> str:
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible — too small to be worth acting on"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def interpret_eta_squared(eta: float) -> str:
    if eta < 0.01:
        return "negligible — group membership explains almost none of the variation"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"


def interpret_cramers_v(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "weak"
    if v < 0.5:
        return "moderate"
    return "strong"


def interpret_rank_biserial(r: float) -> str:
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "small"
    if r < 0.5:
        return "medium"
    return "large"


def mean_difference_ci(a: Any, b: Any, confidence: float = 0.95, paired: bool = False) -> tuple[float, float] | None:
    from scipy import stats

    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return None
    if paired:
        differences = a.to_numpy()[: min(len(a), len(b))] - b.to_numpy()[: min(len(a), len(b))]
        mean = float(np.mean(differences))
        se = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        dof = len(differences) - 1
    else:
        mean = float(a.mean() - b.mean())
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        dof = len(a) + len(b) - 2
    if se == 0:
        return (mean, mean)
    critical = stats.t.ppf(1 - (1 - confidence) / 2, dof)
    return (mean - critical * se, mean + critical * se)


def confidence_interval(values: Any, confidence: float = 0.95) -> tuple[float, float] | None:
    from scipy import stats

    series = pd.Series(values).dropna().astype(float)
    if len(series) < 2:
        return None
    mean = float(series.mean())
    se = float(series.sem())
    critical = stats.t.ppf(1 - (1 - confidence) / 2, len(series) - 1)
    return (mean - critical * se, mean + critical * se)


def correlation_test(x: Any, y: Any, method: str = "pearson", alpha: float = ALPHA) -> TestResult:
    from scipy import stats

    joined = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna().astype(float)
    if len(joined) < 3:
        return TestResult(test="Correlation", statistic=float("nan"), p_value=float("nan"),
                          conclusion="Need at least three paired observations.")
    a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    if method == "spearman":
        coefficient, p_value = stats.spearmanr(a, b)
    elif method == "kendall":
        coefficient, p_value = stats.kendalltau(a, b)
    else:
        coefficient, p_value = stats.pearsonr(a, b)

    from dsai.statistics.descriptive import interpret_correlation

    n = len(joined)
    result = TestResult(
        test=f"{method.title()} correlation", statistic=float(coefficient), p_value=float(p_value),
        significant=bool(p_value < alpha), alpha=alpha, n=n,
        effect_size=float(coefficient), effect_size_name=f"{method} r",
        effect_interpretation=interpret_correlation(float(coefficient)),
        null_hypothesis="The two variables are uncorrelated",
        detail={"r_squared": float(coefficient ** 2)},
    )
    result.conclusion = (
        f"r = {coefficient:.3f} ({result.effect_interpretation}); the two variables share "
        f"{coefficient ** 2:.1%} of their variation. "
        + ("Significant." if result.significant else "Not significant.")
        + " Correlation is not causation: a third variable, reverse causation or coincidence would "
          "all produce this same number."
    )
    result.practical_note = _practical_note(n, result.significant, abs(coefficient), 0.1)
    return result


def multiple_comparison_correction(p_values: list[float], method: str = "holm",
                                   alpha: float = ALPHA) -> dict[str, Any]:
    """Adjust p-values for the number of tests run.

    Running twenty tests at α = 0.05 gives roughly a 64% chance of at least one
    false positive. Any analysis that tests many things at once needs this.
    """
    values = np.asarray(p_values, dtype=float)
    n = len(values)
    if n == 0:
        return {"method": method, "adjusted": [], "significant": []}

    order = np.argsort(values)
    adjusted = np.empty(n)
    if method == "bonferroni":
        adjusted = np.minimum(values * n, 1.0)
    elif method == "fdr_bh":  # Benjamini-Hochberg
        ranked = values[order]
        adjusted_sorted = ranked * n / np.arange(1, n + 1)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    else:  # Holm-Bonferroni
        ranked = values[order]
        adjusted_sorted = np.maximum.accumulate(ranked * (n - np.arange(n)))
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)

    return {
        "method": method,
        "n_tests": n,
        "adjusted": [float(v) for v in adjusted],
        "significant": [bool(v < alpha) for v in adjusted],
        "note": (
            f"{n} tests were run. Without correction the chance of at least one false positive is "
            f"about {1 - (1 - alpha) ** n:.0%}; these p-values are adjusted for that."
        ),
    }
