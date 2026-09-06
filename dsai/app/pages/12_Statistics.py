"""Statistics workbench: run a test directly, with its assumptions checked."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dsai.app.components import (
    apply_theme, caveat, chart, dataframe, empty_state, inference, metric_row, page_header,
    require_data, show_notices, sidebar_chrome,
)
from dsai.app.state import workspace
from dsai.statistics import tests as T
from dsai.statistics.descriptive import (
    correlation_pairs, describe_categorical, describe_numeric, group_summary, outlier_table,
    percentiles, variance_inflation_factors,
)
from dsai.viz import plots

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Statistics",
    "Run a test directly. Every result reports the assumptions it checked, an effect size, and "
    "whether the difference is large enough to matter — separately from whether it is significant.",
)

if not require_data(state):
    st.stop()

profile = state.profile
frame = state.typed_frame if state.typed_frame is not None else state.frame
numeric = [c for c in profile.numeric_columns if c in frame.columns]
categorical = [c for c in profile.categorical_columns if c in frame.columns]

describe_tab, compare_tab, relate_tab, assume_tab = st.tabs(
    ["Describe", "Compare groups", "Relationships", "Assumptions"]
)

# --------------------------------------------------------------------------
with describe_tab:
    if numeric:
        st.markdown("**Numeric variables**")
        dataframe(describe_numeric(frame, numeric).round(4))
        caveat(
            "Where the mean and median differ noticeably the distribution is skewed, and the mean "
            "stops describing a typical case. Report the median for those."
        )
        with st.expander("Percentiles"):
            dataframe(percentiles(frame, numeric).round(4))
        with st.expander("Outliers"):
            dataframe(outlier_table(frame, numeric).round(3))
    if categorical:
        st.markdown("**Categorical variables**")
        dataframe(describe_categorical(frame, categorical))
        caveat(
            "Normalised entropy near 0 means one category dominates and the variable carries little "
            "information; near 1 means the categories are evenly spread."
        )
    if categorical and numeric:
        st.markdown("**By group**")
        columns = st.columns(2)
        group = columns[0].selectbox("Group by", categorical, key="stats_group")
        measures = columns[1].multiselect("Measures", numeric, numeric[:3], key="stats_measures")
        if group and measures:
            dataframe(group_summary(frame, group, measures).round(3))

# --------------------------------------------------------------------------
with compare_tab:
    if not numeric or not categorical:
        empty_state("Needs a numeric measure and a grouping variable",
                    "This dataset does not have both.")
    else:
        columns = st.columns(3)
        value = columns[0].selectbox("Measure", numeric, key="cmp_value")
        group = columns[1].selectbox("Compare across", categorical, key="cmp_group")
        alpha = columns[2].select_slider("Significance level (α)", [0.10, 0.05, 0.01], value=0.05)

        subset = frame[[value, group]].dropna()
        levels = subset[group].value_counts()
        st.caption(f"{len(subset):,} complete rows across {len(levels)} group(s).")

        if len(levels) < 2:
            empty_state("Only one group present", "Nothing to compare against.")
        else:
            if len(levels) == 2:
                names = list(levels.index[:2])
                a = subset.loc[subset[group] == names[0], value]
                b = subset.loc[subset[group] == names[1], value]
                primary = T.t_test(a, b, names=(str(names[0]), str(names[1])), alpha=alpha)
                secondary = T.mann_whitney(a, b, names=(str(names[0]), str(names[1])), alpha=alpha)
            else:
                primary = T.anova(frame, value, group, alpha=alpha)
                secondary = T.kruskal_wallis(frame, value, group, alpha=alpha)

            metric_row([
                ("Test", primary.test.split("(")[0].strip(), ""),
                ("p-value", f"{primary.p_value:.4g}",
                 "Chance of seeing this data if there were no real difference"),
                ("Effect size", f"{primary.effect_size:.3f}" if primary.effect_size is not None else "—",
                 primary.effect_size_name),
                ("Verdict", "significant" if primary.significant else "not significant",
                 f"at α = {alpha}"),
            ])
            inference(primary.conclusion, label="Conclusion")
            caveat(primary.practical_note, label="Practical significance")
            for warning in primary.assumption_warnings:
                caveat(warning, label="Assumption")

            agrees = primary.significant == secondary.significant
            inference(
                f"{secondary.test} {'agrees' if agrees else 'disagrees'}: {secondary.conclusion}",
                label="Non-parametric cross-check",
            )
            if not agrees:
                caveat(
                    "The two tests disagree, which usually means a parametric assumption is "
                    "violated. Trust the non-parametric result.",
                    label="Disagreement",
                )
            figure = plots.box_by_group(frame, value, group, mode=state.theme)
            if figure is not None:
                chart(figure, key="cmp_box")

# --------------------------------------------------------------------------
with relate_tab:
    if len(numeric) >= 2:
        st.markdown("**Correlation between numeric variables**")
        method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True,
                          captions=["straight-line", "monotonic (rank)", "rank, robust to ties"],
                          key="rel_method")
        pairs = correlation_pairs(frame, method=method, columns=numeric, min_abs=0.0)
        if pairs.empty:
            empty_state("No pairs could be computed")
        else:
            dataframe(pairs.round(4))
            figure = plots.correlation_heatmap(frame, numeric, mode=state.theme, method=method)
            if figure is not None:
                chart(figure, key="rel_heat")
            caveat(
                "Correlation shows what moves together. It cannot show what causes what — a third "
                "variable, reverse causation or coincidence all produce the same number."
            )
            st.markdown("**Test one pair**")
            columns = st.columns(2)
            first = columns[0].selectbox("Variable A", numeric, key="rel_a")
            second = columns[1].selectbox("Variable B", [c for c in numeric if c != first], key="rel_b")
            if first and second:
                result = T.correlation_test(frame[first], frame[second], method=method)
                inference(result.conclusion, label=result.test)
                caveat(result.practical_note, label="Practical significance")
                figure = plots.scatter(frame, first, second, mode=state.theme)
                if figure is not None:
                    chart(figure, key="rel_scatter")

    if len(categorical) >= 2:
        st.markdown("**Association between categorical variables**")
        columns = st.columns(2)
        first = columns[0].selectbox("Variable A", categorical, key="cat_a")
        second = columns[1].selectbox("Variable B", [c for c in categorical if c != first], key="cat_b")
        if first and second:
            result = T.chi_square(frame, first, second)
            metric_row([
                ("Test", result.test.split("(")[0].strip(), ""),
                ("p-value", f"{result.p_value:.4g}", ""),
                ("Cramér's V", f"{result.effect_size:.3f}" if result.effect_size is not None else "—",
                 result.effect_interpretation),
            ])
            inference(result.conclusion, label="Conclusion")
            for warning in result.assumption_warnings:
                caveat(warning, label="Assumption")
            dataframe(pd.crosstab(frame[first], frame[second]).reset_index())

# --------------------------------------------------------------------------
with assume_tab:
    st.caption(
        "The checks that decide whether a parametric test is appropriate. Run them before "
        "trusting a t-test or ANOVA, not after."
    )
    if numeric:
        column = st.selectbox("Variable", numeric, key="assume_column")
        result = T.normality_test(frame[column], column)
        metric_row([
            ("Test", result.test.split("(")[0].strip(), ""),
            ("p-value", f"{result.p_value:.4g}" if result.p_value == result.p_value else "—", ""),
            ("Skewness", f"{result.detail.get('skewness', float('nan')):.3f}", "0 is symmetric"),
            ("Kurtosis", f"{result.detail.get('kurtosis', float('nan')):.3f}", "0 is normal-tailed"),
        ])
        inference(result.conclusion, label="Normality")
        if result.practical_note:
            caveat(result.practical_note)
        figure = plots.histogram(frame[column], f"Distribution of {column}", mode=state.theme)
        if figure is not None:
            chart(figure, key="assume_hist")

    if len(numeric) >= 3:
        st.markdown("**Multicollinearity**")
        vif = variance_inflation_factors(frame, numeric)
        if not vif.empty:
            dataframe(vif)
            caveat(
                "A high VIF does not hurt prediction, but it makes individual coefficients "
                "unreliable — their size and even their sign can flip on a slightly different "
                "sample. It matters when you want to explain, not when you only want to predict."
            )

st.divider()
caveat(
    "Running many tests at once inflates the chance of a false positive: twenty comparisons at "
    "α = 0.05 produce at least one about 64% of the time. If you are exploring rather than testing "
    "a specific prior hypothesis, treat what you find here as a lead to confirm, not a result.",
    label="Multiple comparisons",
)
