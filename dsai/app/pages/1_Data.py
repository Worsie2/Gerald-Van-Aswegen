"""Data page: load a dataset, see what the platform detected, correct it."""

from __future__ import annotations

import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, column_card, dataframe, hero, metric_row,
    override_notice, page_header, quality_bars, quality_issues, show_notices, sidebar_chrome,
    workflow_nav,
)
from dsai.app.quality import quality_breakdown
from dsai.app.state import scientist, workspace
from dsai.viz import plots


def column_notes(column, profile) -> list[str]:
    """What the platform would do about this variable, and why.

    Drawn from the same measurements the preprocessing recommender reads, so the
    reading here and the pipeline it later builds cannot disagree.
    """
    notes: list[str] = []
    if column.missing_pct > 40:
        notes.append(
            f"**{column.missing_pct:.0f}% of this column is missing.** Imputing that much invents "
            "most of the variable; dropping the column is usually the more honest choice."
        )
    elif column.missing_pct > 0:
        notes.append(
            f"{column.missing_pct:.1f}% missing — enough to need imputation, little enough that "
            "imputing it is defensible."
        )
    if column.skewness is not None and abs(column.skewness) > 1.5:
        direction = "right" if column.skewness > 0 else "left"
        notes.append(
            f"Strongly {direction}-skewed (skew {column.skewness:.2f}). A log or Yeo-Johnson "
            "transform helps linear models here; a tree does not care either way."
        )
    if column.outlier_pct and column.outlier_pct > 5:
        notes.append(
            f"{column.outlier_pct:.1f}% of values sit outside 1.5×IQR. Whether those are errors or "
            "the interesting part of the data is a question about the business, not the numbers."
        )
    if column.n_unique == 1:
        notes.append("Every row holds the same value, so this column carries no information at all.")
    elif column.n_unique > 0 and profile.n_rows and column.n_unique / profile.n_rows > 0.9:
        notes.append(
            "Nearly every row has a distinct value — this looks like an identifier rather than a "
            "measurement, and one-hot encoding it would create a column per row."
        )
    for suspect in getattr(profile, "leakage_suspects", []) or []:
        if suspect.get("column") == column.name:
            notes.append(f"**Possible leakage.** {suspect.get('reason', '')}")
    return notes
from dsai.core.profiler import override_semantic_type
from dsai.core.schema import SemanticType
from dsai.dataio.loaders import LoadError, excel_sheet_names, list_sql_tables, load_file, load_sql

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header("Data", "Load a dataset. The platform inspects it before asking you anything.", "Step 1 of 7")
workflow_nav("Data", state)

file_tab, sql_tab, sample_tab = st.tabs(["Upload a file", "Database query", "Sample datasets"])

with file_tab:
    uploaded = st.file_uploader(
        "CSV, Excel, JSON, JSONL, Parquet or Feather",
        type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "jsonl", "parquet", "feather"],
    )
    sheet = None
    if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        try:
            sheets = excel_sheet_names(uploaded)
            sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
        except Exception:
            sheet = None
    if uploaded is not None and st.button("Load this file", type="primary"):
        try:
            frame, source = load_file(uploaded, name=uploaded.name, sheet=sheet)
            state.frame, state.source = frame, source
            state.dataset_name = uploaded.name
            state.reset_analysis()
            state.notify("success", f"Loaded {len(frame):,} rows × {frame.shape[1]} columns.")
            st.rerun()
        except LoadError as exc:
            st.error(str(exc))

with sql_tab:
    st.caption("Read-only. Only SELECT and WITH statements are accepted.")
    connection = st.text_input("SQLAlchemy connection string",
                               placeholder="postgresql://user:password@host:5432/database")
    if connection and st.button("List tables"):
        try:
            st.write(list_sql_tables(connection))
        except Exception as exc:
            st.error(f"Could not connect: {exc}")
    query = st.text_area("Query", placeholder="SELECT * FROM customers LIMIT 10000", height=110)
    if connection and query and st.button("Run query", type="primary"):
        try:
            frame, source = load_sql(query, connection)
            state.frame, state.source = frame, source
            state.dataset_name = "sql_query"
            state.reset_analysis()
            state.notify("success", f"Loaded {len(frame):,} rows from the database.")
            st.rerun()
        except LoadError as exc:
            st.error(str(exc))

with sample_tab:
    st.caption("Synthetic datasets for trying the platform out. Each has a known structure.")
    from dsai.app.samples import SAMPLES, build_sample

    choice = st.selectbox("Dataset", list(SAMPLES), format_func=lambda k: SAMPLES[k]["label"])
    st.info(SAMPLES[choice]["description"])
    if st.button("Load sample", type="primary"):
        frame, source = build_sample(choice)
        state.frame, state.source = frame, source
        state.dataset_name = choice
        state.reset_analysis()
        state.notify("success", f"Loaded the {SAMPLES[choice]['label']} sample.")
        st.rerun()

if not state.has_data:
    st.stop()

st.divider()

# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------
if state.profile is None:
    with st.spinner("Profiling the dataset…"):
        run = scientist().understand(
            state.frame, state.dataset_name, state.context, state.source
        )
        state.profile = run.profile
        state.typed_frame = scientist()._typed_frame
        state.objectives = run.objectives

profile = state.profile
st.subheader("Dataset intelligence report")
metric_row([
    ("Rows", f"{profile.n_rows:,}", "Total records"),
    ("Columns", str(profile.n_columns), "Total variables"),
    ("Quality score", f"{profile.quality_score}/100",
     "A blunt heuristic: it flags problems, it does not judge whether they matter for your question"),
    ("Duplicates", f"{profile.n_duplicate_rows:,}", f"{profile.duplicate_pct:.2f}% of rows"),
    ("Missing cells",
     f"{sum(c.n_missing for c in profile.columns.values()):,}",
     "Across all columns"),
])

st.markdown(
    f"The platform detected **{len(profile.numeric_columns)} numeric**, "
    f"**{len(profile.categorical_columns)} categorical**, **{len(profile.datetime_columns)} date/time**, "
    f"**{len(profile.text_columns)} text** and **{len(profile.identifier_columns)} identifier** column(s)."
)

preview_tab, types_tab, inspect_tab, quality_tab, targets_tab, stats_tab = st.tabs(
    ["Preview", "Detected types", "Inspect a variable", "Data quality", "Possible targets",
     "Statistics"]
)

with preview_tab:
    dataframe(state.frame.head(200))
    st.caption(f"First 200 of {len(state.frame):,} rows.")

with types_tab:
    override_notice(
        "The platform inferred these types from the values. Where it is wrong, correct it — "
        "the choice changes how each column is preprocessed and modelled."
    )
    import pandas as pd

    rows = [
        {
            "Column": name,
            "Detected type": column.semantic_type.value,
            "Stored as": column.dtype,
            "Missing %": round(column.missing_pct, 2),
            "Distinct": column.n_unique,
            "Example": str(column.dominant_value)[:40],
            "Corrected": column.semantic_type_overridden,
        }
        for name, column in profile.columns.items()
    ]
    dataframe(pd.DataFrame(rows))

    with st.expander("Correct a detected type"):
        column_name = st.selectbox("Column", list(profile.columns))
        current = profile.columns[column_name].semantic_type
        new_type = st.selectbox(
            "It is actually a…", list(SemanticType),
            index=list(SemanticType).index(current),
            format_func=lambda t: t.value.replace("_", " "),
        )
        if st.button("Apply correction") and new_type is not current:
            override_semantic_type(profile, column_name, new_type)
            state.reset_analysis()
            state.notify("success", f"'{column_name}' is now treated as {new_type.value.replace('_', ' ')}.")
            st.rerun()

with quality_tab:
    score_column, bars_column = st.columns([1, 3], gap="large")
    with score_column:
        hero(f"{profile.quality_score:.0f}", "/100", "Data quality score",
             "A blunt heuristic. It flags problems; whether they matter is your call.")
    with bars_column:
        quality_bars(quality_breakdown(profile))

    critical = profile.issues_by_severity("critical")
    warnings = profile.issues_by_severity("warning")
    if critical or warnings:
        ai_panel(
            f"I found **{len(critical)} critical** and **{len(warnings)} warning-level** problem(s) "
            "in this dataset. Each one below says what it is and what fixing it would take. "
            "The preprocessing recommender already accounts for these — you do not have to fix "
            "them by hand first.",
            heading="AI Analyst · data quality",
            why="Nothing here is fixed automatically without you seeing it. A quality problem the "
                "platform silently repairs is a problem you never learn your data has.",
        )
    else:
        ai_panel(
            "Nothing was flagged in this dataset. That means nothing was **detected** — it is not "
            "a guarantee that the data is correct. Checks cover missingness, duplicates, constant "
            "and near-constant columns, outliers, skew, collinearity and columns that look like "
            "they give away the answer.",
            heading="AI Analyst · data quality",
        )
    quality_issues(profile)

with targets_tab:
    st.caption(
        "Columns that look like they could be the thing you want to predict or explain, "
        "ranked by how strongly they resemble an outcome. You choose — this is a suggestion."
    )
    if profile.target_candidates:
        import pandas as pd

        dataframe(pd.DataFrame([
            {
                "Column": c["column"],
                "Score": round(c["score"], 3),
                "Would be": c["implied_task"].replace("_", " "),
                "Distinct values": c["n_unique"],
                "Why": "; ".join(c["reasons"]),
            }
            for c in profile.target_candidates
        ]))
    else:
        st.info("No column stands out as an outcome variable. This may be a dataset for segmentation or exploration.")

    if profile.leakage_suspects:
        caveat(
            "A model built on these will look excellent in testing and fail in production. "
            "Confirm each is genuinely knowable before the outcome occurs.",
            label="Leakage risk",
        )
        for suspect in profile.leakage_suspects:
            st.markdown(f"- **`{suspect['column']}`** — {suspect['reason']}")

with inspect_tab:
    st.caption(
        "One variable at a time, with its distribution and what the platform makes of it. "
        "The reading below each chart is what preprocessing would do about it, and why."
    )
    frame = state.typed_frame if state.typed_frame is not None else state.frame
    chosen = st.selectbox("Variable", list(profile.columns), key="_inspect_column")
    column = profile.columns[chosen]

    facts_column, chart_column = st.columns([1, 2], gap="large")
    with facts_column:
        column_card(column, state.context.currency)
    with chart_column:
        figure = None
        try:
            if column.semantic_type in (SemanticType.CONTINUOUS, SemanticType.DISCRETE):
                figure = plots.histogram(frame[chosen].dropna(), title=f"{chosen}: distribution",
                                         mode=state.theme)
            elif chosen in frame.columns:
                counts = frame[chosen].astype("string").value_counts().head(12)
                figure = plots.bar(list(counts.index), list(counts.values),
                                   title=f"{chosen}: most common values", mode=state.theme,
                                   orientation="h")
        except Exception:
            figure = None
        if figure is not None:
            chart(figure, key=f"inspect_{chosen}")
        else:
            st.caption("No distribution can be drawn for this kind of column.")

    notes = column_notes(column, profile)
    if notes:
        ai_panel(" ".join(notes), heading=f"AI Analyst · {chosen}")

with stats_tab:
    from dsai.statistics.descriptive import describe_categorical, describe_numeric, variance_inflation_factors

    frame = state.typed_frame if state.typed_frame is not None else state.frame
    numeric = describe_numeric(frame, profile.numeric_columns)
    if not numeric.empty:
        st.markdown("**Numeric variables**")
        dataframe(numeric.round(4))
    categorical = describe_categorical(frame, profile.categorical_columns + profile.text_columns)
    if not categorical.empty:
        st.markdown("**Categorical variables**")
        dataframe(categorical.round(4))
    if len(profile.numeric_columns) >= 3:
        vif = variance_inflation_factors(frame, profile.numeric_columns)
        if not vif.empty:
            st.markdown("**Multicollinearity (VIF)**")
            dataframe(vif)

st.divider()
st.markdown("Next: tell the platform what this data represents on the **Context** page.")
