"""Score page: use the model on data it has never seen."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from dsai.app.components import (
    ai_panel, apply_theme, caveat, chart, dataframe, empty_state, error_state, metric_row,
    page_header, require_run, show_notices, sidebar_chrome,
)
from dsai.app.state import scientist, workspace
from dsai.dataio.loaders import load_dataset
from dsai.engines.scoring import score_new_data
from dsai.viz import plots

state = workspace()
apply_theme(state.theme)
sidebar_chrome(state)
show_notices(state)

page_header(
    "Score new data",
    "Apply the model you built to rows it has never seen. Nothing is refitted — the "
    "transformations applied are the ones learned during training, which is the only way these "
    "predictions are comparable to the scores the model was judged on.",
)

if not require_run(state):
    st.stop()

run = state.run
if run.best is None:
    empty_state(
        "The last run produced no usable model",
        "Scoring needs a model that trained successfully. The **Models** page shows what happened "
        "to each candidate.",
        actions=[("Back to the analysis", "pages/4_Analysis.py")],
    )
    st.stop()

engine = scientist()
model = engine.fitted_model(run)
if model is None:
    error_state(
        "The fitted model is no longer in memory",
        "Models are held for the session that trained them. Restarting the app, or loading a "
        "project saved earlier, leaves the results but not the fitted object.",
        "Re-run the analysis on the **Analysis** page. It will take about as long as it did the "
        "first time, and the seed is recorded so you will get the same model back.",
    )
    st.stop()

training_frame = state.typed_frame if state.typed_frame is not None else state.frame
required = [c for c in (run.best.features or []) if c != run.best.target]

metric_row([
    ("Model", run.best.model_name, "The one selected by the last run"),
    ("Predicts", run.best.target or "segment / anomaly flag", "What it outputs"),
    ("Needs", f"{len(required)} column(s)", "Present under these exact names"),
    ("Trained on", f"{run.best.n_train:,} rows", "Held-out score is on a further "
                                                 f"{run.best.n_test:,}"),
])

with st.expander("The exact columns this model expects"):
    st.markdown(", ".join(f"`{c}`" for c in required))
    st.caption(
        "Names must match. Extra columns are ignored; a column the model needs and cannot find "
        "stops the scoring rather than being quietly filled in."
    )

st.divider()

# --------------------------------------------------------------------------
# where the new rows come from
# --------------------------------------------------------------------------
source_tab, holdout_tab = st.tabs(["Upload a file", "Re-score the loaded dataset"])

new_frame: pd.DataFrame | None = None
label = ""

with source_tab:
    uploaded = st.file_uploader(
        "CSV, Excel, JSON, JSONL, Parquet or Feather",
        type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "jsonl", "parquet", "feather"],
        key="_score_upload",
    )
    if uploaded is not None:
        try:
            loaded, _ = load_dataset(uploaded, name=uploaded.name)
            new_frame, label = loaded, uploaded.name
            st.success(f"Read {len(loaded):,} rows and {loaded.shape[1]} columns from {uploaded.name}.")
        except Exception as exc:
            error_state(
                "That file could not be read",
                f"{type(exc).__name__}: {exc}",
                "Check the file opens in a spreadsheet, that the first row holds column names, and "
                "that the separator is a comma or a tab.",
            )

with holdout_tab:
    st.caption(
        "Runs the model over the dataset already loaded. Useful for producing a prediction for "
        "every row, and for seeing the model's output beside the actual values it was scored on."
    )
    if st.button("Score the loaded dataset", use_container_width=True):
        st.session_state["_score_self"] = True
    if st.session_state.get("_score_self"):
        new_frame, label = (state.frame, state.dataset_name)

if new_frame is None:
    st.info("Upload a file above, or score the dataset already loaded, to see predictions.")
    st.stop()

# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
keep = st.multiselect(
    "Columns to carry through into the output",
    [c for c in new_frame.columns],
    default=[c for c in new_frame.columns if c in (run.profile.identifier_columns if run.profile else [])][:2],
    help="An identifier here makes the predictions easy to join back onto your own records.",
)

with st.spinner("Scoring…"):
    result = score_new_data(model, new_frame, run.best, training_frame=training_frame,
                            keep_columns=keep)

if not result.schema.ok:
    error_state(
        "This data cannot be scored by this model",
        result.schema.summary() + "  \n\nAbsent: "
        + ", ".join(f"`{c}`" for c in result.schema.missing),
        "Add the missing column(s) under those exact names, or train a model on the columns this "
        "file actually has — the **Analysis** page lets you restrict the predictors.",
    )
    st.stop()

ai_panel(
    f"Scored **{result.n_scored:,} row(s)** from *{label}* with **{result.model_name}**. "
    + (result.schema.summary() if result.schema.notes
       else "The new data matches what the model was trained on."),
    heading="AI Analyst · scoring",
    evidence=result.schema.notes[:3] or None,
    why="Nothing was refitted. Every transformation applied here was learned during training, so "
        "these predictions carry the same meaning as the held-out scores the model was judged on.",
)
for note in result.caveats:
    caveat(note)

predictions_tab, drift_tab, distribution_tab = st.tabs(
    ["Predictions", "Has the data changed?", "What the predictions look like"]
)

with predictions_tab:
    dataframe(result.predictions.head(500))
    st.caption(f"First 500 of {len(result.predictions):,} rows.")
    buffer = io.StringIO()
    result.predictions.to_csv(buffer, index=False)
    st.download_button(
        f"Download all {len(result.predictions):,} predictions (.csv)",
        buffer.getvalue(),
        file_name=f"{state.dataset_name}_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

with drift_tab:
    if result.drift.empty:
        st.info("No comparable columns, so there is nothing to compare against training.")
    else:
        st.caption(
            "How far each column has moved since the model was trained. Identifiers and dates are "
            "left out: every ID in a new file is new and every date is later, and reporting that "
            "as drift would bury the columns where a change actually means something."
        )
        dataframe(result.drift)
        caveat(
            "A model does not know when it is out of its depth. It will predict just as "
            "confidently on a population it has never seen — which is why this table matters more "
            "than the predictions look like they need it to."
        )

with distribution_tab:
    numeric = [c for c in result.predictions.columns
               if pd.api.types.is_numeric_dtype(result.predictions[c])
               and c.startswith(("predicted_", "prediction", "anomaly_score"))]
    if numeric:
        column = numeric[0]
        chart(
            plots.histogram(result.predictions[column].dropna(),
                            title=f"Distribution of {column}", mode=state.theme),
            caption="If this looks nothing like the target did during training, the model is being "
                    "asked about a different population from the one it learned on.",
            key="score_hist",
        )
        if run.best.target and run.best.target in training_frame.columns:
            comparison = pd.DataFrame({
                "Statistic": ["Rows", "Mean", "Median", "Min", "Max"],
                "Training target": [
                    f"{len(training_frame):,}",
                    f"{training_frame[run.best.target].mean():,.2f}",
                    f"{training_frame[run.best.target].median():,.2f}",
                    f"{training_frame[run.best.target].min():,.2f}",
                    f"{training_frame[run.best.target].max():,.2f}",
                ],
                "These predictions": [
                    f"{len(result.predictions):,}",
                    f"{result.predictions[column].mean():,.2f}",
                    f"{result.predictions[column].median():,.2f}",
                    f"{result.predictions[column].min():,.2f}",
                    f"{result.predictions[column].max():,.2f}",
                ],
            })
            st.markdown("**Predictions against the target the model learned from**")
            dataframe(comparison)
    else:
        counts = result.predictions.iloc[:, -1].astype("string").value_counts()
        chart(
            plots.bar(list(counts.index), list(counts.values),
                      title="How the predictions are distributed", mode=state.theme,
                      orientation="h"),
            key="score_bar",
        )
