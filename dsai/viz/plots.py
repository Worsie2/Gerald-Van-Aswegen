"""Chart rendering.

Every function returns a Plotly figure, so the hover layer comes for free.
Charts follow one discipline throughout: thin marks, a recessive grid, text in
ink colours rather than series colours, a legend whenever more than one series
is present, and never two y-axes on one plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.viz.theme import ALL_PAIRS_SERIES_CAP, diverging, ink, layout, palette, sequential

MARKER_SIZE = 8
LINE_WIDTH = 2
BAR_GAP = 0.25


def _go():
    import plotly.graph_objects as go

    return go


def _fig(mode: str, title: str, height: int = 380, **overrides):
    go = _go()
    return go.Figure(layout=layout(title, mode, height, **overrides))


# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------

def histogram(series: pd.Series, title: str = "", mode: str = "light",
              bins: int = 40, show_median: bool = True):
    """Distribution with the median marked — the median is what most people should read."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    colours, text = palette(mode)[0], ink(mode)
    figure = _fig(mode, title or f"Distribution of {series.name}")
    figure.add_trace(
        _go().Histogram(
            x=values, nbinsx=bins, marker={"color": colours, "line": {"width": 0}},
            opacity=0.9, name=str(series.name),
            hovertemplate="%{x}<br>%{y} rows<extra></extra>",
        )
    )
    if show_median and not values.empty:
        median, mean = float(values.median()), float(values.mean())
        figure.add_vline(
            x=median, line={"color": text["text_primary"], "width": LINE_WIDTH, "dash": "solid"},
            annotation_text=f"median {median:,.4g}", annotation_position="top",
            annotation_font={"color": text["text_primary"], "size": 11},
        )
        # Only show the mean when it sits somewhere different — otherwise it is noise.
        if abs(mean - median) > 0.05 * max(abs(median), 1e-9):
            figure.add_vline(
                x=mean, line={"color": text["text_muted"], "width": 1, "dash": "dot"},
                annotation_text=f"mean {mean:,.4g}", annotation_position="bottom",
                annotation_font={"color": text["text_muted"], "size": 11},
            )
    figure.update_layout(xaxis_title=str(series.name), yaxis_title="rows", bargap=0.04)
    return figure


def box_by_group(frame: pd.DataFrame, value_column: str, group_column: str,
                 title: str = "", mode: str = "light", max_groups: int = 10):
    """Distribution per group. Spread differences matter as much as average ones."""
    working = frame[[value_column, group_column]].dropna()
    groups = working[group_column].value_counts().head(max_groups).index.tolist()
    colours = palette(mode, n=len(groups))
    figure = _fig(mode, title or f"{value_column} by {group_column}", height=400)
    for i, group in enumerate(groups):
        subset = working.loc[working[group_column] == group, value_column]
        figure.add_trace(
            _go().Box(
                y=subset, name=str(group), boxmean=True,
                marker={"color": colours[i % len(colours)], "size": 5},
                line={"width": 1.5}, fillcolor=colours[i % len(colours)], opacity=0.55,
                hovertemplate=f"{group}<br>%{{y}}<extra></extra>",
            )
        )
    figure.update_layout(
        xaxis_title=group_column, yaxis_title=value_column,
        showlegend=False,  # the x axis already names each group
    )
    return figure


def box_multi(frame: pd.DataFrame, columns: list[str], title: str = "", mode: str = "light",
              standardise: bool = True):
    """Several variables side by side. Standardised, because raw scales are not comparable."""
    figure = _fig(mode, title or "Distribution and outliers", height=400)
    colour = palette(mode)[0]
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if standardise and values.std(ddof=0) > 0:
            values = (values - values.mean()) / values.std(ddof=0)
        figure.add_trace(
            _go().Box(
                y=values, name=column, marker={"color": colour, "size": 5},
                line={"width": 1.5}, fillcolor=colour, opacity=0.45,
                hovertemplate=f"{column}<br>%{{y:.2f}}<extra></extra>",
            )
        )
    figure.update_layout(
        yaxis_title="standard deviations from the mean" if standardise else "value",
        showlegend=False,
    )
    return figure


def missing_bar(profile: Any, title: str = "Missing values by column", mode: str = "light"):
    """Where the gaps are, coloured by severity rather than by identity."""
    from dsai.viz.theme import STATUS

    rows = [(n, c.missing_pct) for n, c in profile.columns.items() if c.missing_pct > 0]
    rows.sort(key=lambda t: t[1], reverse=True)
    if not rows:
        return None
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [
        STATUS["critical"] if v >= 40 else STATUS["warning"] if v >= 5 else STATUS["good"]
        for v in values
    ]
    figure = _fig(mode, title, height=max(280, 24 * len(rows) + 120))
    figure.add_trace(
        _go().Bar(
            x=values, y=names, orientation="h",
            marker={"color": colours, "line": {"width": 0}},
            text=[f"{v:.1f}%" for v in values], textposition="outside",
            textfont={"color": ink(mode)["text_secondary"], "size": 11},
            hovertemplate="%{y}: %{x:.2f}% missing<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="% of rows missing", yaxis={"autorange": "reversed"},
        margin={"l": 160, "r": 60, "t": 52, "b": 52}, bargap=BAR_GAP,
    )
    return figure


# --------------------------------------------------------------------------
# relationships
# --------------------------------------------------------------------------

def scatter(frame: pd.DataFrame, x: str, y: str, colour_by: str | None = None,
            title: str = "", mode: str = "light", trendline: bool = True, max_points: int = 4000):
    """Two variables against each other, optionally split by a third.

    A colour split is capped at three groups: with every pair on screen at once,
    more than that stops being distinguishable for colour-blind readers.
    """
    working = frame[[c for c in (x, y, colour_by) if c]].dropna()
    if len(working) > max_points:
        working = working.sample(max_points, random_state=42)
    figure = _fig(mode, title or f"{y} against {x}", height=420)
    text = ink(mode)

    if colour_by and working[colour_by].nunique() <= ALL_PAIRS_SERIES_CAP:
        groups = working[colour_by].unique()[:ALL_PAIRS_SERIES_CAP]
        colours = palette(mode, n=len(groups), all_pairs=True)
        for i, group in enumerate(groups):
            subset = working[working[colour_by] == group]
            figure.add_trace(
                _go().Scatter(
                    x=subset[x], y=subset[y], mode="markers", name=str(group),
                    marker={"color": colours[i], "size": MARKER_SIZE, "opacity": 0.6,
                            "line": {"width": 1, "color": text["surface"]}},
                    hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>{colour_by}: {group}<extra></extra>",
                )
            )
        figure.update_layout(showlegend=True, legend_title_text=colour_by)
    else:
        figure.add_trace(
            _go().Scatter(
                x=working[x], y=working[y], mode="markers", name=y,
                marker={"color": palette(mode)[0], "size": MARKER_SIZE, "opacity": 0.55,
                        "line": {"width": 1, "color": text["surface"]}},
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
            )
        )

    if trendline and len(working) > 3:
        x_values = pd.to_numeric(working[x], errors="coerce")
        y_values = pd.to_numeric(working[y], errors="coerce")
        valid = x_values.notna() & y_values.notna()
        if valid.sum() > 3 and x_values[valid].std() > 0:
            slope, intercept = np.polyfit(x_values[valid], y_values[valid], 1)
            grid = np.linspace(x_values[valid].min(), x_values[valid].max(), 50)
            correlation = float(np.corrcoef(x_values[valid], y_values[valid])[0, 1])
            figure.add_trace(
                _go().Scatter(
                    x=grid, y=slope * grid + intercept, mode="lines",
                    name=f"trend (r = {correlation:.2f})",
                    line={"color": text["text_primary"], "width": LINE_WIDTH, "dash": "dash"},
                    hoverinfo="skip",
                )
            )
            figure.update_layout(showlegend=True)
    figure.update_layout(xaxis_title=x, yaxis_title=y)
    return figure


def correlation_heatmap(frame: pd.DataFrame, columns: list[str] | None = None,
                        title: str = "Correlation between numeric variables",
                        mode: str = "light", method: str = "pearson"):
    """Diverging scale with a neutral midpoint: zero correlation must read as nothing."""
    columns = columns or list(frame.select_dtypes(include="number").columns)
    columns = columns[:25]
    matrix = frame[columns].corr(method=method)
    size = max(360, 26 * len(columns) + 140)
    figure = _fig(mode, title, height=size)
    figure.add_trace(
        _go().Heatmap(
            z=matrix.to_numpy(), x=columns, y=columns,
            colorscale=[[i / (len(diverging(mode)) - 1), c] for i, c in enumerate(diverging(mode))],
            zmid=0, zmin=-1, zmax=1,
            colorbar={"title": {"text": "r", "font": {"size": 11}}, "thickness": 12, "len": 0.7},
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    if len(columns) <= 12:
        text_colour = ink(mode)["text_primary"]
        figure.update_traces(
            text=matrix.round(2).to_numpy(), texttemplate="%{text}",
            textfont={"size": 10, "color": text_colour},
        )
    figure.update_layout(
        xaxis={"tickangle": -45, "showgrid": False}, yaxis={"autorange": "reversed", "showgrid": False},
        margin={"l": 140, "r": 40, "t": 52, "b": 140},
    )
    return figure


def bar(labels: list[str], values: list[float], title: str = "", mode: str = "light",
        orientation: str = "v", value_format: str = ",.4g", axis_title: str = ""):
    """Single-series bar. One colour: identity is on the axis, not in the hue."""
    figure = _fig(mode, title, height=max(300, 26 * len(labels) + 130) if orientation == "h" else 380)
    colour = palette(mode)[0]
    text = ink(mode)
    labels_text = [f"{v:{value_format}}" for v in values]
    if orientation == "h":
        figure.add_trace(
            _go().Bar(
                x=values, y=labels, orientation="h", marker={"color": colour, "line": {"width": 0}},
                text=labels_text, textposition="outside",
                textfont={"color": text["text_secondary"], "size": 11},
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )
        figure.update_layout(
            yaxis={"autorange": "reversed"}, xaxis_title=axis_title,
            margin={"l": 180, "r": 70, "t": 52, "b": 52},
        )
    else:
        figure.add_trace(
            _go().Bar(
                x=labels, y=values, marker={"color": colour, "line": {"width": 0}},
                text=labels_text, textposition="outside",
                textfont={"color": text["text_secondary"], "size": 11},
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
        )
        figure.update_layout(yaxis_title=axis_title)
    figure.update_layout(bargap=BAR_GAP)
    return figure


def line(frame: pd.DataFrame, x: str, y: str | list[str], title: str = "", mode: str = "light"):
    """Time series. Multiple series get a legend and direct labels."""
    columns = [y] if isinstance(y, str) else y
    working = frame[[x] + columns].dropna().sort_values(x)
    colours = palette(mode, n=len(columns))
    figure = _fig(mode, title or f"{', '.join(columns)} over time", height=380)
    for i, column in enumerate(columns):
        figure.add_trace(
            _go().Scatter(
                x=working[x], y=working[column], mode="lines", name=column,
                line={"color": colours[i % len(colours)], "width": LINE_WIDTH},
                hovertemplate=f"{column}<br>%{{x}}: %{{y:,.4g}}<extra></extra>",
            )
        )
    figure.update_layout(
        xaxis_title=x, yaxis_title=columns[0] if len(columns) == 1 else "value",
        showlegend=len(columns) > 1, hovermode="x unified",
    )
    return figure


# --------------------------------------------------------------------------
# model charts
# --------------------------------------------------------------------------

def feature_importance(explanation: Any, title: str = "What the model relies on",
                       mode: str = "light", top_n: int = 15):
    """Importance ranking. Direction is shown by colour where it is signed."""
    features = explanation.importances[:top_n]
    if not features:
        return None
    names = [f.feature for f in features][::-1]
    values = [abs(f.importance) for f in features][::-1]
    directions = [f.direction for f in features][::-1]

    signed = any(directions)
    colours = palette(mode)
    bar_colours = (
        [colours[0] if d == "increases" else colours[1] for d in directions] if signed
        else colours[0]
    )
    figure = _fig(mode, title, height=max(320, 26 * len(names) + 130))
    figure.add_trace(
        _go().Bar(
            x=values, y=names, orientation="h",
            marker={"color": bar_colours, "line": {"width": 0}},
            error_x={"array": [f.std for f in features][::-1], "color": ink(mode)["text_muted"],
                     "thickness": 1.2, "width": 3} if any(f.std for f in features) else None,
            text=[f"{v:,.4g}" for v in values], textposition="outside",
            textfont={"color": ink(mode)["text_secondary"], "size": 11},
            hovertemplate="%{y}: %{x:,.4g}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title=f"importance ({explanation.method})",
        margin={"l": 190, "r": 80, "t": 52, "b": 52}, bargap=BAR_GAP,
    )
    if signed:
        # Legend by proxy traces, so direction is never colour-alone in the tooltip either.
        for label, colour in (("raises the prediction", colours[0]), ("lowers the prediction", colours[1])):
            figure.add_trace(
                _go().Bar(x=[None], y=[None], name=label, marker={"color": colour}, showlegend=True)
            )
        figure.update_layout(showlegend=True)
    return figure


def predicted_vs_actual(y_true: Any, y_pred: Any, title: str = "Predicted against actual",
                        mode: str = "light", max_points: int = 4000):
    """The most informative single chart for a regression model."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) > max_points:
        index = np.random.default_rng(42).choice(len(y_true), max_points, replace=False)
        y_true, y_pred = y_true[index], y_pred[index]
    text = ink(mode)
    figure = _fig(mode, title, height=420)
    figure.add_trace(
        _go().Scatter(
            x=y_true, y=y_pred, mode="markers", name="predictions",
            marker={"color": palette(mode)[0], "size": MARKER_SIZE, "opacity": 0.55,
                    "line": {"width": 1, "color": text["surface"]}},
            hovertemplate="actual %{x:,.4g}<br>predicted %{y:,.4g}<extra></extra>",
        )
    )
    low, high = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    figure.add_trace(
        _go().Scatter(
            x=[low, high], y=[low, high], mode="lines", name="perfect prediction",
            line={"color": text["text_primary"], "width": LINE_WIDTH, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(xaxis_title="actual", yaxis_title="predicted", showlegend=True)
    return figure


def residual_plot(y_true: Any, y_pred: Any, title: str = "Residuals against fitted values",
                  mode: str = "light", max_points: int = 4000):
    """A funnel shape here means the model is more reliable at one end of the range."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    if len(y_true) > max_points:
        index = np.random.default_rng(42).choice(len(y_true), max_points, replace=False)
        y_pred, residuals = y_pred[index], residuals[index]
    text = ink(mode)
    figure = _fig(mode, title, height=380)
    figure.add_trace(
        _go().Scatter(
            x=y_pred, y=residuals, mode="markers", name="residual",
            marker={"color": palette(mode)[0], "size": MARKER_SIZE, "opacity": 0.55,
                    "line": {"width": 1, "color": text["surface"]}},
            hovertemplate="fitted %{x:,.4g}<br>residual %{y:,.4g}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line={"color": text["text_primary"], "width": LINE_WIDTH, "dash": "dash"})
    figure.update_layout(xaxis_title="fitted value", yaxis_title="residual (actual − predicted)")
    return figure


def confusion_matrix(details: dict[str, Any], title: str = "Confusion matrix", mode: str = "light"):
    """Counts, not colours, carry the meaning — every cell is labelled."""
    matrix = np.asarray(details["matrix"])
    labels = details["labels"]
    figure = _fig(mode, title, height=max(340, 60 * len(labels) + 180))
    figure.add_trace(
        _go().Heatmap(
            z=matrix, x=[f"predicted {l}" for l in labels], y=[f"actual {l}" for l in labels],
            colorscale=[[i / (len(sequential(mode)) - 1), c] for i, c in enumerate(sequential(mode))],
            showscale=False, text=matrix, texttemplate="%{text}",
            textfont={"size": 15, "color": ink(mode)["text_primary"]},
            hovertemplate="%{y}, %{x}: %{z} cases<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis={"showgrid": False}, yaxis={"autorange": "reversed", "showgrid": False},
        margin={"l": 130, "r": 40, "t": 52, "b": 70},
    )
    return figure


def roc_curve(y_true: Any, y_score: Any, title: str = "ROC curve", mode: str = "light"):
    from sklearn.metrics import auc, roc_curve as _roc

    fpr, tpr, _ = _roc(np.asarray(y_true), np.asarray(y_score))
    area = auc(fpr, tpr)
    text = ink(mode)
    figure = _fig(mode, title, height=400)
    figure.add_trace(
        _go().Scatter(
            x=fpr, y=tpr, mode="lines", name=f"model (AUC = {area:.3f})",
            line={"color": palette(mode)[0], "width": LINE_WIDTH},
            hovertemplate="false positive rate %{x:.3f}<br>true positive rate %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        _go().Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="random guessing",
            line={"color": text["text_muted"], "width": 1.5, "dash": "dash"}, hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis_title="false positive rate", yaxis_title="true positive rate", showlegend=True,
    )
    return figure


def precision_recall_curve(y_true: Any, y_score: Any, title: str = "Precision-recall curve",
                           mode: str = "light"):
    from sklearn.metrics import average_precision_score, precision_recall_curve as _pr

    y_true = np.asarray(y_true)
    precision, recall, _ = _pr(y_true, np.asarray(y_score))
    average = average_precision_score(y_true, np.asarray(y_score))
    base_rate = float(np.mean(y_true))
    text = ink(mode)
    figure = _fig(mode, title, height=400)
    figure.add_trace(
        _go().Scatter(
            x=recall, y=precision, mode="lines", name=f"model (AP = {average:.3f})",
            line={"color": palette(mode)[0], "width": LINE_WIDTH},
            hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=base_rate, line={"color": text["text_muted"], "width": 1.5, "dash": "dash"},
        annotation_text=f"base rate {base_rate:.1%}", annotation_position="right",
        annotation_font={"color": text["text_muted"], "size": 11},
    )
    figure.update_layout(xaxis_title="recall", yaxis_title="precision", showlegend=True)
    return figure


def model_comparison(tournament: Any, title: str = "Model comparison", mode: str = "light"):
    """One axis, one metric. Comparing models on two scales at once misleads."""
    ranked = tournament.ranked
    if not ranked:
        return None
    from dsai.engines import metrics as M

    metric = tournament.primary_metric
    names = [r.model_name for r in ranked][::-1]
    values = [r.primary_score for r in ranked][::-1]
    recommended = tournament.recommended.result_id if tournament.recommended else None
    colours = palette(mode)
    bar_colours = [
        colours[0] if r.result_id == recommended else ink(mode)["axis"] for r in ranked
    ][::-1]
    figure = _fig(mode, title, height=max(320, 28 * len(names) + 130))
    figure.add_trace(
        _go().Bar(
            x=values, y=names, orientation="h",
            marker={"color": bar_colours, "line": {"width": 0}},
            text=[M.format_metric(metric, v) for v in values], textposition="outside",
            textfont={"color": ink(mode)["text_secondary"], "size": 11},
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    finite = [v for v in values if v is not None and np.isfinite(v)]
    figure.update_layout(
        xaxis_title=f"{M.METRIC_LABELS.get(metric, metric)}"
                    + (" (lower is better)" if metric in M.LOWER_IS_BETTER else " (higher is better)"),
        # Headroom on the axis so the outside value labels are not clipped.
        xaxis={"range": [0, max(finite) * 1.16]} if finite and min(finite) >= 0 else {},
        margin={"l": 230, "r": 40, "t": 52, "b": 52}, bargap=BAR_GAP,
    )
    return figure


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def cluster_scatter(matrix: Any, labels: Any, title: str = "Segments (PCA projection)",
                    mode: str = "light", max_points: int = 4000):
    """Two-dimensional projection. The axes are components, not real variables."""
    from sklearn.decomposition import PCA

    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    if matrix.shape[1] > 2:
        reducer = PCA(n_components=2, random_state=42)
        coordinates = reducer.fit_transform(matrix)
        explained = reducer.explained_variance_ratio_
        axis_labels = (
            f"component 1 ({explained[0]:.0%} of variance)",
            f"component 2 ({explained[1]:.0%} of variance)",
        )
    else:
        coordinates = matrix if matrix.shape[1] == 2 else np.column_stack([matrix, np.zeros(len(matrix))])
        axis_labels = ("dimension 1", "dimension 2")

    if len(coordinates) > max_points:
        index = np.random.default_rng(42).choice(len(coordinates), max_points, replace=False)
        coordinates, labels = coordinates[index], labels[index]

    unique = sorted(set(labels.tolist()))
    colours = palette(mode, n=len([u for u in unique if u != -1]))
    text = ink(mode)
    figure = _fig(mode, title, height=460)
    colour_index = 0
    for label in unique:
        mask = labels == label
        if label == -1:
            colour, name = text["text_muted"], "unclustered"
        else:
            colour = colours[colour_index % len(colours)]
            name = f"segment {label}"
            colour_index += 1
        figure.add_trace(
            _go().Scatter(
                x=coordinates[mask, 0], y=coordinates[mask, 1], mode="markers", name=name,
                marker={"color": colour, "size": MARKER_SIZE, "opacity": 0.65,
                        "line": {"width": 1, "color": text["surface"]}},
                hovertemplate=f"{name}<extra></extra>",
            )
        )
    figure.update_layout(
        xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], showlegend=True,
    )
    return figure


def cluster_profile(segmentation: Any, title: str = "What distinguishes each segment",
                    mode: str = "light", top_features: int = 6):
    """Each segment against the overall average, in standard deviations."""
    clusters = [c for c in segmentation.clusters if not c.is_noise and c.defining_features]
    if not clusters:
        return None
    features: list[str] = []
    for cluster in clusters:
        for feature in cluster.defining_features:
            if feature["feature"] not in features:
                features.append(feature["feature"])
    features = features[:top_features]

    colours = palette(mode, n=len(clusters))
    figure = _fig(mode, title, height=max(360, 46 * len(features) + 150))
    for i, cluster in enumerate(clusters):
        lookup = {f["feature"]: f["std_deviations"] for f in cluster.defining_features}
        figure.add_trace(
            _go().Bar(
                x=[lookup.get(f, 0.0) for f in features], y=features, orientation="h",
                name=cluster.name, marker={"color": colours[i % len(colours)], "line": {"width": 0}},
                hovertemplate=f"{cluster.name}<br>%{{y}}: %{{x:+.2f}} SD from overall mean<extra></extra>",
            )
        )
    figure.add_vline(x=0, line={"color": ink(mode)["text_primary"], "width": 1.5})
    figure.update_layout(
        barmode="group", bargap=0.22, bargroupgap=0.06,
        xaxis_title="standard deviations from the overall average",
        yaxis={"autorange": "reversed"}, showlegend=True,
        margin={"l": 180, "r": 40, "t": 70, "b": 52},
    )
    return figure


def cluster_sizes(segmentation: Any, title: str = "Segment sizes", mode: str = "light"):
    clusters = segmentation.clusters
    names = [c.name for c in clusters]
    sizes = [c.size for c in clusters]
    figure = bar(names, sizes, title, mode, orientation="h", value_format=",.0f", axis_title="rows")
    return figure


# --------------------------------------------------------------------------
# forecasting
# --------------------------------------------------------------------------

def forecast_plot(result: Any, title: str = "History and forecast", mode: str = "light",
                  target_name: str = "value"):
    """The forecast next to the history it was fitted on, with its interval."""
    history_index = result.extras.get("history_index") or []
    history_values = result.extras.get("history_values") or []
    forecast = result.extras.get("forecast") or []
    if not forecast:
        return None

    colours, text = palette(mode), ink(mode)
    figure = _fig(mode, title, height=400)

    x_history = list(range(len(history_values)))
    x_forecast = list(range(len(history_values) - 1, len(history_values) + len(forecast)))
    labels_history = history_index or [str(i) for i in x_history]

    figure.add_trace(
        _go().Scatter(
            x=x_history, y=history_values, mode="lines", name="history",
            line={"color": text["text_secondary"], "width": LINE_WIDTH},
            customdata=labels_history,
            hovertemplate="%{customdata}<br>%{y:,.4g}<extra></extra>",
        )
    )
    joined = ([history_values[-1]] + list(forecast)) if history_values else list(forecast)
    figure.add_trace(
        _go().Scatter(
            x=x_forecast, y=joined, mode="lines+markers", name="forecast",
            line={"color": colours[0], "width": LINE_WIDTH},
            marker={"size": MARKER_SIZE - 2},
            hovertemplate="period +%{x}<br>%{y:,.4g}<extra></extra>",
        )
    )

    lower, upper = result.extras.get("forecast_lower"), result.extras.get("forecast_upper")
    if lower and upper:
        band_x = x_forecast[1:] + x_forecast[1:][::-1]
        band_y = list(upper) + list(lower)[::-1]
        rgb = tuple(int(colours[0].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        figure.add_trace(
            _go().Scatter(
                x=band_x, y=band_y, fill="toself",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.14)",
                line={"width": 0}, name="95% interval", hoverinfo="skip",
            )
        )
    figure.add_vline(
        x=len(history_values) - 1, line={"color": text["axis"], "width": 1, "dash": "dot"},
        annotation_text="forecast starts", annotation_position="top",
        annotation_font={"color": text["text_muted"], "size": 10},
    )
    figure.update_layout(xaxis_title="period", yaxis_title=target_name, showlegend=True,
                         hovermode="x unified")
    return figure


def decomposition_plot(decomposition: dict[str, Any], title: str = "Trend, seasonality and residual",
                       mode: str = "light"):
    """Three stacked panels — one y-scale each, never overlaid."""
    from plotly.subplots import make_subplots

    if not decomposition.get("supported"):
        return None
    colours, text = palette(mode), ink(mode)
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Trend", "Seasonal", "Residual"),
    )
    for row, (key, colour) in enumerate(
        [("trend", colours[0]), ("seasonal", colours[2]), ("residual", text["text_muted"])], start=1
    ):
        values = decomposition.get(key, [])
        figure.add_trace(
            _go().Scatter(
                x=list(range(len(values))), y=values, mode="lines", name=key.title(),
                line={"color": colour, "width": LINE_WIDTH},
                hovertemplate=f"{key}: %{{y:,.4g}}<extra></extra>",
            ),
            row=row, col=1,
        )
    figure.update_layout(**layout(title, mode, height=620, showlegend=False))
    for axis in figure.layout:
        if axis.startswith(("xaxis", "yaxis")):
            figure.layout[axis].update(gridcolor=text["grid"], linecolor=text["axis"],
                                       tickfont={"color": text["text_muted"], "size": 10})
    for annotation in figure.layout.annotations:
        annotation.font.update(size=12, color=text["text_secondary"])
    return figure


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def render(spec: Any, frame: pd.DataFrame, run: Any = None, mode: str = "light"):
    """Build the figure a :class:`ChartSpec` describes, or None if it cannot be built."""
    kind = spec.kind
    try:
        if kind == "histogram":
            return histogram(frame[spec.columns[0]], spec.title, mode, **spec.options)
        if kind == "box_by_group":
            return box_by_group(frame, spec.columns[0], spec.group_by, spec.title, mode)
        if kind == "bar":
            counts = frame[spec.columns[0]].value_counts()
            return bar([str(i) for i in counts.index], [int(v) for v in counts.to_numpy()],
                       spec.title, mode, value_format=",.0f", axis_title="rows")
        if kind == "box_multi":
            return box_multi(frame, spec.columns, spec.title, mode)
        if kind == "missing_bar":
            return missing_bar(run.profile, spec.title, mode) if run else None
        if kind == "scatter":
            return scatter(frame, spec.columns[0], spec.columns[1], None, spec.title, mode,
                           spec.options.get("trendline", True))
        if kind == "correlation_heatmap":
            return correlation_heatmap(frame, spec.columns, spec.title, mode)
        if kind == "line":
            return line(frame, spec.columns[0], spec.columns[1], spec.title, mode)
        if kind == "predicted_vs_actual" and run and run.best:
            actual = run.best.extras.get("holdout_actual")
            predicted = run.best.extras.get("holdout_predicted")
            return predicted_vs_actual(actual, predicted, spec.title, mode) if predicted else None
        if kind == "residual_plot" and run and run.best:
            actual = run.best.extras.get("holdout_actual")
            predicted = run.best.extras.get("holdout_predicted")
            return residual_plot(actual, predicted, spec.title, mode) if predicted else None
        if kind == "confusion_matrix" and run and run.diagnostics.get("confusion"):
            return confusion_matrix(run.diagnostics["confusion"], spec.title, mode)
        if kind in {"roc_curve", "precision_recall_curve"} and run and run.best:
            scores = run.best.extras.get("holdout_score")
            actual = run.best.extras.get("holdout_actual")
            positive = run.best.extras.get("holdout_positive_label")
            if not scores or positive is None:
                return None
            binary = [1 if str(a) == positive else 0 for a in actual]
            drawer = roc_curve if kind == "roc_curve" else precision_recall_curve
            return drawer(binary, scores, spec.title, mode)
        if kind == "cluster_scatter" and run and run.best:
            matrix = run.best.extras.get("cluster_matrix")
            labels = run.best.extras.get("labels")
            return cluster_scatter(matrix, labels, spec.title, mode) if matrix and labels else None
        if kind == "feature_importance" and run and run.explanation:
            return feature_importance(run.explanation, spec.title, mode)
        if kind == "model_comparison" and run and run.tournament:
            return model_comparison(run.tournament, spec.title, mode)
        if kind == "cluster_profile" and run and run.segmentation:
            return cluster_profile(run.segmentation, spec.title, mode)
        if kind == "cluster_sizes" and run and run.segmentation:
            return cluster_sizes(run.segmentation, spec.title, mode)
        if kind == "forecast" and run and run.best:
            return forecast_plot(run.best, spec.title, mode,
                                 run.objective.target if run.objective else "value")
        if kind == "decomposition" and run and run.series_analysis:
            return decomposition_plot(run.series_analysis.get("decomposition", {}), spec.title, mode)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# before / after — what a preprocessing pipeline actually did
# --------------------------------------------------------------------------

def correlation_comparison(
    before: pd.DataFrame,
    after: pd.DataFrame,
    title: str = "Correlation before and after preprocessing",
    mode: str = "light",
    max_columns: int = 18,
):
    """Two heatmaps side by side, on one shared colour scale.

    A shared scale is the whole point: separate scales would make an unchanged
    matrix look transformed. Reducing collinearity is usually why a pipeline
    exists, and this is the only honest way to see whether it worked.
    """
    from plotly.subplots import make_subplots

    left = before.select_dtypes(include="number")
    right = after.select_dtypes(include="number")
    if left.shape[1] < 2 or right.shape[1] < 2:
        return None
    left = left.iloc[:, :max_columns]
    right = right.iloc[:, :max_columns]

    text = ink(mode)
    scale = [[i / (len(diverging(mode)) - 1), c] for i, c in enumerate(diverging(mode))]
    figure = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.13,
        subplot_titles=(f"Before — {left.shape[1]} variables", f"After — {right.shape[1]} variables"),
    )
    for column, frame in ((1, left), (2, right)):
        matrix = frame.corr()
        figure.add_trace(
            _go().Heatmap(
                z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.columns),
                colorscale=scale, zmid=0, zmin=-1, zmax=1,
                showscale=column == 2,
                colorbar={"title": {"text": "r", "font": {"size": 11}}, "thickness": 11, "len": 0.75},
                hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
            ),
            row=1, col=column,
        )
    height = max(380, 22 * max(left.shape[1], right.shape[1]) + 190)
    figure.update_layout(**layout(title, mode, height=height, showlegend=False))
    figure.update_xaxes(tickangle=-45, showgrid=False, tickfont={"size": 9, "color": text["text_muted"]})
    figure.update_yaxes(autorange="reversed", showgrid=False,
                        tickfont={"size": 9, "color": text["text_muted"]})
    for annotation in figure.layout.annotations:
        annotation.font.update(size=11, color=text["text_secondary"])
    return figure


def distribution_comparison(
    before: pd.Series,
    after: pd.Series,
    title: str = "",
    mode: str = "light",
    bins: int = 40,
):
    """One column before and after a transform, on separate panels.

    Deliberately not overlaid on one axis: a log or scaling step changes the
    units, and drawing two different scales on one axis would misrepresent both.
    """
    from plotly.subplots import make_subplots

    left = pd.to_numeric(before, errors="coerce").dropna()
    right = pd.to_numeric(after, errors="coerce").dropna()
    if left.empty or right.empty:
        return None

    colours, text = palette(mode), ink(mode)
    figure = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=(
            f"Before — skew {left.skew():.2f}",
            f"After — skew {right.skew():.2f}",
        ),
    )
    for column, (values, colour) in enumerate(((left, text["text_secondary"]), (right, colours[0])), start=1):
        figure.add_trace(
            _go().Histogram(
                x=values, nbinsx=bins, marker={"color": colour, "line": {"width": 0}},
                opacity=0.9, hovertemplate="%{x}<br>%{y} rows<extra></extra>",
            ),
            row=1, col=column,
        )
        figure.add_vline(
            x=float(values.median()), row=1, col=column,
            line={"color": text["text_primary"], "width": LINE_WIDTH, "dash": "dot"},
        )
    figure.update_layout(**layout(title or f"{before.name}: before and after", mode,
                                  height=340, showlegend=False), bargap=0.04)
    figure.update_yaxes(title_text="rows", row=1, col=1)
    for annotation in figure.layout.annotations:
        annotation.font.update(size=11, color=text["text_secondary"])
    return figure


def vif_comparison(before: dict[str, float], after: dict[str, float],
                   title: str = "Multicollinearity before and after", mode: str = "light",
                   top_n: int = 12):
    """VIF per variable, before against after, with the danger thresholds marked."""
    if not before:
        return None
    ranked = sorted(before.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [n for n, _ in ranked][::-1]
    before_values = [min(v, 100) for _, v in ranked][::-1]
    after_values = [min(after.get(n, 0.0), 100) for n in names]

    colours, text = palette(mode), ink(mode)
    figure = _fig(mode, title, height=max(320, 30 * len(names) + 140))
    for values, label, colour in (
        (before_values, "before", text["axis"]),
        (after_values, "after", colours[0]),
    ):
        figure.add_trace(
            _go().Bar(
                x=values, y=names, orientation="h", name=label,
                marker={"color": colour, "line": {"width": 0}},
                hovertemplate=f"{label}<br>%{{y}}: VIF %{{x:.1f}}<extra></extra>",
            )
        )
    for threshold, note in ((5, "caution"), (10, "severe")):
        figure.add_vline(
            x=threshold, line={"color": text["text_muted"], "width": 1, "dash": "dot"},
            annotation_text=note, annotation_position="top",
            annotation_font={"color": text["text_muted"], "size": 10},
        )
    figure.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.08, showlegend=True,
        xaxis_title="variance inflation factor (capped at 100)",
        yaxis={"autorange": "reversed"},
        margin={"l": 190, "r": 40, "t": 70, "b": 52},
    )
    return figure


def pipeline_shape(stages: list[dict[str, Any]], title: str = "Columns through the pipeline",
                   mode: str = "light"):
    """How the column count moves at each step — where width is added or removed."""
    labelled = [s for s in stages if s.get("columns_after") is not None]
    if len(labelled) < 2:
        return None
    labels = [s["step"].split("→")[0].strip()[:28] for s in labelled]
    counts = [s["columns_after"] for s in labelled]
    colours = palette(mode)
    figure = _fig(mode, title, height=340)
    figure.add_trace(
        _go().Scatter(
            x=list(range(len(counts))), y=counts, mode="lines+markers",
            line={"color": colours[0], "width": LINE_WIDTH},
            marker={"size": MARKER_SIZE, "line": {"width": 1, "color": ink(mode)["surface"]}},
            customdata=labels, hovertemplate="%{customdata}<br>%{y} columns<extra></extra>",
            name="columns",
        )
    )
    figure.update_layout(
        xaxis={"tickmode": "array", "tickvals": list(range(len(labels))),
               "ticktext": labels, "tickangle": -32},
        yaxis_title="columns", margin={"l": 60, "r": 24, "t": 52, "b": 130},
    )
    return figure
