"""Turn cluster labels into segments a business can actually act on.

A silhouette score tells you the clusters are separable. It does not tell you
who is in them or what to do about them. This module produces the profile that
does: size, defining characteristics, how each segment differs from the average,
and a plain-English description with the evidence attached.
"""

from __future__ import annotations

from dsai.engines.metrics import human_number

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import Confidence, DatasetProfile, Finding, EvidenceKind, JsonMixin


@dataclass
class ClusterProfile(JsonMixin):
    label: int
    size: int
    share: float
    name: str = ""
    description: str = ""
    defining_features: list[dict[str, Any]] = field(default_factory=list)
    numeric_means: dict[str, float] = field(default_factory=dict)
    categorical_modes: dict[str, str] = field(default_factory=dict)
    representative_rows: list[int] = field(default_factory=list)
    is_noise: bool = False


@dataclass
class SegmentationProfile(JsonMixin):
    n_clusters: int = 0
    n_noise: int = 0
    clusters: list[ClusterProfile] = field(default_factory=list)
    separating_features: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    quality: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def narrative(self) -> str:
        lines = [f"{self.n_clusters} segment(s) were found."]
        for cluster in self.clusters:
            lines.append(f"\n{cluster.name} — {cluster.size:,} rows ({cluster.share:.1%})\n  {cluster.description}")
        return "\n".join(lines)


def profile_clusters(
    frame: pd.DataFrame,
    labels: Any,
    profile: DatasetProfile | None = None,
    feature_columns: list[str] | None = None,
    currency: str = "ZAR",
    max_defining: int = 5,
) -> SegmentationProfile:
    """Describe each cluster in terms of the original, un-transformed variables.

    Profiling on the raw columns rather than the scaled ones is deliberate: a
    segment described as "0.8 standard deviations above the mean" is useless to
    anyone outside the analysis.
    """
    labels = np.asarray(labels)
    working = frame.copy()
    if len(labels) != len(working):
        raise ValueError(f"Got {len(labels)} labels for {len(working)} rows.")
    working["__cluster__"] = labels

    columns = feature_columns or [c for c in frame.columns if c != "__cluster__"]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in columns if c not in numeric and frame[c].nunique() <= 50]

    out = SegmentationProfile()
    unique = sorted(set(labels.tolist()))
    out.n_noise = int((labels == -1).sum())
    out.n_clusters = len([u for u in unique if u != -1])

    overall_numeric = working[numeric].mean() if numeric else pd.Series(dtype=float)
    overall_std = working[numeric].std(ddof=0).replace(0, np.nan) if numeric else pd.Series(dtype=float)

    for label in unique:
        subset = working[working["__cluster__"] == label]
        cluster = ClusterProfile(
            label=int(label),
            size=len(subset),
            share=len(subset) / max(len(working), 1),
            is_noise=(label == -1),
        )
        if numeric:
            means = subset[numeric].mean()
            cluster.numeric_means = {c: float(means[c]) for c in numeric}
            # Standardised deviation from the overall mean is how a segment is
            # actually distinctive, rather than merely large or small.
            deviations = ((means - overall_numeric) / overall_std).dropna()
            for column in deviations.abs().sort_values(ascending=False).head(max_defining).index:
                deviation = float(deviations[column])
                if abs(deviation) < 0.25:
                    continue
                cluster.defining_features.append(
                    {
                        "feature": column,
                        "cluster_mean": float(means[column]),
                        "overall_mean": float(overall_numeric[column]),
                        "std_deviations": round(deviation, 2),
                        "direction": "higher" if deviation > 0 else "lower",
                        "pct_difference": round(
                            100 * (means[column] - overall_numeric[column]) / abs(overall_numeric[column]), 1
                        ) if abs(overall_numeric[column]) > 1e-9 else None,
                    }
                )
        for column in categorical:
            values = subset[column].dropna()
            if not values.empty:
                cluster.categorical_modes[column] = str(values.mode().iloc[0])

        cluster.representative_rows = _representative_rows(subset, numeric, means if numeric else None)
        cluster.name = _name_cluster(cluster, out.n_clusters)
        cluster.description = _describe_cluster(cluster, currency)
        out.clusters.append(cluster)

    out.separating_features = _separating_features(working, numeric, categorical)
    out.findings = _cluster_findings(out, len(working))
    out.warnings = _cluster_warnings(out, len(working))
    return out


def _representative_rows(subset: pd.DataFrame, numeric: list[str], means, top_n: int = 3) -> list[int]:
    """Rows closest to the cluster centre — real examples beat abstractions."""
    if not numeric or means is None or subset.empty:
        return [int(i) for i in subset.index[:top_n]]
    values = subset[numeric].fillna(means)
    spread = values.std(ddof=0).replace(0, 1.0)
    distances = (((values - means) / spread) ** 2).sum(axis=1)
    return [int(i) for i in distances.nsmallest(min(top_n, len(distances))).index]


def _name_cluster(cluster: ClusterProfile, total_clusters: int) -> str:
    if cluster.is_noise:
        return "Unclustered / noise"
    if not cluster.defining_features:
        return f"Segment {cluster.label} (average across the board)"
    top = cluster.defining_features[0]
    return f"Segment {cluster.label} ({top['direction']} {top['feature']})"


def _describe_cluster(cluster: ClusterProfile, currency: str) -> str:
    if cluster.is_noise:
        return (
            f"{cluster.size:,} rows ({cluster.share:.1%}) did not fit any dense region. These are "
            "worth inspecting individually — they are either genuine outliers or a signal that the "
            "clustering settings are too strict."
        )
    if not cluster.defining_features:
        return (
            f"{cluster.size:,} rows ({cluster.share:.1%}) that sit close to the overall average on "
            "every measured variable. This is the 'typical' group, defined by not being distinctive."
        )
    parts = []
    for feature in cluster.defining_features[:3]:
        pct = feature.get("pct_difference")
        magnitude = f"{abs(pct):.0f}% {feature['direction']}" if pct is not None else (
            f"{abs(feature['std_deviations']):.1f} standard deviations {feature['direction']}"
        )
        parts.append(
            f"{feature['feature']} averages {human_number(feature['cluster_mean'])} "
            f"({magnitude} than the overall average of {human_number(feature['overall_mean'])})"
        )
    description = f"{cluster.size:,} rows ({cluster.share:.1%}). " + "; ".join(parts) + "."
    if cluster.categorical_modes:
        modes = ", ".join(f"{k}: {v}" for k, v in list(cluster.categorical_modes.items())[:3])
        description += f" Most common characteristics — {modes}."
    return description


def _separating_features(working: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[dict[str, Any]]:
    """Which variables actually separate the clusters, ranked by test statistic."""
    from scipy import stats

    core = working[working["__cluster__"] != -1]
    if core["__cluster__"].nunique() < 2:
        return []
    rows = []
    for column in numeric:
        groups = [g[column].dropna().to_numpy() for _, g in core.groupby("__cluster__", observed=True)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        try:
            statistic, p_value = stats.f_oneway(*groups)
        except Exception:
            continue
        if not np.isfinite(statistic):
            continue
        grand_mean = float(np.concatenate(groups).mean())
        ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
        ss_total = float(((np.concatenate(groups) - grand_mean) ** 2).sum())
        rows.append(
            {
                "feature": column,
                "type": "numeric",
                "f_statistic": float(statistic),
                "p_value": float(p_value),
                "variance_explained": round(ss_between / ss_total, 4) if ss_total > 0 else 0.0,
                "separates": bool(p_value < 0.05),
            }
        )
    for column in categorical:
        try:
            table = pd.crosstab(core[column], core["__cluster__"])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            statistic, p_value, _, _ = stats.chi2_contingency(table)
            n = int(table.to_numpy().sum())
            cramers_v = float(np.sqrt(statistic / (n * (min(table.shape) - 1)))) if n else 0.0
        except Exception:
            continue
        rows.append(
            {
                "feature": column,
                "type": "categorical",
                "chi_square": float(statistic),
                "p_value": float(p_value),
                "cramers_v": round(cramers_v, 4),
                "separates": bool(p_value < 0.05),
            }
        )
    rows.sort(key=lambda r: r.get("variance_explained", r.get("cramers_v", 0)), reverse=True)
    return rows


def _cluster_findings(out: SegmentationProfile, n_rows: int) -> list[Finding]:
    findings: list[Finding] = []
    real = [c for c in out.clusters if not c.is_noise]
    if not real:
        return findings

    largest = max(real, key=lambda c: c.size)
    smallest = min(real, key=lambda c: c.size)
    findings.append(
        Finding(
            title=f"{len(real)} distinct segments identified",
            detail=(
                f"The largest holds {largest.size:,} rows ({largest.share:.1%}) and the smallest "
                f"{smallest.size:,} ({smallest.share:.1%})."
            ),
            kind=EvidenceKind.MODEL,
            evidence=[f"{c.name}: {c.size:,} rows" for c in real],
            confidence=Confidence.MODERATE,
        )
    )

    strong = [f for f in out.separating_features if f.get("separates")][:3]
    if strong:
        names = ", ".join(f["feature"] for f in strong)
        findings.append(
            Finding(
                title=f"Segments are driven mainly by {names}",
                detail=(
                    "These variables differ across segments by more than chance would produce. "
                    "The rest contribute little to the separation."
                ),
                kind=EvidenceKind.STATISTICAL,
                evidence=[
                    f"{f['feature']}: p = {f['p_value']:.2e}"
                    + (f", explains {f['variance_explained']:.1%} of the variance" if "variance_explained" in f else "")
                    for f in strong
                ],
                columns=[f["feature"] for f in strong],
                confidence=Confidence.HIGH,
            )
        )

    for cluster in real:
        if not cluster.defining_features:
            continue
        top = cluster.defining_features[0]
        if abs(top["std_deviations"]) >= 1.0:
            findings.append(
                Finding(
                    title=f"{cluster.name} is genuinely distinctive",
                    detail=cluster.description,
                    kind=EvidenceKind.MODEL,
                    evidence=[
                        f"{f['feature']}: {human_number(f['cluster_mean'])} versus {human_number(f['overall_mean'])} overall"
                        for f in cluster.defining_features[:3]
                    ],
                    columns=[f["feature"] for f in cluster.defining_features[:3]],
                    confidence=Confidence.MODERATE,
                    caveats=[
                        "Segments are a description of this dataset, not a law about the population. "
                        "Re-running on new data can produce different boundaries."
                    ],
                )
            )
    return findings


def _cluster_warnings(out: SegmentationProfile, n_rows: int) -> list[str]:
    warnings: list[str] = []
    real = [c for c in out.clusters if not c.is_noise]
    if not real:
        warnings.append("No clusters were formed — every row was classified as noise.")
        return warnings
    tiny = [c for c in real if c.size < max(10, 0.02 * n_rows)]
    if tiny:
        warnings.append(
            f"{len(tiny)} segment(s) contain fewer than 2% of rows. Segments that small are usually "
            "outlier pockets rather than actionable groups."
        )
    if out.n_noise > 0.3 * n_rows:
        warnings.append(
            f"{out.n_noise:,} rows ({out.n_noise / n_rows:.0%}) were left unclustered. Either loosen "
            "the density settings or accept that a large part of the data has no group structure."
        )
    undistinctive = [c for c in real if not c.defining_features]
    if len(undistinctive) >= max(1, len(real) // 2):
        warnings.append(
            "Half or more of the segments have no defining characteristic. The clustering may be "
            "splitting a single continuous population rather than finding real groups."
        )
    return warnings


def suggest_cluster_count(
    matrix: Any,
    k_range: range | None = None,
    method: str = "kmeans",
    random_state: int = 42,
) -> dict[str, Any]:
    """Sweep k and report elbow, silhouette, Calinski-Harabasz and Davies-Bouldin.

    The four measures rarely agree exactly; where they do, that is a much
    stronger signal than any one of them alone.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    k_range = k_range or range(2, min(11, max(3, n // 10)))
    rows = []
    for k in k_range:
        if k >= n:
            break
        try:
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(matrix)
            labels = model.labels_
            rows.append(
                {
                    "k": int(k),
                    "inertia": float(model.inertia_),
                    "silhouette": float(silhouette_score(matrix, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                    "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
                }
            )
        except Exception:
            continue
    if not rows:
        return {"supported": False, "reason": "Could not fit any clustering over the requested range."}

    table = pd.DataFrame(rows)
    best_silhouette = int(table.loc[table.silhouette.idxmax(), "k"])
    best_calinski = int(table.loc[table.calinski_harabasz.idxmax(), "k"])
    best_davies = int(table.loc[table.davies_bouldin.idxmin(), "k"])
    elbow = _elbow_point(table.k.to_numpy(), table.inertia.to_numpy())

    votes = pd.Series([best_silhouette, best_calinski, best_davies] + ([elbow] if elbow else []))
    consensus = int(votes.mode().iloc[0])
    agreement = int((votes == consensus).sum())

    return {
        "supported": True,
        "table": rows,
        "elbow_k": elbow,
        "best_silhouette_k": best_silhouette,
        "best_calinski_k": best_calinski,
        "best_davies_bouldin_k": best_davies,
        "recommended_k": consensus,
        "agreement": f"{agreement} of {len(votes)} measures",
        "confidence": "high" if agreement >= 3 else "moderate" if agreement == 2 else "low",
        "interpretation": (
            f"{agreement} of {len(votes)} measures point to k = {consensus}. "
            + (
                "That level of agreement is a genuine signal."
                if agreement >= 3 else
                "The measures disagree, which usually means the data has no sharp cluster structure — "
                "choose k on what is useful to the business rather than on the metrics."
            )
        ),
    }


def _elbow_point(ks: np.ndarray, inertias: np.ndarray) -> int | None:
    """Kneedle-style elbow: the point furthest from the line joining the ends."""
    if len(ks) < 3:
        return None
    x = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
    y = (inertias - inertias.min()) / max(inertias.max() - inertias.min(), 1e-12)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line = line / max(np.linalg.norm(line), 1e-12)
    distances = []
    for i in range(len(x)):
        point = np.array([x[i], y[i]]) - start
        distances.append(float(np.linalg.norm(point - np.dot(point, line) * line)))
    return int(ks[int(np.argmax(distances))])
