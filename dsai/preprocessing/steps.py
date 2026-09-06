"""The preprocessing step catalogue.

Like the model registry, preprocessing is metadata-driven: each operation is a
:class:`StepSpec` describing what it does, what it needs, what it costs you, and
whether it is safe to apply before splitting the data.

The ``leakage_safe`` flag is the important one. Steps that learn anything from
the data (a mean, a scale, a category-to-target map) must be fitted inside the
cross-validation fold, never on the full dataset. Steps marked unsafe are placed
inside the modelling pipeline automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dsai.core.schema import JsonMixin, SemanticType


@dataclass
class StepParam(JsonMixin):
    name: str
    kind: str                       # int | float | categorical | bool | text
    default: Any = None
    low: float | None = None
    high: float | None = None
    choices: list[Any] = field(default_factory=list)
    description: str = ""


@dataclass
class StepSpec(JsonMixin):
    """One preprocessing operation available to the pipeline builder."""

    key: str
    name: str
    category: str                   # imputation | outliers | scaling | transformation |
                                    # encoding | feature_engineering | feature_selection |
                                    # dimensionality | cleaning
    description: str
    builder: Callable[..., Any] | None = None
    scope: str = "column"           # column = fitted transformer | row = changes the row set
    applies_to: list[SemanticType] = field(default_factory=list)
    params: list[StepParam] = field(default_factory=list)
    leakage_safe: bool = True       # True = may be applied before splitting
    changes_row_count: bool = False
    needs_target: bool = False
    output_mode: str = "replace"    # replace | append
    when_to_use: str = ""
    caveats: list[str] = field(default_factory=list)
    module: str = "sklearn"

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def build(self, **params: Any):
        if self.builder is None:
            raise RuntimeError(f"Preprocessing step '{self.key}' has no builder.")
        return self.builder(**{**self.default_params(), **params})


class StepRegistry:
    """Catalogue of preprocessing operations, queryable by column type."""

    def __init__(self) -> None:
        self._steps: dict[str, StepSpec] = {}

    def register(self, spec: StepSpec, replace: bool = True) -> StepSpec:
        if spec.key in self._steps and not replace:
            raise ValueError(f"Preprocessing step '{spec.key}' already registered.")
        self._steps[spec.key] = spec
        return spec

    def get(self, key: str) -> StepSpec:
        if key not in self._steps:
            raise KeyError(f"Unknown preprocessing step '{key}'. Known: {', '.join(sorted(self._steps))}")
        return self._steps[key]

    def all(self) -> list[StepSpec]:
        return list(self._steps.values())

    def keys(self) -> list[str]:
        return list(self._steps)

    def categories(self) -> list[str]:
        return sorted({s.category for s in self._steps.values()})

    def for_semantic(self, semantic: SemanticType) -> list[StepSpec]:
        return [s for s in self._steps.values() if not s.applies_to or semantic in s.applies_to]

    def by_category(self, category: str) -> list[StepSpec]:
        return [s for s in self._steps.values() if s.category == category]

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, key: object) -> bool:
        return key in self._steps


STEPS = StepRegistry()


def _p(name: str, kind: str, default: Any = None, low=None, high=None,
       choices=None, description: str = "") -> StepParam:
    return StepParam(name=name, kind=kind, default=default, low=low, high=high,
                     choices=choices or [], description=description)


NUMERIC = [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]
CATEGORICAL = [
    SemanticType.CATEGORICAL_NOMINAL,
    SemanticType.CATEGORICAL_ORDINAL,
    SemanticType.BINARY,
    SemanticType.HIGH_CARDINALITY_CATEGORICAL,
]


def _register_all() -> None:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import (
        SelectKBest, VarianceThreshold, f_classif, f_regression, mutual_info_regression,
    )
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import (
        Binarizer, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, Normalizer,
        OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer,
        QuantileTransformer, RobustScaler, StandardScaler,
    )

    from dsai.preprocessing import transformers as T

    def _enable_iterative():
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer

    steps = [
        # ---------------- cleaning (row scope) ----------------
        StepSpec(
            key="drop_duplicates",
            name="Remove duplicate rows",
            category="cleaning",
            description="Delete rows that are exact copies of an earlier row.",
            scope="row",
            changes_row_count=True,
            when_to_use="When repeated rows are data-entry artefacts rather than genuine repeat events.",
            caveats=["Legitimate repeat transactions can look identical — check before removing."],
            params=[_p("subset", "categorical", None, description="Limit the comparison to these columns.")],
        ),
        StepSpec(
            key="drop_rows_missing",
            name="Drop rows with missing values",
            category="cleaning",
            description="Delete any row containing a missing value in the chosen columns.",
            scope="row",
            changes_row_count=True,
            when_to_use="Only when missingness is rare and clearly random.",
            caveats=[
                "Throws away real information",
                "Biases the sample if the values are not missing at random",
            ],
            params=[_p("threshold", "float", 1.0, 0.0, 1.0, description="Drop a row when this share of its values is missing.")],
        ),
        StepSpec(
            key="drop_columns",
            name="Drop columns",
            category="cleaning",
            description="Remove named columns from the analysis entirely.",
            scope="column",
            builder=lambda columns=None, **_: T.ColumnSelector(columns=columns or [], mode="drop"),
            when_to_use="Identifiers, constants, leakage suspects and anything the user excludes.",
            params=[_p("columns", "categorical", None, description="Columns to remove.")],
        ),
        StepSpec(
            key="drop_constant_columns",
            name="Drop constant / near-constant columns",
            category="cleaning",
            description="Remove columns with (almost) no variation.",
            scope="column",
            builder=lambda threshold=0.0, **_: T.FrameTransformer(VarianceThreshold(threshold=threshold), None),
            applies_to=NUMERIC,
            when_to_use="Columns that never change cannot explain anything that does.",
            params=[_p("threshold", "float", 0.0, 0.0, 0.1, description="Minimum variance a column must have to survive.")],
        ),

        # ---------------- imputation ----------------
        StepSpec(
            key="impute_mean",
            name="Impute missing with the mean",
            category="imputation",
            description="Fill gaps in numeric columns with the column mean.",
            builder=lambda columns=None, **_: T.FrameTransformer(SimpleImputer(strategy="mean"), columns),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Roughly symmetric numeric columns with a small share of gaps.",
            caveats=["Shrinks variance", "Distorted by outliers — prefer the median when the column is skewed"],
        ),
        StepSpec(
            key="impute_median",
            name="Impute missing with the median",
            category="imputation",
            description="Fill gaps in numeric columns with the column median.",
            builder=lambda columns=None, **_: T.FrameTransformer(SimpleImputer(strategy="median"), columns),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="The safe default for skewed or outlier-prone numeric columns.",
            caveats=["Still shrinks variance", "Ignores relationships with other columns"],
        ),
        StepSpec(
            key="impute_mode",
            name="Impute missing with the most frequent value",
            category="imputation",
            description="Fill gaps in categorical columns with the most common category.",
            builder=lambda columns=None, **_: T.FrameTransformer(SimpleImputer(strategy="most_frequent"), columns),
            applies_to=CATEGORICAL,
            leakage_safe=False,
            when_to_use="Categorical columns with few gaps.",
            caveats=["Over-represents the dominant category"],
        ),
        StepSpec(
            key="impute_constant",
            name="Impute with a fixed value / 'Missing' category",
            category="imputation",
            description="Fill gaps with a constant, treating 'missing' as its own meaningful level.",
            builder=lambda columns=None, fill_value="__missing__", **_: T.FrameTransformer(
                SimpleImputer(strategy="constant", fill_value=fill_value), columns
            ),
            leakage_safe=True,
            when_to_use="When the fact that a value is missing is itself informative.",
            params=[_p("fill_value", "text", "__missing__", description="Value to insert.")],
        ),
        StepSpec(
            key="impute_knn",
            name="KNN imputation",
            category="imputation",
            description="Estimate each missing value from the most similar complete rows.",
            builder=lambda columns=None, n_neighbors=5, **_: T.FrameTransformer(
                KNNImputer(n_neighbors=n_neighbors), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Numeric data where columns are correlated, so neighbours are informative.",
            caveats=["Slow on large datasets", "Requires the numeric columns to be on similar scales"],
            params=[_p("n_neighbors", "int", 5, 1, 25, description="Neighbours to average over.")],
        ),
        StepSpec(
            key="impute_iterative",
            name="Iterative (MICE-style) imputation",
            category="imputation",
            description="Model each column with gaps from the others, repeatedly, until estimates settle.",
            builder=lambda columns=None, max_iter=10, **_: T.FrameTransformer(
                _enable_iterative()(max_iter=max_iter, random_state=42), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Substantial numeric missingness with strong inter-column relationships.",
            caveats=["Slowest option", "Can invent structure that is not really there", "Still experimental in scikit-learn"],
            params=[_p("max_iter", "int", 10, 3, 50)],
        ),
        StepSpec(
            key="impute_forward_fill",
            name="Forward / backward fill",
            category="imputation",
            description="Carry the previous observation forward (then backwards for any leading gaps).",
            scope="row",
            leakage_safe=True,
            when_to_use="Time-ordered data where the last known value is the best guess.",
            caveats=["Only valid when rows are genuinely in time order"],
            params=[_p("direction", "categorical", "forward", choices=["forward", "backward", "both"])],
        ),

        # ---------------- outliers ----------------
        StepSpec(
            key="winsorize",
            name="Winsorise (clip to percentiles)",
            category="outliers",
            description="Pull extreme values in to the 1st and 99th percentile instead of deleting them.",
            builder=lambda columns=None, lower_quantile=0.01, upper_quantile=0.99, **_: T.FrameTransformer(
                T.WinsorizeTransformer(lower_quantile, upper_quantile), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Extreme values are real but you do not want them dominating the fit.",
            caveats=["Changes the distribution's tails — say so when reporting"],
            params=[
                _p("lower_quantile", "float", 0.01, 0.0, 0.2),
                _p("upper_quantile", "float", 0.99, 0.8, 1.0),
            ],
        ),
        StepSpec(
            key="clip_iqr",
            name="Clip to Tukey (IQR) fences",
            category="outliers",
            description="Clip values outside Q1 − 1.5·IQR and Q3 + 1.5·IQR.",
            builder=lambda columns=None, factor=1.5, **_: T.FrameTransformer(
                T.OutlierClipTransformer(factor), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Standard, defensible outlier treatment that keeps every row.",
            params=[_p("factor", "float", 1.5, 1.0, 3.0)],
        ),
        StepSpec(
            key="remove_outlier_rows",
            name="Remove outlier rows",
            category="outliers",
            description="Delete rows lying beyond the IQR fences on any selected column.",
            scope="row",
            changes_row_count=True,
            leakage_safe=True,
            when_to_use="Only when you have established the rows are errors, not genuine extremes.",
            caveats=["Outliers are often the most interesting observations — investigate before deleting"],
            params=[_p("factor", "float", 3.0, 1.5, 5.0)],
        ),

        # ---------------- scaling ----------------
        StepSpec(
            key="standard_scale",
            name="Standardise (z-score)",
            category="scaling",
            description="Centre each column on 0 with a standard deviation of 1.",
            builder=lambda columns=None, **_: T.FrameTransformer(StandardScaler(), columns),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Required by distance and gradient-based models: SVM, KNN, neural networks, PCA, regularised linear models.",
            caveats=["Assumes roughly symmetric data", "Outliers stretch the scale"],
        ),
        StepSpec(
            key="minmax_scale",
            name="Min-max scale to [0, 1]",
            category="scaling",
            description="Rescale each column into a fixed range.",
            builder=lambda columns=None, feature_range=(0, 1), **_: T.FrameTransformer(
                MinMaxScaler(feature_range=tuple(feature_range)), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Neural networks, or anywhere a bounded input is needed.",
            caveats=["Very sensitive to outliers", "New extreme values fall outside the range"],
            params=[_p("feature_range", "categorical", (0, 1), choices=[(0, 1), (-1, 1)])],
        ),
        StepSpec(
            key="robust_scale",
            name="Robust scale (median / IQR)",
            category="scaling",
            description="Centre on the median and scale by the interquartile range.",
            builder=lambda columns=None, **_: T.FrameTransformer(RobustScaler(), columns),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Scaling that outliers cannot distort.",
        ),
        StepSpec(
            key="maxabs_scale",
            name="Max-abs scale",
            category="scaling",
            description="Divide each column by its largest absolute value, preserving zeros and sparsity.",
            builder=lambda columns=None, **_: T.FrameTransformer(MaxAbsScaler(), columns),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Sparse data where centring would destroy the sparsity.",
        ),
        StepSpec(
            key="normalize_rows",
            name="Normalise rows to unit length",
            category="scaling",
            description="Scale each row (not column) to unit norm.",
            builder=lambda columns=None, norm="l2", **_: T.FrameTransformer(Normalizer(norm=norm), columns),
            applies_to=NUMERIC,
            when_to_use="When the profile across a row matters more than absolute magnitudes.",
            params=[_p("norm", "categorical", "l2", choices=["l1", "l2", "max"])],
        ),

        # ---------------- transformation ----------------
        StepSpec(
            key="log_transform",
            name="Log transform",
            category="transformation",
            description="Apply log1p, shifting first if the column contains non-positive values.",
            builder=lambda columns=None, base="natural", **_: T.FrameTransformer(T.LogTransformer(base), columns),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Right-skewed positive quantities: income, spend, counts, durations.",
            caveats=["Results are on the log scale — remember to back-transform before reporting rands"],
            params=[_p("base", "categorical", "natural", choices=["natural", "log10", "log2"])],
        ),
        StepSpec(
            key="yeo_johnson",
            name="Yeo-Johnson power transform",
            category="transformation",
            description="Fit a power transform that makes a column as close to normal as possible.",
            builder=lambda columns=None, standardize=True, **_: T.FrameTransformer(
                PowerTransformer(method="yeo-johnson", standardize=standardize), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Skewed data that includes zero or negative values (where Box-Cox cannot be used).",
            caveats=["The transformed values are no longer in the original units"],
            params=[_p("standardize", "bool", True)],
        ),
        StepSpec(
            key="box_cox",
            name="Box-Cox power transform",
            category="transformation",
            description="Classic power transform towards normality. Strictly positive data only.",
            builder=lambda columns=None, standardize=True, **_: T.FrameTransformer(
                PowerTransformer(method="box-cox", standardize=standardize), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Strictly positive skewed columns.",
            caveats=["Fails outright on zero or negative values — use Yeo-Johnson instead"],
            params=[_p("standardize", "bool", True)],
        ),
        StepSpec(
            key="quantile_normal",
            name="Quantile transform to normal",
            category="transformation",
            description="Map each column onto a normal distribution by rank.",
            builder=lambda columns=None, n_quantiles=1000, **_: T.FrameTransformer(
                QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles, random_state=42), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Stubbornly non-normal columns where a power transform is not enough.",
            caveats=["Non-linear and rank-based, so relationships between variables are distorted"],
            params=[_p("n_quantiles", "int", 1000, 10, 5000)],
        ),
        StepSpec(
            key="binning",
            name="Bin into intervals (discretise)",
            category="transformation",
            description="Convert a continuous column into ordered bands.",
            builder=lambda columns=None, n_bins=5, strategy="quantile", **_: T.FrameTransformer(
                KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy, quantile_method="linear"), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="When bands are easier for the business to act on than a continuous score.",
            caveats=["Throws away information", "Bin edges are arbitrary decisions"],
            params=[
                _p("n_bins", "int", 5, 2, 20),
                _p("strategy", "categorical", "quantile", choices=["quantile", "uniform", "kmeans"]),
            ],
        ),
        StepSpec(
            key="binarize",
            name="Binarise at a threshold",
            category="transformation",
            description="Turn a numeric column into 0/1 above and below a cut-off.",
            builder=lambda columns=None, threshold=0.0, **_: T.FrameTransformer(
                Binarizer(threshold=threshold), columns
            ),
            applies_to=NUMERIC,
            when_to_use="When only 'present / absent' or 'above / below target' matters.",
            params=[_p("threshold", "float", 0.0)],
        ),

        # ---------------- encoding ----------------
        StepSpec(
            key="one_hot_encode",
            name="One-hot encode",
            category="encoding",
            description="Create one 0/1 indicator column per category.",
            builder=lambda columns=None, drop=None, min_frequency=None, **_: T.FrameTransformer(
                OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist",
                              drop=drop, min_frequency=min_frequency), columns
            ),
            applies_to=CATEGORICAL,
            leakage_safe=False,
            when_to_use="The default for nominal categories with a manageable number of levels.",
            caveats=["Column count explodes with high-cardinality columns", "Use drop='first' for linear models to avoid the dummy trap"],
            params=[
                _p("drop", "categorical", None, choices=[None, "first", "if_binary"]),
                _p("min_frequency", "float", None, 0.0, 0.2, description="Fold categories rarer than this into one bucket."),
            ],
        ),
        StepSpec(
            key="ordinal_encode",
            name="Ordinal encode",
            category="encoding",
            description="Map categories to integers following a meaningful order.",
            builder=lambda columns=None, categories="auto", **_: T.FrameTransformer(
                OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1), columns
            ),
            applies_to=[SemanticType.CATEGORICAL_ORDINAL, SemanticType.BINARY],
            leakage_safe=False,
            when_to_use="Genuinely ordered categories: low/medium/high, bronze/silver/gold.",
            caveats=["Applying this to unordered categories invents an order the model will believe"],
            params=[_p("categories", "categorical", "auto", description="Explicit level order, or 'auto' for alphabetical.")],
        ),
        StepSpec(
            key="frequency_encode",
            name="Frequency encode",
            category="encoding",
            description="Replace each category with how often it occurs.",
            builder=lambda columns=None, normalise=True, **_: T.FrameTransformer(
                T.FrequencyEncoder(normalise), columns
            ),
            applies_to=CATEGORICAL,
            leakage_safe=False,
            when_to_use="High-cardinality columns where rarity itself carries signal.",
            caveats=["Two unrelated categories with the same frequency become indistinguishable"],
            params=[_p("normalise", "bool", True)],
        ),
        StepSpec(
            key="target_encode",
            name="Target (mean) encode",
            category="encoding",
            description="Replace each category with the smoothed average target for that category.",
            builder=lambda columns=None, smoothing=10.0, **_: T.FrameTransformer(
                T.TargetEncoder(smoothing=smoothing), columns
            ),
            applies_to=CATEGORICAL,
            leakage_safe=False,
            needs_target=True,
            when_to_use="High-cardinality categories in a supervised problem.",
            caveats=[
                "Leaks the target badly if fitted outside the cross-validation fold",
                "This platform always fits it inside the fold — do not replicate it manually on the full dataset",
            ],
            params=[_p("smoothing", "float", 10.0, 1.0, 100.0, description="Higher pulls rare categories harder towards the overall mean.")],
        ),
        StepSpec(
            key="group_rare_categories",
            name="Group rare categories into 'Other'",
            category="encoding",
            description="Fold categories below a frequency threshold into one level.",
            builder=lambda columns=None, min_frequency=0.01, **_: T.FrameTransformer(
                T.RareCategoryGrouper(min_frequency), columns
            ),
            applies_to=CATEGORICAL,
            leakage_safe=False,
            when_to_use="Before one-hot encoding a column with a long tail of rare levels.",
            params=[_p("min_frequency", "float", 0.01, 0.001, 0.2)],
        ),

        # ---------------- feature engineering ----------------
        StepSpec(
            key="datetime_features",
            name="Extract date/time features",
            category="feature_engineering",
            description="Expand a date column into year, month, day, weekday, quarter and cyclical terms.",
            builder=lambda columns=None, cyclical=True, **_: T.FrameTransformer(
                T.DateTimeFeatures(cyclical=cyclical), columns
            ),
            applies_to=[SemanticType.DATETIME],
            when_to_use="Whenever a date column is present — raw timestamps are useless to most models.",
            params=[_p("cyclical", "bool", True, description="Add sin/cos terms so December sits next to January.")],
        ),
        StepSpec(
            key="text_features",
            name="Extract text features",
            category="feature_engineering",
            description="Summarise free text as length, word count, digit share and vocabulary richness.",
            builder=lambda columns=None, **_: T.FrameTransformer(T.TextFeatures(), columns),
            applies_to=[SemanticType.TEXT],
            when_to_use="A light-touch way to use text columns without exploding the feature space.",
            caveats=["Captures none of the meaning — use TF-IDF plus SVD when the wording matters"],
        ),
        StepSpec(
            key="polynomial_features",
            name="Polynomial features",
            category="feature_engineering",
            description="Add squared, cubed and interaction terms.",
            builder=lambda columns=None, degree=2, interaction_only=False, **_: T.FrameTransformer(
                PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False), columns
            ),
            applies_to=NUMERIC,
            when_to_use="Letting a linear model capture curvature and interactions.",
            caveats=["Feature count grows very quickly", "Nearly always needs regularisation afterwards"],
            params=[
                _p("degree", "int", 2, 2, 4),
                _p("interaction_only", "bool", False, description="Cross terms only, no powers."),
            ],
        ),
        StepSpec(
            key="interaction_features",
            name="Pairwise interaction terms",
            category="feature_engineering",
            description="Multiply chosen numeric columns together (optionally divide them too).",
            builder=lambda columns=None, max_features=8, include_ratios=False, **_: T.FrameTransformer(
                T.InteractionFeatures(max_features, include_ratios), columns
            ),
            applies_to=NUMERIC,
            when_to_use="When you suspect two drivers only matter in combination.",
            params=[_p("max_features", "int", 8, 2, 20), _p("include_ratios", "bool", False)],
        ),
        StepSpec(
            key="aggregation_features",
            name="Row aggregation features",
            category="feature_engineering",
            description="Add row-wise totals, means, spread and range across numeric columns.",
            builder=lambda columns=None, **_: T.FrameTransformer(T.AggregationFeatures(), columns),
            applies_to=NUMERIC,
            when_to_use="Wide numeric tables such as monthly spend by category.",
        ),

        # ---------------- feature selection ----------------
        StepSpec(
            key="select_k_best_regression",
            name="Select K best features (regression)",
            category="feature_selection",
            description="Keep the K features with the strongest linear association with a numeric target.",
            builder=lambda columns=None, k=10, **_: T.FrameTransformer(
                SelectKBest(score_func=f_regression, k=k), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            needs_target=True,
            when_to_use="Trimming a wide numeric table before modelling.",
            caveats=["Only sees one feature at a time — misses combinations", "Only detects linear association"],
            params=[_p("k", "int", 10, 1, 100)],
        ),
        StepSpec(
            key="select_k_best_classification",
            name="Select K best features (classification)",
            category="feature_selection",
            description="Keep the K features that best separate the classes by ANOVA F-test.",
            builder=lambda columns=None, k=10, **_: T.FrameTransformer(
                SelectKBest(score_func=f_classif, k=k), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            needs_target=True,
            when_to_use="Trimming features before a classifier.",
            caveats=["Univariate — a feature useful only in combination will be discarded"],
            params=[_p("k", "int", 10, 1, 100)],
        ),
        StepSpec(
            key="select_mutual_info",
            name="Select by mutual information",
            category="feature_selection",
            description="Keep features with the highest mutual information with the target.",
            builder=lambda columns=None, k=10, **_: T.FrameTransformer(
                SelectKBest(score_func=mutual_info_regression, k=k), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            needs_target=True,
            when_to_use="When relationships are nonlinear and correlation-based selection would miss them.",
            caveats=["Slower", "Estimates are noisy on small samples"],
            params=[_p("k", "int", 10, 1, 100)],
        ),
        StepSpec(
            key="drop_correlated",
            name="Drop highly correlated features",
            category="feature_selection",
            description="Where two features correlate above a threshold, keep one and drop the other.",
            builder=lambda columns=None, threshold=0.95, **_: T.FrameTransformer(
                CorrelationFilter(threshold=threshold), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Reducing multicollinearity before fitting an interpretable linear model.",
            caveats=["Which of the pair survives is arbitrary — say so when reporting coefficients"],
            params=[_p("threshold", "float", 0.95, 0.7, 0.999)],
        ),

        # ---------------- dimensionality ----------------
        StepSpec(
            key="pca_reduce",
            name="PCA dimensionality reduction",
            category="dimensionality",
            description="Replace correlated numeric columns with a smaller set of components.",
            builder=lambda columns=None, n_components=0.95, **_: T.FrameTransformer(
                T.NamedPCA(n_components=n_components, random_state=42), columns
            ),
            applies_to=NUMERIC,
            leakage_safe=False,
            when_to_use="Severe multicollinearity, or far more columns than rows.",
            caveats=["Components mix the original variables, so individual-variable interpretation is lost"],
            params=[_p("n_components", "float", 0.95, 0.5, 0.999,
                       description="A fraction keeps that much variance; an integer keeps that many components.")],
        ),
        StepSpec(
            key="coerce_numeric",
            name="Force numeric and fill any remaining gaps",
            category="cleaning",
            description="Final safety net before the model: everything numeric, nothing infinite, nothing missing.",
            builder=lambda fill_value=0.0, **_: T.NumericCoercer(fill_value=fill_value),
            when_to_use="Automatically appended to every pipeline; you should not need to add it yourself.",
            params=[_p("fill_value", "float", 0.0)],
        ),
    ]
    for step in steps:
        STEPS.register(step)


class CorrelationFilter:
    """Drop one of every pair of features correlating above a threshold."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        import pandas as pd

        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        corr = frame.corr().abs()
        drop: set[str] = set()
        columns = list(corr.columns)
        for i, a in enumerate(columns):
            if a in drop:
                continue
            for b in columns[i + 1:]:
                if b in drop:
                    continue
                value = corr.loc[a, b]
                if value == value and value >= self.threshold:
                    drop.add(b)
        self.kept_ = [c for c in columns if c not in drop]
        self.dropped_ = sorted(drop)
        return self

    def transform(self, X):
        import pandas as pd

        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return frame.reindex(columns=self.kept_)

    def get_feature_names_out(self, input_features=None):
        import numpy as np

        return np.asarray(self.kept_, dtype=object)


_register_all()
