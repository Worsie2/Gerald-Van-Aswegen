"""Scikit-learn compatible wrappers for algorithms that need a little glue.

Everything here follows the estimator API (``fit`` / ``predict`` / ``get_params``)
so the rest of the platform can treat these exactly like any other model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# --------------------------------------------------------------------------
# regression adapters
# --------------------------------------------------------------------------

class PolynomialRegressor(BaseEstimator, RegressorMixin):
    """Polynomial expansion followed by a ridge fit, exposed as one estimator."""

    def __init__(self, degree: int = 2, alpha: float = 1.0, include_bias: bool = False):
        self.degree = degree
        self.alpha = alpha
        self.include_bias = include_bias

    def fit(self, X, y):
        from sklearn.linear_model import Ridge

        self.pipeline_ = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline_.fit(X, y)
        self.coef_ = self.pipeline_.named_steps["ridge"].coef_
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def feature_names_out(self, input_features=None):
        return self.pipeline_.named_steps["poly"].get_feature_names_out(input_features)


class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    """OLS with the inference statsmodels provides: p-values, CIs, diagnostics."""

    def __init__(self, add_constant: bool = True, robust_se: bool = False):
        self.add_constant = add_constant
        self.robust_se = robust_se

    def fit(self, X, y):
        import statsmodels.api as sm

        X_arr = np.asarray(X, dtype=float)
        self.feature_names_in_ = getattr(X, "columns", None)
        design = sm.add_constant(X_arr, has_constant="add") if self.add_constant else X_arr
        model = sm.OLS(np.asarray(y, dtype=float), design)
        self.results_ = model.fit(cov_type="HC3") if self.robust_se else model.fit()
        params = np.asarray(self.results_.params)
        self.intercept_ = float(params[0]) if self.add_constant else 0.0
        self.coef_ = params[1:] if self.add_constant else params
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X):
        import statsmodels.api as sm

        X_arr = np.asarray(X, dtype=float)
        design = sm.add_constant(X_arr, has_constant="add") if self.add_constant else X_arr
        return np.asarray(self.results_.predict(design))

    def inference_table(self, feature_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Coefficients with standard errors, p-values and 95% confidence intervals."""
        res = self.results_
        names = list(feature_names or [])
        if self.add_constant:
            names = ["(intercept)"] + names
        conf = np.asarray(res.conf_int())
        rows = []
        for i, coef in enumerate(np.asarray(res.params)):
            rows.append(
                {
                    "term": names[i] if i < len(names) else f"x{i}",
                    "coefficient": float(coef),
                    "std_error": float(np.asarray(res.bse)[i]),
                    "t_statistic": float(np.asarray(res.tvalues)[i]),
                    "p_value": float(np.asarray(res.pvalues)[i]),
                    "ci_lower": float(conf[i, 0]),
                    "ci_upper": float(conf[i, 1]),
                    "significant_at_5pct": bool(np.asarray(res.pvalues)[i] < 0.05),
                }
            )
        return rows

    def diagnostics(self) -> dict[str, float]:
        res = self.results_
        out = {
            "r_squared": float(res.rsquared),
            "adj_r_squared": float(res.rsquared_adj),
            "f_statistic": float(res.fvalue) if res.fvalue is not None else float("nan"),
            "f_pvalue": float(res.f_pvalue) if res.f_pvalue is not None else float("nan"),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "n_observations": int(res.nobs),
        }
        try:
            from statsmodels.stats.stattools import durbin_watson, jarque_bera

            out["durbin_watson"] = float(durbin_watson(res.resid))
            jb = jarque_bera(res.resid)
            out["jarque_bera_stat"] = float(jb[0])
            out["jarque_bera_pvalue"] = float(jb[1])
        except Exception:
            pass
        return out


def DefaultStackingRegressor(cv: int = 5, random_state: int = 42, **_: Any):
    """A sensible stack: regularised linear + forest + boosting, ridge meta-learner."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import Ridge, RidgeCV

    return StackingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=1.0)),
            ("forest", RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)),
            ("boost", GradientBoostingRegressor(random_state=random_state)),
        ],
        final_estimator=RidgeCV(),
        cv=cv,
        n_jobs=-1,
    )


def DefaultVotingRegressor(random_state: int = 42, **_: Any):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import Ridge

    return VotingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=1.0)),
            ("forest", RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)),
            ("boost", GradientBoostingRegressor(random_state=random_state)),
        ],
        n_jobs=-1,
    )


def DefaultStackingClassifier(cv: int = 5, random_state: int = 42, **_: Any):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression

    return StackingClassifier(
        estimators=[
            ("logreg", LogisticRegression(max_iter=2000)),
            ("forest", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)),
            ("boost", GradientBoostingClassifier(random_state=random_state)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=cv,
        n_jobs=-1,
    )


def DefaultVotingClassifier(voting: str = "soft", random_state: int = 42, **_: Any):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression

    return VotingClassifier(
        estimators=[
            ("logreg", LogisticRegression(max_iter=2000)),
            ("forest", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)),
            ("boost", GradientBoostingClassifier(random_state=random_state)),
        ],
        voting=voting,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------
# clustering adapters
# --------------------------------------------------------------------------

class KMedoids(BaseEstimator, ClusterMixin):
    """K-Medoids (PAM-style, alternating heuristic).

    Cluster centres are actual observations, which makes each cluster easy to
    describe ("this real customer typifies the segment") and far less sensitive
    to outliers than K-Means.
    """

    def __init__(self, n_clusters: int = 3, metric: str = "euclidean", max_iter: int = 300, random_state: int = 42):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.metrics import pairwise_distances

        X = np.asarray(X, dtype=float)
        n = len(X)
        if self.n_clusters > n:
            raise ValueError(f"Cannot form {self.n_clusters} clusters from {n} rows.")
        distances = pairwise_distances(X, metric=self.metric)
        rng = np.random.default_rng(self.random_state)
        # k-means++ style seeding on the distance matrix
        medoids = [int(rng.integers(n))]
        for _ in range(self.n_clusters - 1):
            d_min = distances[:, medoids].min(axis=1)
            probs = d_min ** 2
            total = probs.sum()
            medoids.append(int(rng.choice(n, p=probs / total)) if total > 0 else int(rng.integers(n)))
        medoids = np.array(medoids)

        for _ in range(self.max_iter):
            labels = np.argmin(distances[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for k in range(self.n_clusters):
                members = np.flatnonzero(labels == k)
                if members.size == 0:
                    continue
                costs = distances[np.ix_(members, members)].sum(axis=1)
                new_medoids[k] = members[int(np.argmin(costs))]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = np.argmin(distances[:, medoids], axis=1)
        self.inertia_ = float(distances[np.arange(n), medoids[self.labels_]].sum())
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(np.asarray(X, dtype=float), self.cluster_centers_, metric=self.metric)
        return np.argmin(distances, axis=1)


class FuzzyCMeans(BaseEstimator, ClusterMixin):
    """Fuzzy C-Means: every row gets a membership share in every cluster.

    Useful when segments genuinely overlap — a customer can be 60% "value
    seeker" and 40% "premium" rather than being forced into one box.
    """

    def __init__(self, n_clusters: int = 3, m: float = 2.0, max_iter: int = 300,
                 tol: float = 1e-5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        u = rng.random((n, self.n_clusters))
        u /= u.sum(axis=1, keepdims=True)

        centers = np.zeros((self.n_clusters, X.shape[1]))
        for _ in range(self.max_iter):
            um = u ** self.m
            centers = (um.T @ X) / np.maximum(um.sum(axis=0)[:, None], 1e-12)
            dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            dist = np.maximum(dist, 1e-12)
            power = 2.0 / (self.m - 1.0)
            inv = (1.0 / dist) ** power
            new_u = inv / inv.sum(axis=1, keepdims=True)
            if np.linalg.norm(new_u - u) < self.tol:
                u = new_u
                break
            u = new_u

        self.cluster_centers_ = centers
        self.membership_ = u
        self.labels_ = np.argmax(u, axis=1)
        # Fuzzy partition coefficient: 1.0 = crisp clusters, 1/k = maximally fuzzy
        self.partition_coefficient_ = float((u ** 2).sum() / n)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dist = np.maximum(np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2), 1e-12)
        inv = (1.0 / dist) ** (2.0 / (self.m - 1.0))
        return np.argmax(inv / inv.sum(axis=1, keepdims=True), axis=1)


class LatentClassAnalysis(BaseEstimator, ClusterMixin):
    """Latent Class Analysis: a mixture of independent Bernoulli items.

    This is not a distance-based clustering method — it fits P(item = 1 |
    class) for every indicator, for every class, by EM, on the assumption
    that within a class the items are independent (the "local independence"
    assumption LCA is built on). What K-Means calls a centroid, LCA calls a
    class's item-response profile: "63% of class 2 has service_tier_premium",
    not "class 2's average is 0.63 standard deviations from the mean".

    LCA needs binary or categorical indicators, not continuous measurements —
    the preprocessing pipeline one-hot-encodes categoricals into 0/1 columns
    the same way for every clustering model, but does not know to leave
    continuous columns out for this one. Rather than fail on them, a column
    that is not already binary is median-split at fit time (the threshold is
    stored and reused at predict time) — a real, standard way to bring a
    continuous item into an LCA, but a choice made for you, not a neutral
    default. It shows up in the model's limitations for exactly that reason.
    """

    def __init__(self, n_clusters: int = 3, n_init: int = 10, max_iter: int = 200,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _binarize(self, X: np.ndarray, fitting: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if fitting:
            is_binary = np.array([set(np.unique(col)) <= {0.0, 1.0} for col in X.T])
            thresholds = np.where(is_binary, 0.5, np.median(X, axis=0))
            self.thresholds_ = thresholds
            self.was_binary_ = is_binary
        return (X > self.thresholds_).astype(float)

    @staticmethod
    def _log_likelihood(Xb: np.ndarray, log_prior: np.ndarray,
                        log_theta: np.ndarray, log_one_minus_theta: np.ndarray) -> np.ndarray:
        # (n, K): log P(row, class=k) for every row and class at once.
        return log_prior[None, :] + Xb @ log_theta.T + (1.0 - Xb) @ log_one_minus_theta.T

    def _fit_once(self, Xb: np.ndarray, rng: np.random.Generator):
        from scipy.special import logsumexp

        n, j = Xb.shape
        k = self.n_clusters
        eps = 1e-6
        theta = rng.uniform(0.25, 0.75, size=(k, j))
        prior = np.full(k, 1.0 / k)
        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            theta_c = np.clip(theta, eps, 1 - eps)
            log_joint = self._log_likelihood(Xb, np.log(prior), np.log(theta_c), np.log1p(-theta_c))
            log_norm = logsumexp(log_joint, axis=1)
            total_ll = float(log_norm.sum())
            resp = np.exp(log_joint - log_norm[:, None])

            weight = resp.sum(axis=0)
            prior = np.maximum(weight, eps) / n
            theta = (resp.T @ Xb) / np.maximum(weight, eps)[:, None]

            if total_ll - prev_ll < self.tol * abs(prev_ll or 1.0):
                prev_ll = total_ll
                break
            prev_ll = total_ll

        return prior, theta, resp, prev_ll, iteration

    def fit(self, X, y=None):
        Xb = self._binarize(X, fitting=True)
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(max(1, self.n_init)):
            result = self._fit_once(Xb, rng)
            if best is None or result[3] > best[3]:
                best = result

        prior, theta, resp, total_ll, n_iter = best
        order = np.argsort(-prior)  # largest class first, so labels read consistently run to run
        self.class_prior_ = prior[order]
        self.item_probs_ = theta[order]
        self.responsibilities_ = resp[:, order]
        self.labels_ = np.argmax(self.responsibilities_, axis=1)
        self.lower_bound_ = total_ll / len(Xb)
        self.n_iter_ = n_iter
        self.n_features_in_ = Xb.shape[1]
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def _responsibilities(self, X) -> np.ndarray:
        from scipy.special import logsumexp

        Xb = self._binarize(X, fitting=False)
        theta_c = np.clip(self.item_probs_, 1e-6, 1 - 1e-6)
        log_joint = self._log_likelihood(Xb, np.log(self.class_prior_), np.log(theta_c), np.log1p(-theta_c))
        return np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

    def predict(self, X):
        return np.argmax(self._responsibilities(X), axis=1)

    def predict_proba(self, X):
        return self._responsibilities(X)

    def score(self, X, y=None) -> float:
        """Average per-row log-likelihood, the same quantity sklearn's
        GaussianMixture.score returns — what bic()/aic() are built from."""
        from scipy.special import logsumexp

        Xb = self._binarize(X, fitting=False)
        theta_c = np.clip(self.item_probs_, 1e-6, 1 - 1e-6)
        log_joint = self._log_likelihood(Xb, np.log(self.class_prior_), np.log(theta_c), np.log1p(-theta_c))
        return float(logsumexp(log_joint, axis=1).mean())

    def _n_parameters(self) -> int:
        k, j = self.item_probs_.shape
        return (k - 1) + k * j  # class priors (sum to 1) + one item-response rate per class per item

    def bic(self, X) -> float:
        n = len(np.asarray(X))
        return -2.0 * self.score(X) * n + self._n_parameters() * np.log(n)

    def aic(self, X) -> float:
        n = len(np.asarray(X))
        return -2.0 * self.score(X) * n + 2.0 * self._n_parameters()


# --------------------------------------------------------------------------
# anomaly detection adapters
# --------------------------------------------------------------------------

class ZScoreOutlierDetector(BaseEstimator):
    """Flags rows where any feature sits more than `threshold` SDs from its mean."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def decision_function(self, X):
        z = np.abs((np.asarray(X, dtype=float) - self.mean_) / self.std_)
        return self.threshold - z.max(axis=1)  # negative = anomalous, matches sklearn's sign

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


class IQROutlierDetector(BaseEstimator):
    """Flags rows outside Q1 - k·IQR / Q3 + k·IQR on any feature."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        self.iqr_ = np.maximum(self.q3_ - self.q1_, 1e-12)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        below = (self.q1_ - X) / self.iqr_
        above = (X - self.q3_) / self.iqr_
        excess = np.maximum(below, above).max(axis=1)
        return self.factor - excess

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


class MahalanobisOutlierDetector(BaseEstimator):
    """Multivariate outliers via robust Mahalanobis distance (chi-square cut-off)."""

    def __init__(self, contamination: float = 0.05, support_fraction: float | None = None):
        self.contamination = contamination
        self.support_fraction = support_fraction

    def fit(self, X, y=None):
        from sklearn.covariance import MinCovDet

        X = np.asarray(X, dtype=float)
        self.estimator_ = MinCovDet(support_fraction=self.support_fraction, random_state=42).fit(X)
        distances = self.estimator_.mahalanobis(X)
        self.threshold_ = float(np.quantile(distances, 1 - self.contamination))
        return self

    def decision_function(self, X):
        return self.threshold_ - self.estimator_.mahalanobis(np.asarray(X, dtype=float))

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


class DBSCANOutlierDetector(BaseEstimator):
    """Treats DBSCAN's noise points (-1) as anomalies."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        from sklearn.cluster import DBSCAN

        self.model_ = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(np.asarray(X, dtype=float))
        self.labels_ = self.model_.labels_
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.where(self.labels_ == -1, -1, 1)

    def predict(self, X):
        return self.fit_predict(X)
