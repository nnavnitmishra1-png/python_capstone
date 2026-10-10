"""
statistics_analysis.py
======================
Statistical analysis (SciPy + Statsmodels) for the diabetes project.

Why no ARIMA / forecasting?
    BRFSS 2015 is a single cross-sectional survey (no time dimension), so
    time-series forecasting would be wrong. The correct tools here are
    hypothesis tests, effect sizes and logistic regression (the target is binary).

Method choices (and why)
    * Shapiro-Wilk  : check normality of BMI -> decides parametric vs non-parametric
    * Mann-Whitney U: BMI / MentHlth / PhysHlth are skewed -> rank-based test
      (Welch t-test shown as a robustness check, valid for large n by the CLT)
    * Chi-square    : association between two categorical variables
    * Cramer's V, Cohen's d, rank-biserial r : EFFECT SIZES. With n ~ 230,000
      almost every p-value is tiny, so effect size says whether it matters.
    * Bonferroni    : correction for running many tests at once
    * Logistic regression (statsmodels) : adjusted odds ratios, so that a factor's
      effect is estimated while controlling for the others
    * VIF           : check multicollinearity before trusting the odds ratios
    * Train/test split + AUC : check the model generalises
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from database_manipulation import TARGET

BINARY_PREDICTORS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_PREDICTORS = ["Age", "Education", "Income", "GenHlth"]
CONTINUOUS_PREDICTORS = ["BMI", "MentHlth", "PhysHlth"]


# --------------------------------------------------------------------------
# Descriptive statistics and normality
# --------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = CONTINUOUS_PREDICTORS + ORDINAL_PREDICTORS
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurt()
    return desc.round(3)


def normality_check(df: pd.DataFrame, col: str = "BMI", sample_size: int = 5000,
                    seed: int = 42) -> pd.DataFrame:
    """Shapiro-Wilk on a random sample (the test is unreliable above ~5000 rows)."""
    rows = []
    for label, grp in df.groupby("Diabetes_Label"):
        sample = grp[col].sample(min(sample_size, len(grp)), random_state=seed)
        w, p = stats.shapiro(sample)
        rows.append({"group": label, "W": w, "p_value": p, "skewness": stats.skew(grp[col])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Hypothesis tests
# --------------------------------------------------------------------------
def compare_continuous(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Diabetic vs non-diabetic: Mann-Whitney U, Welch t-test, Cohen's d, rank-biserial r."""
    cols = cols or CONTINUOUS_PREDICTORS
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0]
    rows = []
    for col in cols:
        x1, x0 = pos[col].to_numpy(float), neg[col].to_numpy(float)
        n1, n0 = len(x1), len(x0)
        u = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        t = stats.ttest_ind(x1, x0, equal_var=False)
        pooled_sd = np.sqrt(((n1 - 1) * x1.var(ddof=1) + (n0 - 1) * x0.var(ddof=1)) / (n1 + n0 - 2))
        rows.append({
            "variable": col,
            "median_diabetic": np.median(x1), "median_non_diabetic": np.median(x0),
            "mean_diabetic": x1.mean(), "mean_non_diabetic": x0.mean(),
            "mannwhitney_p": u.pvalue, "welch_t": t.statistic, "welch_p": t.pvalue,
            "cohens_d": (x1.mean() - x0.mean()) / pooled_sd,
            "rank_biserial_r": 2 * u.statistic / (n1 * n0) - 1,
        })
    return pd.DataFrame(rows).set_index("variable")


def cramers_v_label(v: float) -> str:
    """Cohen's conventional thresholds for 1 degree of freedom."""
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def chi_square_tests(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Chi-square test of independence with Diabetes_binary for each variable.
    Adds Cramer's V, odds ratio (2x2 tables) and a Bonferroni-adjusted p-value.
    """
    cols = cols or (BINARY_PREDICTORS + ORDINAL_PREDICTORS)
    rows = []
    for col in cols:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.to_numpy().sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        odds_ratio = np.nan
        if ct.shape == (2, 2):
            t = ct.reindex(index=[0, 1], columns=[0, 1])
            odds_ratio = (t.loc[1, 1] * t.loc[0, 0]) / (t.loc[1, 0] * t.loc[0, 1])
        rows.append({"variable": col, "chi2": chi2, "dof": dof, "p_value": p,
                     "cramers_v": v, "effect_size": cramers_v_label(v),
                     "crude_odds_ratio": odds_ratio})
    out = pd.DataFrame(rows).set_index("variable")
    out["p_bonferroni"] = multipletests(out["p_value"], method="bonferroni")[1]
    out["significant_0.05"] = out["p_bonferroni"] < 0.05
    return out.sort_values("cramers_v", ascending=False)


def spearman_ordinal(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Spearman rank correlation of ordinal predictors with the binary target."""
    cols = cols or ORDINAL_PREDICTORS
    rows = []
    for col in cols:
        rho, p = stats.spearmanr(df[col], df[TARGET])
        rows.append({"variable": col, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows).set_index("variable")


def kruskal_bmi_by_health(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis: does BMI differ across the 5 self-rated health levels?"""
    groups = [g["BMI"].to_numpy() for _, g in df.groupby("GenHlth")]
    h, p = stats.kruskal(*groups)
    return {"H": float(h), "p_value": float(p)}


# --------------------------------------------------------------------------
# Logistic regression
# --------------------------------------------------------------------------
def prepare_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Predictor matrix. BMI is winsorised and scaled per 5 units (easier to interpret)."""
    feats = BINARY_PREDICTORS + ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    X = df[feats].astype(float).copy()
    X.insert(3, "BMI_per5", df["BMI_Winsor"].astype(float) / 5)
    y = df[TARGET].astype(float)
    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor (VIF > 5 suggests problematic collinearity)."""
    Xc = sm.add_constant(X)
    vif = [variance_inflation_factor(Xc.to_numpy(), i) for i in range(1, Xc.shape[1])]
    return pd.DataFrame({"feature": X.columns, "VIF": vif}).sort_values("VIF", ascending=False)


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney) formula."""
    ranks = stats.rankdata(scores)
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve with NumPy (handles tied scores)."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y_true[order]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = (1 + idx) - tps
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]


def fit_logistic(df: pd.DataFrame, seed: int = 42) -> dict:
    """
    1) Fit on ALL data  -> adjusted odds ratios with 95% CI (inference).
    2) Fit on 80% train, evaluate on 20% test -> AUC, sensitivity, specificity.
    """
    X, y = prepare_model_matrix(df)
    full = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)

    ci = np.exp(full.conf_int())
    or_table = pd.DataFrame({
        "OR": np.exp(full.params), "CI_low": ci[0], "CI_high": ci[1], "p_value": full.pvalues,
    }).drop(index="const")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = perm[:cut], perm[cut:]
    m = sm.Logit(y.iloc[tr], sm.add_constant(X.iloc[tr])).fit(disp=0, maxiter=200)
    scores = m.predict(sm.add_constant(X.iloc[te])).to_numpy()
    y_te = y.iloc[te].to_numpy()
    threshold = y.iloc[tr].mean()          # = ~0.5 because the sample is balanced
    pred = (scores >= threshold).astype(int)
    fpr, tpr = roc_curve_points(y_te, scores)

    metrics = {
        "n_observations": int(full.nobs),
        "pseudo_r2_mcfadden": float(full.prsquared),
        "aic": float(full.aic),
        "test_auc": auc_score(y_te, scores),
        "threshold": float(threshold),
        "sensitivity": float(((pred == 1) & (y_te == 1)).sum() / (y_te == 1).sum()),
        "specificity": float(((pred == 0) & (y_te == 0)).sum() / (y_te == 0).sum()),
        "accuracy": float((pred == y_te).mean()),
    }
    return {"or_table": or_table, "metrics": metrics, "vif": compute_vif(X),
            "roc": (fpr, tpr), "summary": full.summary().as_text()}


def compare_or_tables(a: pd.DataFrame, b: pd.DataFrame,
                      label_a: str = "with_duplicates", label_b: str = "deduplicated") -> pd.DataFrame:
    """Sensitivity analysis: do odds ratios change if duplicates are kept/removed?"""
    out = pd.DataFrame({f"OR_{label_a}": a["OR"], f"OR_{label_b}": b["OR"]})
    out["abs_difference"] = (out.iloc[:, 0] - out.iloc[:, 1]).abs()
    out["same_direction"] = (out.iloc[:, 0] > 1) == (out.iloc[:, 1] > 1)
    return out.sort_values("abs_difference", ascending=False)
