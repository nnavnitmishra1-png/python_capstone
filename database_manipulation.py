"""
database_manipulation.py
====================
Loading, validation, cleaning and feature engineering for the
"Analysis of Factors Associated with Diabetes Status" project.

Dataset : Diabetes Health Indicators Dataset (CDC BRFSS 2015) - Kaggle, Alex Teboul
File    : diabetesData.csv (the balanced 50/50 version: 35,346 diabetic + 35,346 non-diabetic)
Target  : Diabetes_binary  (0 = no diabetes, 1 = pre-diabetes or diabetes)

Everything here is written with vectorised Pandas / NumPy operations
(no row-by-row Python loops) -> efficient on ~250,000 rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants describing the dataset (data dictionary)
# --------------------------------------------------------------------------
TARGET = "Diabetes_binary"

BINARY_COLS = [
    TARGET, "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]

# Plausible value ranges used for validation (BRFSS coding)
VALID_RANGES = {
    "BMI": (12, 98),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
    12: "75-79", 13: "80+",
}
GENHLTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION_LABELS = {
    1: "None/Kindergarten", 2: "Elementary", 3: "Some high school",
    4: "High school grad", 5: "Some college", 6: "College grad",
}
INCOME_LABELS = {
    1: "<$10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k",
    5: "$25-35k", 6: "$35-50k", 7: "$50-75k", 8: "$75k+",
}
BMI_BINS = [0, 18.5, 25, 30, 35, np.inf]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese I", "Obese II+"]


# --------------------------------------------------------------------------
# 1. Loading and inspection
# --------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and make sure the expected columns are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download the CSV from Kaggle "
            "and place it in the data/ folder (see README.md)."
        )
    df = pd.read_csv(path)
    expected = set(BINARY_COLS) | set(VALID_RANGES)
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing_cols)}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Quick structural summary used in the report (shape, NaNs, duplicates)."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)),
    }


def head_preview(df: pd.DataFrame, n_rows: int = 5, n_cols: int = 8) -> pd.DataFrame:
    """Positional slicing with .iloc - first n rows / first n columns."""
    return df.iloc[:n_rows, :n_cols]


# --------------------------------------------------------------------------
# 2. Cleaning steps (each step is a separate, testable function)
# --------------------------------------------------------------------------
def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Missing values strategy
    - rows without the target label are dropped (cannot be used for analysis)
    - continuous / count variables  -> median (robust to skew)
    - binary / ordinal variables    -> mode (keeps valid category codes)
    The BRFSS extract is normally complete, so this is a safeguard that also
    makes the pipeline reusable on a modified dataset.
    """
    n_missing = int(df.isna().sum().sum())
    if n_missing == 0:
        return df, 0

    df = df.dropna(subset=[TARGET]).copy()
    median_cols = ["BMI", "MentHlth", "PhysHlth"]
    for col in df.columns[df.isna().any()]:
        fill = df[col].median() if col in median_cols else df[col].mode().iloc[0]
        df[col] = df[col].fillna(fill)
    return df, n_missing


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate rows.
    Justification: the extract has no respondent ID; with 21 mostly binary
    columns identical rows can be genuine different people, so the project
    also runs a sensitivity analysis (statistics_analysis.compare_or_tables)
    to show conclusions do not depend on this choice.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def validate_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with values outside the legal coding range (boolean masks + .loc)."""
    valid = pd.Series(True, index=df.index)
    for col, (low, high) in VALID_RANGES.items():
        valid &= df[col].between(low, high)
    valid &= df[BINARY_COLS].isin([0, 1]).all(axis=1)
    n_invalid = int((~valid).sum())
    return df.loc[valid].reset_index(drop=True), n_invalid


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences: [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def treat_bmi_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    BMI outliers
    - flag values outside the IQR fences (column BMI_Outlier)
    - create BMI_Winsor: BMI clipped to the 1st-99th percentile (np.clip)
    Justification: extreme BMI values (e.g. 80+) are medically possible, so
    they are NOT deleted; the original BMI is kept for descriptive plots and
    the winsorised version is used in the regression to limit leverage.
    MentHlth / PhysHlth (0-30 days) are bounded, so spikes at 0 and 30 are
    valid answers and are not treated as outliers.
    """
    low, high = iqr_bounds(df["BMI"])
    p1, p99 = np.percentile(df["BMI"], [1, 99])
    df = df.copy()
    df["BMI_Outlier"] = ~df["BMI"].between(low, high)
    df["BMI_Winsor"] = np.clip(df["BMI"], p1, p99)
    info = {
        "bmi_iqr_low": round(float(low), 2),
        "bmi_iqr_high": round(float(high), 2),
        "bmi_outliers_flagged": int(df["BMI_Outlier"].sum()),
        "bmi_winsor_p1": float(p1),
        "bmi_winsor_p99": float(p99),
    }
    return df, info


# --------------------------------------------------------------------------
# 3. Feature engineering (vectorised)
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create readable labels and composite indicators for EDA / modelling."""
    df = df.copy()

    # readable categorical labels (ordered so plots follow natural order)
    df["Age_Group"] = pd.Categorical(
        df["Age"].map(AGE_LABELS), categories=list(AGE_LABELS.values()), ordered=True)
    df["GenHlth_Label"] = pd.Categorical(
        df["GenHlth"].map(GENHLTH_LABELS), categories=list(GENHLTH_LABELS.values()), ordered=True)
    df["Education_Label"] = pd.Categorical(
        df["Education"].map(EDUCATION_LABELS), categories=list(EDUCATION_LABELS.values()), ordered=True)
    df["Income_Label"] = pd.Categorical(
        df["Income"].map(INCOME_LABELS), categories=list(INCOME_LABELS.values()), ordered=True)
    df["BMI_Category"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    df["Sex_Label"] = np.where(df["Sex"] == 1, "Male", "Female")
    df["Diabetes_Label"] = np.where(df[TARGET] == 1, "Diabetic/Pre-diabetic", "Non-diabetic")

    # composite indicators (arithmetic on whole columns = vectorised)
    # Lifestyle score 0 (worst) - 4 (best): activity + fruits + veggies + non-smoker.
    # HvyAlcoholConsump is deliberately EXCLUDED: in this survey it shows an inverse
    # association with diabetes (sick-quitter effect), which would distort the score.
    healthy = ["PhysActivity", "Fruits", "Veggies"]
    df["Lifestyle_Score"] = df[healthy].sum(axis=1) + (1 - df["Smoker"])
    df["Comorbidity_Count"] = df[["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]].sum(axis=1)
    df["Unhealthy_Days"] = df["MentHlth"] + df["PhysHlth"]
    return df


def add_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """NumPy z-score standardisation: (x - mean) / std for each column."""
    df = df.copy()
    values = df[cols].to_numpy(dtype=float)
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    for i, col in enumerate(cols):
        df[f"{col}_z"] = z[:, i]
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer-valued float columns to small ints to save memory."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if col == "BMI_Winsor":
            continue
        if (df[col] % 1 == 0).all():
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# --------------------------------------------------------------------------
# 4. Group-wise summaries (used by EDA, plots and report)
# --------------------------------------------------------------------------
def prevalence_by(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """
    % of respondents with diabetes in every level of the grouping column(s).
    NOTE: the sample is balanced 50/50 by design, so this is NOT population
    prevalence - compare groups against the 50% baseline (relative comparison).
    """
    out = (df.groupby(by, observed=True)[TARGET]
             .agg(n="size", cases="sum", prevalence_pct="mean")
             .reset_index())
    out["prevalence_pct"] = out["prevalence_pct"] * 100
    return out


def high_risk_segment(df: pd.DataFrame) -> dict:
    """
    Compare the diabetes share in a 'high-risk profile' vs everybody else.
    Profile: high BP + high cholesterol + obese (BMI >= 30). Uses .loc masks.
    """
    mask = (df["HighBP"] == 1) & (df["HighChol"] == 1) & (df["BMI"] >= 30)
    seg = df.loc[mask, TARGET]
    rest = df.loc[~mask, TARGET]
    return {
        "segment_size": int(mask.sum()),
        "segment_prevalence_pct": float(seg.mean() * 100),
        "rest_prevalence_pct": float(rest.mean() * 100),
    }


def correlation_with_target(df: pd.DataFrame) -> pd.Series:
    """Spearman correlation of every numeric column with the target, sorted."""
    num = df.select_dtypes(include="number").drop(columns=["BMI_Outlier"], errors="ignore")
    corr = num.corr(method="spearman")[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------
# 5. Full pipeline
# --------------------------------------------------------------------------
def clean_pipeline(raw_path: str | Path, out_path: str | Path | None = None,
                   drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run every cleaning step in order and return (clean_df, cleaning_log)."""
    df = load_data(raw_path)
    log = {"raw": inspect_data(df)}

    df, log["missing_cells_found"] = handle_missing(df)
    if drop_duplicates:
        df, log["duplicates_removed"] = remove_duplicates(df)
    else:
        log["duplicates_removed"] = 0
    df, log["invalid_rows_removed"] = validate_ranges(df)
    df, log["bmi"] = treat_bmi_outliers(df)
    df = engineer_features(df)
    df = optimize_dtypes(df)

    log["clean"] = inspect_data(df)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log["saved_to"] = str(out_path)
    return df, log
