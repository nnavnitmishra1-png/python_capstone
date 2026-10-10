"""
visualization.py
================
All Matplotlib / Seaborn charts for the diabetes project.

Chart-choice rationale
    * bar chart        : compare % with diabetes across categories
    * histogram + box  : distribution and spread of a continuous variable (BMI)
    * heatmap          : many pairwise correlations at once
    * forest plot      : odds ratios with confidence intervals
    * ROC curve        : model discrimination
Every chart has a title, labelled axes and (where useful) value labels.
Charts show '% with diabetes' inside each group. The dataset is balanced 50/50 by design,
    so the 50% line is the baseline: bars above it = over-represented among diabetics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from database_manipulation import TARGET, prevalence_by

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Non-diabetic": "#4C9F70", "Diabetic/Pre-diabetic": "#D1495B"}
ACCENT = "#2E6F95"


def _finish(fig, path: Path, show: bool) -> Path:
    """Save the figure (and optionally display it)."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _bar_with_labels(ax, labels, values, color=ACCENT, fmt="%.1f%%", rotation=0):
    bars = ax.bar(range(len(values)), values, color=color)
    ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8)   # balanced-sample baseline
    ax.set_ylim(0, max(max(values), 50) * 1.18)


# --------------------------------------------------------------------------
def plot_class_balance(df, fig_dir: Path, show=False):
    counts = df["Diabetes_Label"].value_counts().reindex(list(PALETTE))
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    ax.bar_label(bars, labels=[f"{c:,}\n({p:.1f}%)" for c, p in zip(counts, pct)], padding=3)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Target classes (balanced 50/50 by design)")
    ax.set_ylabel("Number of respondents")
    return _finish(fig, fig_dir / "01_class_balance.png", show)


def plot_bmi_distribution(df, fig_dir: Path, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(data=df, x="BMI_Winsor", hue="Diabetes_Label", palette=PALETTE, stat="density",
                 common_norm=False, element="step", bins=40, ax=axes[0])
    axes[0].set_title("BMI distribution by diabetes status")
    axes[0].set_xlabel("BMI (winsorised at 1st-99th percentile)")
    axes[0].set_ylabel("Density")

    order = list(PALETTE)
    data = [df.loc[df["Diabetes_Label"] == k, "BMI"] for k in order]
    bp = axes[1].boxplot(data, patch_artist=True, flierprops={"markersize": 2, "alpha": 0.3})
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[k])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(order)
    axes[1].set_title("BMI spread and outliers (raw BMI)")
    axes[1].set_ylabel("BMI")
    return _finish(fig, fig_dir / "02_bmi_distribution.png", show)


def plot_prevalence_by_age(df, fig_dir: Path, show=False):
    prev = prevalence_by(df, "Age_Group")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _bar_with_labels(ax, prev["Age_Group"].astype(str), prev["prevalence_pct"])
    ax.set_title("Share with diabetes rises with age (50% = baseline)")
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("% with diabetes in group")
    return _finish(fig, fig_dir / "03_prevalence_by_age.png", show)


def plot_socioeconomic(df, fig_dir: Path, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, title in zip(axes, ["Income_Label", "Education_Label"],
                              ["Annual household income", "Education level"]):
        prev = prevalence_by(df, col)
        _bar_with_labels(ax, prev[col].astype(str), prev["prevalence_pct"], rotation=30)
        ax.set_title(f"% with diabetes by {title.lower()}")
        ax.set_xlabel(title)
        ax.set_ylabel("% with diabetes in group")
    return _finish(fig, fig_dir / "04_socioeconomic.png", show)


def plot_risk_factor_prevalence(df, fig_dir: Path, show=False):
    factors = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk", "Smoker",
               "HvyAlcoholConsump", "PhysActivity", "Fruits", "Veggies", "NoDocbcCost"]
    rows = []
    for f in factors:
        g = df.groupby(f)[TARGET].mean() * 100
        rows.append({"factor": f, "Absent": g.get(0, np.nan), "Present": g.get(1, np.nan)})
    t = pd.DataFrame(rows)
    t["gap"] = t["Present"] - t["Absent"]
    t = t.sort_values("gap")

    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - 0.2, t["Absent"], height=0.4, color=PALETTE["Non-diabetic"], label="Factor absent (0)")
    ax.barh(y + 0.2, t["Present"], height=0.4, color=PALETTE["Diabetic/Pre-diabetic"], label="Factor present (1)")
    ax.set_yticks(y)
    ax.set_yticklabels(t["factor"])
    ax.axvline(50, color="black", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.set_xlabel("% with diabetes in group")
    ax.set_title("% with diabetes with vs without each factor (50% = baseline)")
    ax.legend(loc="lower right")
    return _finish(fig, fig_dir / "05_risk_factor_prevalence.png", show)


def plot_correlation_heatmap(df, fig_dir: Path, show=False):
    cols = [TARGET, "HighBP", "HighChol", "BMI", "Age", "GenHlth", "PhysHlth", "MentHlth",
            "DiffWalk", "Income", "Education", "PhysActivity", "HeartDiseaseorAttack", "Smoker"]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"label": "Spearman correlation"}, ax=ax)
    ax.set_title("Spearman correlation matrix of key indicators")
    return _finish(fig, fig_dir / "06_correlation_heatmap.png", show)


def plot_general_health(df, fig_dir: Path, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    prev = prevalence_by(df, "GenHlth_Label")
    _bar_with_labels(axes[0], prev["GenHlth_Label"].astype(str), prev["prevalence_pct"])
    axes[0].set_title("% with diabetes by self-rated general health")
    axes[0].set_xlabel("General health")
    axes[0].set_ylabel("% with diabetes in group")

    means = df.groupby("Diabetes_Label")[["PhysHlth", "MentHlth"]].mean().reindex(list(PALETTE))
    means.plot(kind="bar", ax=axes[1], color=["#2E6F95", "#E0A458"], rot=0)
    axes[1].set_title("Average unhealthy days in the past 30 days")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean days")
    axes[1].legend(["Physical health", "Mental health"])
    return _finish(fig, fig_dir / "07_general_health.png", show)


def plot_bmi_category_by_sex(df, fig_dir: Path, show=False):
    prev = prevalence_by(df, ["BMI_Category", "Sex_Label"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=prev, x="BMI_Category", y="prevalence_pct", hue="Sex_Label",
                palette=["#C2185B", "#1976D2"], ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
    ax.set_title("% with diabetes by BMI category and sex")
    ax.set_xlabel("BMI category")
    ax.set_ylabel("% with diabetes in group")
    ax.legend(title="Sex")
    return _finish(fig, fig_dir / "08_bmi_category_by_sex.png", show)


def plot_comorbidity_lifestyle(df, fig_dir: Path, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, title in zip(axes, ["Comorbidity_Count", "Lifestyle_Score"],
                              ["Number of comorbidities (BP, cholesterol, stroke, heart disease)",
                               "Healthy-lifestyle score (0 = worst, 4 = best)"]):
        prev = prevalence_by(df, col)
        _bar_with_labels(ax, prev[col].astype(str), prev["prevalence_pct"])
        ax.set_title("% with diabetes by " + title.split(" (")[0].lower())
        ax.set_xlabel(title, fontsize=8)
        ax.set_ylabel("% with diabetes in group")
    return _finish(fig, fig_dir / "09_comorbidity_lifestyle.png", show)


def plot_odds_ratio_forest(or_table: pd.DataFrame, fig_dir: Path, show=False):
    t = or_table.sort_values("OR")
    y = np.arange(len(t))
    colors = np.where(t["OR"] > 1, PALETTE["Diabetic/Pre-diabetic"], PALETTE["Non-diabetic"])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.errorbar(t["OR"], y, xerr=[t["OR"] - t["CI_low"], t["CI_high"] - t["OR"]],
                fmt="none", ecolor="grey", capsize=3, zorder=1)
    ax.scatter(t["OR"], y, c=colors, s=45, zorder=2)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.set_xlabel("Adjusted odds ratio (log scale, 95% CI)")
    ax.set_title("Logistic regression: factors associated with diabetes\n(OR > 1 = higher odds)")
    return _finish(fig, fig_dir / "10_odds_ratio_forest.png", show)


def plot_roc(roc, auc: float, fig_dir: Path, show=False):
    fpr, tpr = roc
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"Logistic model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve on the 20% test set")
    ax.legend(loc="lower right")
    return _finish(fig, fig_dir / "11_roc_curve.png", show)


def generate_all_plots(df, or_table, roc, auc, fig_dir, show=False) -> list[Path]:
    """Create every figure; returns the list of saved paths."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_class_balance(df, fig_dir, show),
        plot_bmi_distribution(df, fig_dir, show),
        plot_prevalence_by_age(df, fig_dir, show),
        plot_socioeconomic(df, fig_dir, show),
        plot_risk_factor_prevalence(df, fig_dir, show),
        plot_correlation_heatmap(df, fig_dir, show),
        plot_general_health(df, fig_dir, show),
        plot_bmi_category_by_sex(df, fig_dir, show),
        plot_comorbidity_lifestyle(df, fig_dir, show),
        plot_odds_ratio_forest(or_table, fig_dir, show),
        plot_roc(roc, auc, fig_dir, show),
    ]
