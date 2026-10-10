"""
main.py - Analysis of Factors Associated with Diabetes Status
=============================================================
Python Programming Laboratory (316U50L301) - Capstone Mini Project
Domain: Healthcare | Dataset: CDC BRFSS 2015 Diabetes Health Indicators

Run:
    python main.py                                  # full pipeline
    python main.py --input data/your_file.csv       # custom CSV path
    python main.py --show                           # also pop up the plots
    python main.py --explore HighBP                 # quick look at ONE variable

Pipeline
    1. load + inspect raw data            (database_manipulation.py)
    2. clean + engineer features + save   (database_manipulation.py)
    3. statistical analysis               (statistics_analysis.py)
    4. visualisations                     (visualization.py)
    5. write outputs/summary.txt + tables
"""

import argparse
import os
import sys
from pathlib import Path

# Use a non-interactive backend unless the user asked to see the plots.
if "--show" not in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # noqa: E402

import database_manipulation as dm  # noqa: E402
import statistics_analysis as sa  # noqa: E402
import visualization as viz  # noqa: E402

DEFAULT_INPUT = "data/diabetesData.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diabetes factors analysis (BRFSS 2015)")
    p.add_argument("--input", default=DEFAULT_INPUT, help="path to the raw CSV")
    p.add_argument("--output-dir", default="outputs", help="folder for results")
    p.add_argument("--show", action="store_true", help="display plots on screen")
    p.add_argument("--explore", metavar="COLUMN", help="quick analysis of a single column")
    return p.parse_args()


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def explore_variable(df: pd.DataFrame, col: str) -> None:
    """Viva helper: prevalence table + the appropriate test for any single column."""
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' not found. Available: {list(df.columns)}")
    banner(f"Quick exploration of '{col}'")
    if df[col].nunique() <= 15:
        print(dm.prevalence_by(df, col).round(2).to_string(index=False))
        print("\nChi-square test of independence:")
        print(sa.chi_square_tests(df, [col]).round(4).T.to_string())
    else:
        print(sa.compare_continuous(df, [col]).round(4).T.to_string())


def key_findings(df: pd.DataFrame) -> list[str]:
    """Plain-English findings computed from the data (no hard-coded numbers)."""
    age = dm.prevalence_by(df, "Age_Group").set_index("Age_Group")["prevalence_pct"]
    gen = dm.prevalence_by(df, "GenHlth_Label").set_index("GenHlth_Label")["prevalence_pct"]
    inc = dm.prevalence_by(df, "Income_Label").set_index("Income_Label")["prevalence_pct"]
    bmi = dm.prevalence_by(df, "BMI_Category").set_index("BMI_Category")["prevalence_pct"]
    bp = dm.prevalence_by(df, "HighBP").set_index("HighBP")["prevalence_pct"]
    seg = dm.high_risk_segment(df)
    return [
        f"Sample is balanced by design ({df[dm.TARGET].mean() * 100:.1f}% diabetic), so rates below are relative to a 50% baseline.",
        f"Diabetic share rises with age: {age.iloc[0]:.1f}% (18-24) vs {age.idxmax()} peak at {age.max():.1f}%.",
        f"Poor self-rated health: {gen['Poor']:.1f}% vs excellent health: {gen['Excellent']:.1f}%.",
        f"Lowest income band: {inc.iloc[0]:.1f}% vs highest band: {inc.iloc[-1]:.1f}%.",
        f"Obese II+ respondents: {bmi['Obese II+']:.1f}% vs normal BMI: {bmi['Normal']:.1f}%.",
        f"High blood pressure: {bp[1]:.1f}% vs no high blood pressure: {bp[0]:.1f}%.",
        f"High BP + high cholesterol + obese profile ({seg['segment_size']:,} people): "
        f"{seg['segment_prevalence_pct']:.1f}% vs {seg['rest_prevalence_pct']:.1f}% for everyone else.",
    ]


def write_summary(path: Path, log: dict, findings: list[str], desc: pd.DataFrame,
                  normality: pd.DataFrame, cont: pd.DataFrame, chi: pd.DataFrame,
                  model: dict, sens: pd.DataFrame, kruskal: dict) -> None:
    """Collect every result into one text file that can be pasted into the report."""
    r = log["raw"]
    c = log["clean"]
    b = log["bmi"]
    m = model["metrics"]
    top_or = model["or_table"].sort_values("OR", ascending=False)
    lines = [
        "ANALYSIS OF FACTORS ASSOCIATED WITH DIABETES STATUS - RESULTS SUMMARY",
        "=" * 70,
        "\n1. DATA CLEANING LOG",
        f"Raw data: {r['rows']:,} rows x {r['columns']} columns, {r['missing_cells']} missing cells, "
        f"{r['duplicate_rows']:,} duplicate rows.",
        f"Missing cells handled: {log['missing_cells_found']}",
        f"Duplicate rows removed: {log['duplicates_removed']:,}",
        f"Out-of-range rows removed: {log['invalid_rows_removed']}",
        f"BMI IQR fences: [{b['bmi_iqr_low']}, {b['bmi_iqr_high']}] -> {b['bmi_outliers_flagged']:,} outliers flagged "
        f"(kept; winsorised to [{b['bmi_winsor_p1']}, {b['bmi_winsor_p99']}] for modelling).",
        f"Clean data: {c['rows']:,} rows x {c['columns']} columns, memory {r['memory_mb']} MB -> {c['memory_mb']} MB.",
        "\n2. KEY FINDINGS (descriptive)",
        *[f" - {f}" for f in findings],
        "\n3. DESCRIPTIVE STATISTICS",
        desc.to_string(),
        "\n4. NORMALITY OF BMI (Shapiro-Wilk, 5000-row sample per group)",
        normality.round(4).to_string(index=False),
        "Interpretation: p < 0.05 and positive skew -> BMI is not normal -> Mann-Whitney U is the primary test.",
        "\n5. DIABETIC vs NON-DIABETIC: CONTINUOUS VARIABLES",
        cont.round(4).T.to_string(),
        f"\nKruskal-Wallis (BMI across general-health levels): H = {kruskal['H']:.1f}, p = {kruskal['p_value']:.3g}",
        "\n6. CHI-SQUARE TESTS (sorted by effect size, Bonferroni corrected)",
        chi.round(4).to_string(),
        "Interpretation: with a large n most p-values are tiny, so judge importance by Cramer's V.",
        "\n7. LOGISTIC REGRESSION (adjusted odds ratios)",
        model["or_table"].sort_values("OR", ascending=False).round(4).to_string(),
        f"\nStrongest risk-associated factors : {', '.join(top_or.index[:4])}",
        f"Strongest protective-associated   : {', '.join(top_or.index[::-1][:4])}",
        "CAUTION: CholCheck (cholesterol checked in last 5 years) is probably a DETECTION effect - people with diabetes "
        "see doctors more - not a biological risk factor. HvyAlcoholConsump < 1 is a well-known survey artefact "
        "(sick-quitter effect), NOT proof that alcohol protects. Odds ratios near 1 with CI crossing 1 are not significant.",
        f"Pseudo R2 (McFadden) = {m['pseudo_r2_mcfadden']:.3f} | test AUC = {m['test_auc']:.3f} | "
        f"sensitivity = {m['sensitivity']:.3f} | specificity = {m['specificity']:.3f} "
        f"(threshold = {m['threshold']:.3f})",
        "\nMulticollinearity (VIF):",
        model["vif"].round(2).to_string(index=False),
        "Rule of thumb: VIF < 5 -> acceptable.",
        "\n8. SENSITIVITY ANALYSIS: duplicates kept vs removed",
        sens.round(4).to_string(),
        "\n9. LIMITATIONS",
        " - Cross-sectional survey: results show ASSOCIATION, not causation.",
        " - Self-reported data (recall / social-desirability bias).",
        " - Target merges pre-diabetes and diabetes.",
        " - The sample is artificially balanced 50/50: group percentages are NOT population prevalence;",
        "   use odds ratios / relative differences, and note predictive accuracy is on a balanced sample.",
        " - Statistical significance is inflated by the large sample size -> effect sizes reported.",
        " - Forecasting is not applicable: the data have no time dimension.",
    ]
    path.write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # 1-2. load, inspect, clean --------------------------------------------
    banner("STEP 1-2: Load, inspect and clean the data")
    df, log = dm.clean_pipeline(args.input, out / "diabetes_cleaned.csv")
    for key in ("raw", "missing_cells_found", "duplicates_removed", "invalid_rows_removed", "bmi", "clean"):
        print(f"{key:>22}: {log[key]}")
    print(f"\nCleaned CSV saved to: {log['saved_to']}")

    if args.explore:
        explore_variable(df, args.explore)
        return

    # 3. statistics ----------------------------------------------------------
    banner("STEP 3: Statistical analysis")
    desc = sa.descriptive_stats(df)
    normality = sa.normality_check(df)
    cont = sa.compare_continuous(df)
    chi = sa.chi_square_tests(df)
    spear = sa.spearman_ordinal(df)
    kruskal = sa.kruskal_bmi_by_health(df)
    model = sa.fit_logistic(df)

    # sensitivity: re-run the model WITHOUT dropping duplicates
    df_dup, _ = dm.clean_pipeline(args.input, out_path=None, drop_duplicates=False)
    model_dup = sa.fit_logistic(df_dup)
    sens = sa.compare_or_tables(model_dup["or_table"], model["or_table"])

    print(chi[["chi2", "p_bonferroni", "cramers_v", "effect_size", "crude_odds_ratio"]].round(4).to_string())
    print(f"\nTest AUC = {model['metrics']['test_auc']:.3f}")

    # 4. visualisations --------------------------------------------------------
    banner("STEP 4: Visualisations")
    paths = viz.generate_all_plots(df, model["or_table"], model["roc"],
                                   model["metrics"]["test_auc"], out / "figures", show=args.show)
    for p in paths:
        print("saved", p)

    # 5. outputs -----------------------------------------------------------------
    banner("STEP 5: Writing tables and summary")
    tables = {
        "descriptive_stats": desc, "continuous_tests": cont, "chi_square_tests": chi,
        "spearman_ordinal": spear, "odds_ratios": model["or_table"], "vif": model["vif"],
        "duplicate_sensitivity": sens,
        "prevalence_by_age": dm.prevalence_by(df, "Age_Group"),
        "prevalence_by_income": dm.prevalence_by(df, "Income_Label"),
        "prevalence_by_bmi_category": dm.prevalence_by(df, "BMI_Category"),
    }
    for name, tbl in tables.items():
        tbl.to_csv(out / "tables" / f"{name}.csv")
    (out / "tables" / "logit_full_summary.txt").write_text(model["summary"], encoding="utf-8")

    findings = key_findings(df)
    write_summary(out / "summary.txt", log, findings, desc, normality, cont, chi, model, sens, kruskal)
    print(f"Summary written to {out / 'summary.txt'}\n")
    for f in findings:
        print(" -", f)
    print("\nDone.")


if __name__ == "__main__":
    main()
