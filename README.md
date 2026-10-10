# Analysis of Factors Associated with Diabetes Status
Python Programming Laboratory (316U50L301) - Capstone Mini Project | EXCP - Healthcare
Dataset: Diabetes Health Indicators (CDC BRFSS 2015, Kaggle - Alex Teboul), balanced 50/50 version
(`data/diabetesData.csv`, 70,692 rows x 22 columns).

## How to run
1. `pip install -r requirements.txt`
2. Keep `diabetesData.csv` inside the `data/` folder (already included).
3. `python main.py`  ->  creates everything inside `outputs/`

Other commands:
- `python main.py --show` : also pop up the plots
- `python main.py --explore Income` : quick prevalence table + test for ANY column (handy in the viva)
- `python main.py --input other.csv` : run on a different CSV

## Files
| File | Role | Rubric criteria |
|---|---|---|
| main.py | runs the pipeline, writes the summary | 2 |
| database_manipulation.py | load, validate, clean, feature engineering, groupby summaries | 2, 3 |
| visualization.py | 11 Matplotlib/Seaborn charts | 4 |
| statistics_analysis.py | tests, effect sizes, logistic regression, VIF, AUC | 5 |
| outputs/diabetes_cleaned.csv | cleaned + feature-engineered dataset | 3 |
| outputs/figures/ | 11 charts for the report | 4 |
| outputs/tables/ | all result tables as CSV | 5, 6 |
| outputs/summary.txt | every number used in the report | 6 |

## Important: this dataset is balanced 50/50 by design
Percentages like "66.9% of people with high BP have diabetes" are shares INSIDE this sample,
not population prevalence. Compare groups to the 50% baseline, and rely on odds ratios.

## Viva preparation
- Why logistic regression, not ARIMA? Target is binary; data is cross-sectional (no time axis).
- Why Mann-Whitney not t-test? BMI is right-skewed (Shapiro-Wilk, skew 1.7); Welch t shown as robustness check.
- Why Cramer's V when p < 0.001? With 69,000 rows nearly everything is "significant"; effect size shows importance.
- Why Bonferroni? 18 chi-square tests at once inflate false positives.
- Why keep BMI outliers? Extreme BMI is medically possible; flagged (IQR) and winsorised (1st-99th pct) instead.
- Why remove duplicates? 1,635 exact repeats; the model was re-run with duplicates kept - odds ratios barely change.
- What does OR = 2.03 for HighBP mean? Odds of diabetes about double, holding other variables fixed.
- Why is CholCheck OR so high (3.9)? Detection effect: diabetics see doctors more. Not a biological risk factor.
- Why does heavy alcohol look protective (OR 0.47)? Survey artefact (sick-quitter effect) - not causal; excluded from lifestyle score.
- Why AUC? It judges ranking quality at every threshold. Test AUC = 0.817 on a balanced hold-out set.
- Can we say HighBP causes diabetes? No - cross-sectional, self-reported: association only.
