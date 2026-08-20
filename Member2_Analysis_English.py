"""
Member 2: LendingClub Exploratory Data Analysis and Statistical Analysis

This program follows the course requirements:
1. Clean real loan data and handle noise/outliers
2. Explore default risk versus FICO, DTI, income, interest rate, grade, and state GDP
3. Validate visual findings with correlations, tests, and logistic regression
4. Use robust standard errors, staged models, and VIF checks
5. Export presentation-ready PNG charts, CSV tables, and an English report

Run:
python Member2_Analysis_English.py \
  --input "/Users/sophia/Downloads/LendingClub_GDP_Final.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


REQUIRED_COLUMNS = [
    "fico", "dti", "annual_inc", "int_rate", "grade",
    "state_gdp_2012", "default", "loan_amnt", "term",
    "home_ownership", "revol_util", "delinq_2yrs",
    "inq_last_6mths", "pub_rec", "total_acc",
]

COLORS = {
    "safe": "#4C78A8",
    "risk": "#E45756",
    "orange": "#F58518",
    "green": "#54A24B",
    "purple": "#B279A2",
    "gray": "#6B7280",
    "light": "#E8EEF5",
}


def configure_charts() -> None:
    """Configure consistent fonts, colors, and resolution."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Heiti SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
    })


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Member 2: EDA and statistical analysis")
    parser.add_argument("--input", required=True, help="Path to LendingClub_GDP_Final.csv")
    parser.add_argument(
        "--output",
        default="member2_lendingclub_analysis/outputs_English",
        help="Output folder",
    )
    return parser.parse_args()


def read_and_clean(input_path: Path, tables_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read the CSV and perform the course-required data cleaning."""
    raw = pd.read_csv(input_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"The CSV is missing required columns: {missing_columns}")

    df = raw.copy()
    raw_rows = len(df)

    # Convert percentage strings: "14.65%" -> 14.65.
    for column in ["int_rate", "revol_util"]:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )

    # Convert loan-term strings: "36 months" -> 36.
    df["term"] = pd.to_numeric(
        df["term"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    numeric_columns = [
        "fico", "dti", "annual_inc", "int_rate", "state_gdp_2012",
        "default", "loan_amnt", "term", "revol_util", "delinq_2yrs",
        "inq_last_6mths", "pub_rec", "total_acc",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_table = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean(),
    }).sort_values("missing_rate", ascending=False)
    missing_table.to_csv(tables_dir / "01_missing_values.csv", encoding="utf-8-sig")

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    essential = [
        "default", "fico", "dti", "annual_inc", "int_rate",
        "grade", "state_gdp_2012",
    ]
    before_drop = len(df)
    df = df.dropna(subset=essential).copy()
    missing_rows_removed = before_drop - len(df)
    df = df[df["default"].isin([0, 1])].copy()
    df = df[(df["annual_inc"] > 0) & (df["state_gdp_2012"] > 0)].copy()
    df["default"] = df["default"].astype(int)

    # Winsorize rather than delete extreme loans to reduce distortion.
    winsor_limits = {}
    for column in ["dti", "annual_inc", "loan_amnt", "state_gdp_2012"]:
        low, high = df[column].quantile([0.01, 0.99])
        winsor_limits[column] = {"p01": float(low), "p99": float(high)}
        df[f"{column}_w"] = df[column].clip(low, high)

    # Log-transform right-skewed income and GDP for regression.
    df["log_income"] = np.log1p(df["annual_inc_w"])
    df["log_state_gdp"] = np.log(df["state_gdp_2012_w"])

    # Rescale units so odds ratios are easier to interpret.
    df["fico_10"] = df["fico"] / 10
    df["dti_5"] = df["dti_w"] / 5
    df["loan_amnt_1000"] = df["loan_amnt_w"] / 1000

    cleaning_summary = {
        "raw_rows": raw_rows,
        "final_rows": len(df),
        "duplicates_removed": duplicate_count,
        "missing_rows_removed": missing_rows_removed,
        "defaults": int(df["default"].sum()),
        "default_rate": float(df["default"].mean()),
        "winsor_limits": winsor_limits,
    }

    df.to_csv(tables_dir / "02_cleaned_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "raw_rows": raw_rows,
        "final_rows": len(df),
        "duplicates_removed": duplicate_count,
        "missing_rows_removed": missing_rows_removed,
        "defaults": int(df["default"].sum()),
        "default_rate": df["default"].mean(),
    }]).to_csv(tables_dir / "02_cleaning_summary.csv", index=False, encoding="utf-8-sig")

    return df, cleaning_summary


def describe_data(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    columns = [
        "default", "fico", "dti", "annual_inc", "int_rate",
        "state_gdp_2012", "loan_amnt", "revol_util",
        "delinq_2yrs", "inq_last_6mths", "pub_rec", "total_acc",
    ]
    result = df[columns].describe(percentiles=[0.25, 0.50, 0.75]).T
    result.to_csv(tables_dir / "03_descriptive_statistics.csv", encoding="utf-8-sig")
    return result


def add_rate_labels(ax, values, y_offset=0.006) -> None:
    for index, value in enumerate(values):
        ax.text(index, value + y_offset, f"{value:.1%}", ha="center", fontsize=10, fontweight="bold")


def binomial_ci(rate: pd.Series, size: pd.Series) -> pd.Series:
    return 1.96 * np.sqrt(rate * (1 - rate) / size)


def quantile_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    working = df[[column, "default"]].dropna().copy()
    working["group"] = pd.qcut(working[column], q=5, duplicates="drop")
    result = (
        working.groupby("group", observed=True)
        .agg(
            group_mean=(column, "mean"),
            group_median=(column, "median"),
            default_rate=("default", "mean"),
            sample_size=("default", "size"),
        )
        .reset_index()
    )
    result["ci_95"] = binomial_ci(result["default_rate"], result["sample_size"])
    result["group_label"] = [f"Group {i}" for i in range(1, len(result) + 1)]
    result["group"] = result["group"].astype(str)
    return result


def save_overview_chart(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df["default"].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Non-default (0)", "Default (1)"],
        counts.values,
        color=[COLORS["safe"], COLORS["risk"]],
        width=0.55,
    )
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, count + 120, f"{count:,} loans", ha="center", fontweight="bold")
    ax.text(
        0.98, 0.93,
        f"Overall default rate = {df['default'].mean():.1%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=14,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLORS["light"], "edgecolor": "none"},
    )
    ax.set_title("Figure 1. Distribution of Loan Defaults")
    ax.set_ylabel("Number of loans")
    ax.set_ylim(0, counts.max() * 1.16)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "01_default_distribution.png", bbox_inches="tight")
    plt.close(fig)


def save_fico_chart(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    summary = quantile_summary(df, "fico")
    summary.to_csv(tables_dir / "04_fico_quintiles.csv", index=False, encoding="utf-8-sig")

    univariate = smf.logit("default ~ fico", data=df).fit(disp=False)
    fico_grid = pd.DataFrame({"fico": np.linspace(df["fico"].min(), df["fico"].max(), 250)})
    fico_grid["predicted_default"] = univariate.predict(fico_grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sns.boxplot(
        data=df, x="default", y="fico", hue="default",
        palette={0: COLORS["safe"], 1: COLORS["risk"]},
        legend=False, ax=axes[0], showfliers=False,
    )
    axes[0].set_xticks([0, 1], ["Non-default", "Default"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("FICO score")
    axes[0].set_title("FICO distribution by loan outcome")

    axes[1].errorbar(
        summary["group_label"], summary["default_rate"],
        yerr=summary["ci_95"], marker="o", capsize=4,
        linewidth=2.5, color=COLORS["safe"],
    )
    for i, row in summary.iterrows():
        axes[1].text(i, row["default_rate"] + row["ci_95"] + 0.008, f"{row['default_rate']:.1%}", ha="center")
    axes[1].set_xlabel("FICO quintiles: low to high")
    axes[1].set_ylabel("Default rate")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Higher FICO, lower default rate")

    axes[2].plot(fico_grid["fico"], fico_grid["predicted_default"], color=COLORS["purple"], linewidth=3)
    axes[2].fill_between(fico_grid["fico"], 0, fico_grid["predicted_default"], color=COLORS["purple"], alpha=0.12)
    axes[2].set_xlabel("FICO score")
    axes[2].set_ylabel("Predicted default probability")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_title("Univariate logistic regression curve")

    fig.suptitle("Figure 2. FICO and Default Risk", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_fico_and_default.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def save_continuous_chart(
    df: pd.DataFrame,
    column: str,
    display_column: str,
    y_label: str,
    title: str,
    file_stem: str,
    tables_dir: Path,
    figures_dir: Path,
    use_log_axis: bool = False,
) -> pd.DataFrame:
    summary = quantile_summary(df, column)
    summary.to_csv(tables_dir / f"04_{file_stem}_quintiles.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.boxplot(
        data=df, x="default", y=display_column, hue="default",
        palette={0: COLORS["safe"], 1: COLORS["risk"]},
        legend=False, ax=axes[0], showfliers=False,
    )
    axes[0].set_xticks([0, 1], ["Non-default", "Default"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel(y_label)
    axes[0].set_title(f"{title} distribution by loan outcome")
    if use_log_axis:
        axes[0].set_yscale("log")

    axes[1].errorbar(
        summary["group_label"], summary["default_rate"],
        yerr=summary["ci_95"], marker="o", capsize=4,
        linewidth=2.5, color=COLORS["orange"],
    )
    for i, row in summary.iterrows():
        axes[1].text(i, row["default_rate"] + row["ci_95"] + 0.007, f"{row['default_rate']:.1%}", ha="center")
    axes[1].set_xlabel(f"{title} quintiles: low to high")
    axes[1].set_ylabel("Default rate")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Default rate with 95% confidence interval")
    lower_limit = max(0, float((summary["default_rate"] - summary["ci_95"]).min()) - 0.01)
    upper_limit = float((summary["default_rate"] + summary["ci_95"]).max()) + 0.025
    axes[1].set_ylim(lower_limit, upper_limit)

    fig.suptitle(f"{title} and Default Risk", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{file_stem}.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def save_grade_chart(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    grade = (
        df.groupby("grade", observed=True)
        .agg(default_rate=("default", "mean"), sample_size=("default", "size"))
        .reindex(list("ABCDEFG"))
        .dropna()
        .reset_index()
    )
    grade["ci_95"] = binomial_ci(grade["default_rate"], grade["sample_size"])
    grade.to_csv(tables_dir / "04_grade_default_rates.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("RdYlGn_r", len(grade))
    bars = ax.bar(
        grade["grade"], grade["default_rate"],
        yerr=grade["ci_95"], capsize=4, color=colors,
    )
    for bar, rate, n in zip(bars, grade["default_rate"], grade["sample_size"]):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            rate + 0.025,
            f"{rate:.1%}\n(n={n:,})",
            ha="center", fontsize=9,
        )
    ax.set_title("Figure 6. Lower Loan Grade, Higher Default Rate")
    ax.set_xlabel("Loan grade (A best, G worst)")
    ax.set_ylabel("Default rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, min(0.70, (grade["default_rate"] + grade["ci_95"]).max() + 0.13))
    fig.tight_layout()
    fig.savefig(figures_dir / "06_grade_and_default.png", bbox_inches="tight")
    plt.close(fig)
    return grade


def correlation_and_tests(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation_columns = [
        "default", "fico", "dti_w", "log_income", "int_rate",
        "log_state_gdp", "loan_amnt_1000", "revol_util",
        "delinq_2yrs", "inq_last_6mths", "pub_rec", "total_acc",
    ]
    corr = df[correlation_columns].corr(method="spearman")
    corr.to_csv(tables_dir / "07_spearman_correlation_matrix.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Figure 8. Spearman Correlation Heatmap\nRed = positive correlation; blue = negative correlation")
    fig.tight_layout()
    fig.savefig(figures_dir / "08_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    tests = []
    for column in ["fico", "dti_w", "log_income", "int_rate", "log_state_gdp"]:
        group0 = df.loc[df["default"] == 0, column].dropna()
        group1 = df.loc[df["default"] == 1, column].dropna()
        u, p = mannwhitneyu(group0, group1, alternative="two-sided")
        rho, rho_p = spearmanr(df[column], df["default"], nan_policy="omit")
        tests.append({
            "variable": column,
            "test": "Mann-Whitney U",
            "non_default_median": group0.median(),
            "default_median": group1.median(),
            "statistic": u,
            "p_value": p,
            "spearman_rho": rho,
            "spearman_p_value": rho_p,
        })

    contingency = pd.crosstab(df["grade"], df["default"])
    chi2, chi_p, dof, _ = chi2_contingency(contingency)
    tests.append({
        "variable": "grade",
        "test": "Chi-square",
        "statistic": chi2,
        "p_value": chi_p,
        "degrees_of_freedom": dof,
    })
    tests_df = pd.DataFrame(tests)
    tests_df["significant_5pct"] = tests_df["p_value"] < 0.05
    tests_df.to_csv(tables_dir / "07_statistical_tests.csv", index=False, encoding="utf-8-sig")
    return corr, tests_df


def model_result(model) -> pd.DataFrame:
    ci = model.conf_int()
    result = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "robust_std_error": model.bse.values,
        "odds_ratio": np.exp(model.params.values),
        "ci_lower": np.exp(ci[0].values),
        "ci_upper": np.exp(ci[1].values),
        "p_value": model.pvalues.values,
    })
    result["significant_5pct"] = result["p_value"] < 0.05
    return result


def fit_statistical_models(df: pd.DataFrame, tables_dir: Path) -> dict:
    # Model 1: borrower financial condition and macro environment.
    formula1 = "default ~ fico_10 + dti_5 + log_income + log_state_gdp"

    # Model 2: add platform risk pricing and loan terms.
    formula2 = (
        "default ~ fico_10 + dti_5 + log_income + log_state_gdp + int_rate "
        "+ C(grade, Treatment(reference='A')) + loan_amnt_1000 "
        "+ C(term, Treatment(reference=36)) "
        "+ C(home_ownership, Treatment(reference='MORTGAGE'))"
    )

    model1 = smf.logit(formula1, data=df).fit(disp=False, maxiter=300, cov_type="HC3")
    model2 = smf.logit(formula2, data=df).fit(disp=False, maxiter=300, cov_type="HC3")

    for name, model in [("model1_borrower_macro", model1), ("model2_full", model2)]:
        model_result(model).to_csv(tables_dir / f"08_{name}_odds_ratios.csv", index=False, encoding="utf-8-sig")
        (tables_dir / f"08_{name}_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")

    # Standardized model: compare one-standard-deviation changes.
    standard_columns = ["fico", "dti_w", "log_income", "int_rate", "log_state_gdp", "loan_amnt_1000"]
    standardized = df.copy()
    for column in standard_columns:
        standardized[f"{column}_z"] = (standardized[column] - standardized[column].mean()) / standardized[column].std()

    formula_z = (
        "default ~ fico_z + dti_w_z + log_income_z + int_rate_z + log_state_gdp_z "
        "+ loan_amnt_1000_z + C(grade, Treatment(reference='A')) "
        "+ C(term, Treatment(reference=36)) "
        "+ C(home_ownership, Treatment(reference='MORTGAGE'))"
    )
    model_z = smf.logit(formula_z, data=standardized).fit(disp=False, maxiter=300, cov_type="HC3")
    model_result(model_z).to_csv(tables_dir / "08_standardized_model_odds_ratios.csv", index=False, encoding="utf-8-sig")

    # VIF check for severe multicollinearity among continuous variables.
    vif_columns = ["fico_10", "dti_5", "log_income", "log_state_gdp", "int_rate", "loan_amnt_1000"]
    vif_x = sm.add_constant(df[vif_columns].dropna())
    vif = pd.DataFrame({
        "variable": vif_x.columns,
        "vif": [variance_inflation_factor(vif_x.values, i) for i in range(vif_x.shape[1])],
    })
    vif.to_csv(tables_dir / "08_vif.csv", index=False, encoding="utf-8-sig")

    return {"model1": model1, "model2": model2, "model_z": model_z, "vif": vif}


def save_model_charts(models: dict, figures_dir: Path) -> None:
    m1 = model_result(models["model1"]).set_index("variable")
    m2 = model_result(models["model2"]).set_index("variable")
    shared = ["fico_10", "dti_5", "log_income", "log_state_gdp"]
    labels = ["FICO (per 10 points)", "DTI (per 5 points)", "Log income", "Log state GDP"]
    y = np.arange(len(shared))

    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, table, label, color in [
        (-0.10, m1, "Model 1: borrower + GDP", COLORS["safe"]),
        (0.10, m2, "Model 2: + rate, grade, and loan terms", COLORS["risk"]),
    ]:
        values = table.loc[shared, "odds_ratio"]
        lower = values - table.loc[shared, "ci_lower"]
        upper = table.loc[shared, "ci_upper"] - values
        ax.errorbar(values, y + offset, xerr=[lower, upper], fmt="o", capsize=4, label=label, color=color)
    ax.axvline(1, color=COLORS["gray"], linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Odds ratio (point estimate with 95% confidence interval)")
    ax.set_title("Figure 9. FICO's Independent Effect Weakens after Adding Rate and Grade")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "09_model_comparison_forest.png", bbox_inches="tight")
    plt.close(fig)

    mz = model_result(models["model_z"]).set_index("variable")
    variables = ["fico_z", "dti_w_z", "log_income_z", "int_rate_z", "log_state_gdp_z", "loan_amnt_1000_z"]
    labels = ["FICO", "DTI", "Income", "Interest rate", "State GDP", "Loan amount"]
    plot = mz.loc[variables].copy()
    plot["label"] = labels
    plot = plot.sort_values("odds_ratio")
    y = np.arange(len(plot))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS["safe"] if value < 1 else COLORS["risk"] for value in plot["odds_ratio"]]
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.errorbar(
            row["odds_ratio"], i,
            xerr=[[row["odds_ratio"] - row["ci_lower"]], [row["ci_upper"] - row["odds_ratio"]]],
            fmt="o", capsize=4, color=colors[i], markersize=8,
        )
        significance = "significant" if row["p_value"] < 0.05 else "not significant"
        ax.text(row["ci_upper"] + 0.02, i, f"OR={row['odds_ratio']:.2f}, {significance}", va="center", fontsize=10)
    ax.axvline(1, color=COLORS["gray"], linestyle="--")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Standardized odds ratio (one-standard-deviation increase)")
    ax.set_title("Figure 10. Direction and Relative Importance in the Full Model")
    fig.tight_layout()
    fig.savefig(figures_dir / "10_standardized_odds_ratios.png", bbox_inches="tight")
    plt.close(fig)


def out_of_sample_logistic(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    features = ["fico", "dti_w", "log_income", "int_rate", "grade", "log_state_gdp"]
    X = df[features].copy()
    y = df["default"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    numeric = ["fico", "dti_w", "log_income", "int_rate", "log_state_gdp"]
    categorical = ["grade"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    metrics = pd.DataFrame([{
        "test_observations": len(y_test),
        "default_rate_test": y_test.mean(),
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
    }])
    metrics.to_csv(tables_dir / "11_test_set_metrics.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[0], name="Logistic regression")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], label="Random guess")
    axes[0].set_title(f"Test-set ROC curve (AUC={metrics.loc[0, 'roc_auc']:.3f})")
    axes[0].legend()

    cm = confusion_matrix(y_test, prediction)
    ConfusionMatrixDisplay(cm, display_labels=["Non-default", "Default"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False,
    )
    axes[1].set_title("Test-set confusion matrix (threshold = 0.50)")
    fig.suptitle("Figure 11. Out-of-Sample Logistic Regression Performance", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "11_roc_and_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return metrics


def create_dashboard(
    df: pd.DataFrame,
    fico: pd.DataFrame,
    dti: pd.DataFrame,
    income: pd.DataFrame,
    interest: pd.DataFrame,
    grade: pd.DataFrame,
    gdp: pd.DataFrame,
    models: dict,
    figures_dir: Path,
) -> None:
    mz = model_result(models["model_z"]).set_index("variable")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.text(0.02, 0.95, "LendingClub Default Risk: Member 2 Key Findings", fontsize=24, fontweight="bold", va="top")
    ax.text(0.02, 0.87, f"Sample: {len(df):,} loans | Defaults: {int(df.default.sum()):,} | Overall default rate: {df.default.mean():.1%}", fontsize=16)

    findings = [
        ("FICO", f"Lowest {fico.default_rate.iloc[0]:.1%} → highest {fico.default_rate.iloc[-1]:.1%}", "Higher score, lower risk"),
        ("DTI", f"Lowest {dti.default_rate.iloc[0]:.1%} → highest {dti.default_rate.iloc[-1]:.1%}", "Higher debt burden, higher risk"),
        ("Income", f"Lowest {income.default_rate.iloc[0]:.1%} → highest {income.default_rate.iloc[-1]:.1%}", "Higher income, lower risk"),
        ("Interest rate", f"Lowest {interest.default_rate.iloc[0]:.1%} → highest {interest.default_rate.iloc[-1]:.1%}", "Platform pricing embeds credit risk"),
        ("Grade", f"Grade A {grade.default_rate.iloc[0]:.1%} → G {grade.default_rate.iloc[-1]:.1%}", "Worse grade, higher risk"),
        ("State GDP", f"Lowest {gdp.default_rate.iloc[0]:.1%} → highest {gdp.default_rate.iloc[-1]:.1%}", "Small, insignificant full-model difference"),
    ]
    y = 0.75
    for name, numbers, insight in findings:
        ax.text(0.03, y, name, fontsize=15, fontweight="bold", color=COLORS["safe"])
        ax.text(0.18, y, numbers, fontsize=14)
        ax.text(0.54, y, insight, fontsize=14)
        y -= 0.085

    ax.text(0.03, 0.20, "Full-model conclusions", fontsize=16, fontweight="bold")
    significant = []
    for variable, label in [("dti_w_z", "DTI"), ("log_income_z", "Income"), ("int_rate_z", "Interest rate"), ("fico_z", "FICO"), ("log_state_gdp_z", "State GDP")]:
        row = mz.loc[variable]
        significant.append(f"{label}: OR={row.odds_ratio:.2f} ({'significant' if row.p_value < .05 else 'not significant'})")
    ax.text(0.03, 0.14, " | ".join(significant), fontsize=13)
    ax.text(0.03, 0.07, "Caution: correlation is not causation. FICO, interest rate, and grade contain overlapping risk information.", fontsize=13, color=COLORS["risk"])
    fig.tight_layout()
    fig.savefig(figures_dir / "00_core_findings_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def create_english_explanation_report(
    df: pd.DataFrame,
    cleaning: dict,
    fico: pd.DataFrame,
    dti: pd.DataFrame,
    income: pd.DataFrame,
    interest: pd.DataFrame,
    grade: pd.DataFrame,
    gdp: pd.DataFrame,
    tests: pd.DataFrame,
    models: dict,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a beginner-friendly English interpretation for every chart."""
    m1 = model_result(models["model1"]).set_index("variable")
    m2 = model_result(models["model2"]).set_index("variable")

    def p_text(value: float) -> str:
        return "p<0.001" if value < 0.001 else f"p={value:.3f}"

    report = f"""# Member 2: Exploratory Data Analysis and Statistical Analysis

## 1. Research question

This section studies how default risk is associated with FICO score, DTI, income, interest rate, loan grade, and state GDP. Following the course workflow, it first cleans and explores the data, then checks the visual patterns using correlations, statistical tests, and logistic regression.

## 2. Data cleaning

- Raw sample: {cleaning['raw_rows']:,} loans.
- Final sample: {cleaning['final_rows']:,} loans.
- Duplicate rows removed: {cleaning['duplicates_removed']:,}.
- Rows removed for missing essential fields: {cleaning['missing_rows_removed']:,}.
- Percentage strings were converted to numbers, and loan terms were converted to 36 or 60.
- DTI, income, loan amount, and state GDP were winsorized at the 1st and 99th percentiles.
- Income and GDP were log-transformed to reduce right skew.
- `default=1` means default; `default=0` means non-default.

![Key findings](figures/00_core_findings_dashboard.png)

## 3. Figure 1: Default distribution

![Default distribution](figures/01_default_distribution.png)

**How it was made:** Count the 0s and 1s in `default`, then divide defaults by total loans.

**What it shows:** There are {len(df):,} loans and {int(df.default.sum()):,} defaults, giving an overall default rate of {df.default.mean():.1%}.

**Interpretation:** The classes are imbalanced, so accuracy alone is insufficient. Recall, F1, and AUC should also be reported.

## 4. Figure 2: FICO and default risk

![FICO](figures/02_fico_and_default.png)

**How it was made:** Compare FICO distributions, calculate default rates across five FICO groups, and fit a one-variable logistic curve.

**What it shows:** Default falls from {fico.default_rate.iloc[0]:.1%} in the lowest FICO group to {fico.default_rate.iloc[-1]:.1%} in the highest group.

**Interpretation:** Higher FICO signals better credit history. In Model 1, a 10-point increase has OR={m1.loc['fico_10','odds_ratio']:.3f} ({p_text(m1.loc['fico_10','p_value'])}); after rate and grade are added, OR={m2.loc['fico_10','odds_ratio']:.3f} ({p_text(m2.loc['fico_10','p_value'])}), showing overlapping risk information.

## 5. Figure 3: DTI and default risk

![DTI](figures/03_dti_and_default.png)

**How it was made:** Compare DTI distributions and calculate quintile default rates with 95% confidence intervals.

**What it shows:** Default rises from {dti.default_rate.iloc[0]:.1%} in the lowest DTI group to {dti.default_rate.iloc[-1]:.1%} in the highest.

**Interpretation:** Higher DTI means heavier existing debt burden. In the full model, a five-point increase has OR={m2.loc['dti_5','odds_ratio']:.3f} ({p_text(m2.loc['dti_5','p_value'])}).

## 6. Figure 4: Income and default risk

![Income](figures/04_income_and_default.png)

**How it was made:** Winsorize income, use a log scale for the box plot, and compare income-quintile default rates.

**What it shows:** Default declines from {income.default_rate.iloc[0]:.1%} in the lowest income group to {income.default_rate.iloc[-1]:.1%} in the highest.

**Interpretation:** Higher income generally improves repayment capacity. Log income has OR={m2.loc['log_income','odds_ratio']:.3f} ({p_text(m2.loc['log_income','p_value'])}) in the full model.

## 7. Figure 5: Interest rate and default risk

![Interest rate](figures/05_interest_rate_and_default.png)

**How it was made:** Convert percentage text to numbers, compare distributions, and calculate rate-quintile default rates.

**What it shows:** Default rises from {interest.default_rate.iloc[0]:.1%} in the lowest-rate group to {interest.default_rate.iloc[-1]:.1%} in the highest.

**Interpretation:** Interest rate is a risk-pricing outcome. A high rate may reflect pre-existing borrower risk rather than cause default. Each one-percentage-point increase has OR={m2.loc['int_rate','odds_ratio']:.3f} ({p_text(m2.loc['int_rate','p_value'])}).

## 8. Figure 6: Loan grade and default risk

![Loan grade](figures/06_grade_and_default.png)

**How it was made:** Calculate the default rate, sample size, and 95% confidence interval for grades A through G.

**What it shows:** Grade A defaults at {grade.default_rate.iloc[0]:.1%}, versus {grade.default_rate.iloc[-1]:.1%} for grade G.

**Interpretation:** LendingClub grade summarizes multiple risks. Grade G has only {int(grade.sample_size.iloc[-1])} observations, so its wide confidence interval requires caution.

## 9. Figure 7: State GDP and default risk

![State GDP](figures/07_gdp_and_default.png)

**How it was made:** Winsorize GDP, log it for regression, and compare default rates across five GDP groups.

**What it shows:** Default changes only from {gdp.default_rate.iloc[0]:.1%} in the lowest group to {gdp.default_rate.iloc[-1]:.1%} in the highest.

**Interpretation:** Log GDP has OR={m1.loc['log_state_gdp','odds_ratio']:.3f} ({p_text(m1.loc['log_state_gdp','p_value'])}) in Model 1 and OR={m2.loc['log_state_gdp','odds_ratio']:.3f} ({p_text(m2.loc['log_state_gdp','p_value'])}) in the full model. The CSV contains no state code, so a state map or state-clustered standard errors are not possible.

## 10. Figure 8: Correlation heatmap

![Correlation](figures/08_correlation_heatmap.png)

**How it was made:** Compute Spearman correlations, ranging from -1 to +1, which are robust to skew and outliers.

**Interpretation:** Interest rate has the clearest positive association with default, while FICO is negative. Correlation does not establish causation.

## 11. Figure 9: Staged logistic regression

![Model comparison](figures/09_model_comparison_forest.png)

**How it was made:** Model 1 uses FICO, DTI, income, and GDP. Model 2 adds rate, grade, loan amount, term, and housing. Dots are odds ratios; lines are 95% confidence intervals.

**Interpretation:** FICO is strong in the simpler model but weakens after platform pricing variables are added. DTI and income are more stable.

## 12. Figure 10: Important factors in the full model

![Standardized odds ratios](figures/10_standardized_odds_ratios.png)

**How it was made:** Standardize continuous variables before refitting, so each odds ratio represents a one-standard-deviation increase.

**Interpretation:** Income is protective; DTI and interest rate increase risk. State GDP is not significant after controls. Importance should consider effect size, confidence intervals, p-values, and stability across models.

## 13. Figure 11: Out-of-sample performance

![ROC and confusion matrix](figures/11_roc_and_confusion_matrix.png)

**How it was made:** Train on 80% of the data and evaluate on an unseen 20% test set.

**What it shows:** AUC={metrics.loc[0,'roc_auc']:.3f}, accuracy={metrics.loc[0,'accuracy']:.1%}, precision={metrics.loc[0,'precision']:.1%}, recall={metrics.loc[0,'recall']:.1%}, and F1={metrics.loc[0,'f1']:.3f}.

**Interpretation:** The model beats random guessing but is not a strong classifier. In credit risk, false negatives are costly, so accuracy alone can be misleading.

## 14. Final conclusions

1. Interest rate and loan grade show the strongest unadjusted risk gradients, but they are risk-pricing variables and should not be interpreted causally.
2. DTI and income are the most stable factors: higher DTI raises risk, while higher income lowers it.
3. FICO is useful alone, but its independent significance falls after adding interest rate and grade because these variables overlap.
4. State GDP has a weak association and no statistically significant independent effect in the full model.
5. These are statistical associations, not causal estimates.
"""
    (output_dir / "Member2_Chart_Interpretation_English.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    configure_charts()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("1/8 Reading and cleaning data...")
    df, cleaning = read_and_clean(input_path, tables_dir)
    describe_data(df, tables_dir)

    print("2/8 Creating the default distribution...")
    save_overview_chart(df, figures_dir)

    print("3/8 Analyzing FICO, DTI, income, interest rate, grade, and GDP...")
    fico = save_fico_chart(df, tables_dir, figures_dir)
    dti = save_continuous_chart(df, "dti_w", "dti_w", "DTI", "DTI", "03_dti_and_default", tables_dir, figures_dir)
    income = save_continuous_chart(
        df, "annual_inc_w", "annual_inc_w", "Annual income (USD, log scale)", "Income",
        "04_income_and_default", tables_dir, figures_dir, use_log_axis=True,
    )
    interest = save_continuous_chart(
        df, "int_rate", "int_rate", "Annual interest rate (%)", "Interest Rate",
        "05_interest_rate_and_default", tables_dir, figures_dir,
    )
    grade = save_grade_chart(df, tables_dir, figures_dir)
    gdp = save_continuous_chart(
        df, "state_gdp_2012_w", "state_gdp_2012_w", "State GDP", "State GDP",
        "07_gdp_and_default", tables_dir, figures_dir,
    )

    print("4/8 Computing correlations and statistical tests...")
    _, tests = correlation_and_tests(df, tables_dir, figures_dir)

    print("5/8 Fitting logistic regressions and checking VIF...")
    models = fit_statistical_models(df, tables_dir)
    save_model_charts(models, figures_dir)

    print("6/8 Evaluating out-of-sample performance...")
    metrics = out_of_sample_logistic(df, tables_dir, figures_dir)

    print("7/8 Creating the key-findings dashboard and chart explanations...")
    create_dashboard(df, fico, dti, income, interest, grade, gdp, models, figures_dir)
    create_english_explanation_report(
        df, cleaning, fico, dti, income, interest, grade, gdp,
        tests, models, metrics, output_dir,
    )

    print("8/8 Complete.")
    print(f"Output folder: {output_dir}")
    print(f"Charts: {figures_dir}")
    print(f"Statistical tables: {tables_dir}")
    print(f"Chart explanations: {output_dir / 'Member2_Chart_Interpretation_English.md'}")


if __name__ == "__main__":
    main()
