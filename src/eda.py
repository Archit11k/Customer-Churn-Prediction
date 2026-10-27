"""
Exploratory Data Analysis: descriptive statistics, churn-rate tables and
figures. Every number written here is computed from the dataset at run time.

Run:  python -m src.eda
Outputs: reports/figures/eda_*.png and reports/eda_summary.json
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless backend; figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config
from src.data_preprocessing import (
    count_addons,
    data_quality_report,
    iqr_outliers,
    load_clean_data,
    load_raw_data,
    tenure_group,
)

sns.set_theme(style="whitegrid", context="notebook")
CHURN_PALETTE = {"No": "#4C72B0", "Yes": "#DD8452"}


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _label_churn(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ChurnLabel"] = out[config.TARGET].map({0: "No", 1: "Yes"})
    return out


# ----------------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, mode, variance, std, skewness etc. for numeric features."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        s = df[col]
        rows.append({
            "feature": col,
            "count": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "mode": s.mode().iloc[0],
            "variance": s.var(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "skewness": s.skew(),
        })
    return pd.DataFrame(rows).set_index("feature")


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customer count and churn rate (%) for each level of a categorical column."""
    g = df.groupby(col, observed=True)[config.TARGET].agg(customers="count", churned="sum")
    g["churn_rate_%"] = 100 * g["churned"] / g["customers"]
    return g.sort_values("churn_rate_%", ascending=False)


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of independence between each categorical feature and Churn."""
    rows = []
    for col in config.CATEGORICAL_FEATURES:
        table = pd.crosstab(df[col], df[config.TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        rows.append({"feature": col, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_group_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median by churn group plus Mann-Whitney U test (no normality assumption)."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        a = df.loc[df[config.TARGET] == 0, col]
        b = df.loc[df[config.TARGET] == 1, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "feature": col,
            "mean_no_churn": a.mean(), "mean_churn": b.mean(),
            "median_no_churn": a.median(), "median_churn": b.median(),
            "mann_whitney_p": p,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_churn_distribution(df):
    d = _label_churn(df)
    counts = d["ChurnLabel"].value_counts().reindex(["No", "Yes"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=d, x="ChurnLabel", order=["No", "Yes"], hue="ChurnLabel",
                  palette=CHURN_PALETTE, legend=False, ax=axes[0])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    axes[0].set(title="Churn count", xlabel="Churn", ylabel="Customers")
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]], startangle=90)
    axes[1].set_title("Churn share")
    _save(fig, "eda_churn_distribution.png")


def plot_numeric_histograms(df):
    d = _label_churn(df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.histplot(data=d, x=col, hue="ChurnLabel", bins=30, kde=True, stat="density",
                     common_norm=False, palette=CHURN_PALETTE, ax=ax)
        ax.set_title(f"{col} distribution by churn")
    _save(fig, "eda_numeric_histograms.png")


def plot_boxplots(df):
    d = _label_churn(df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=d, x="ChurnLabel", y=col, order=["No", "Yes"], hue="ChurnLabel",
                    palette=CHURN_PALETTE, legend=False, ax=ax)
        ax.set(title=f"{col} by churn (IQR box plot)", xlabel="Churn")
    _save(fig, "eda_boxplots.png")


def plot_churn_rate_bar(df, col, fname, rotate=0):
    rate = churn_rate_by(df, col).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rate.index.astype(str), y=rate["churn_rate_%"], color="#DD8452", ax=ax)
    for i, v in enumerate(rate["churn_rate_%"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    ax.axhline(100 * df[config.TARGET].mean(), ls="--", c="grey", label="overall churn rate")
    ax.set(title=f"Churn rate by {col}", xlabel=col, ylabel="Churn rate (%)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center")
    _save(fig, fname)


def plot_service_grid(df):
    cols = ["InternetService", "OnlineSecurity", "TechSupport", "OnlineBackup",
            "DeviceProtection", "StreamingTV", "PaperlessBilling", "SeniorCitizen", "Partner",
            "Dependents", "PhoneService", "gender"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, col in zip(axes.ravel(), cols):
        rate = churn_rate_by(df, col).sort_index()
        sns.barplot(x=rate.index.astype(str), y=rate["churn_rate_%"], color="#DD8452", ax=ax)
        ax.axhline(100 * df[config.TARGET].mean(), ls="--", c="grey")
        ax.set(title=col, xlabel="", ylabel="Churn rate (%)")
    fig.suptitle("Churn rate by customer / service attribute (dashed = overall rate)", y=1.01)
    fig.tight_layout()
    _save(fig, "eda_service_features.png")


def plot_scatter(df):
    d = _label_churn(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=d, x="tenure", y="MonthlyCharges", hue="ChurnLabel", alpha=0.45,
                    s=14, palette=CHURN_PALETTE, ax=ax)
    ax.set_title("Tenure vs monthly charges, coloured by churn")
    _save(fig, "eda_scatter_tenure_monthly.png")


def plot_correlation(df):
    d = df.copy()
    d["Contract_months"] = d["Contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24})
    d["NumAddOns"] = count_addons(d)
    cols = config.NUMERIC_FEATURES + ["Contract_months", "NumAddOns", config.TARGET]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Pearson correlation (numeric + ordinal features)")
    _save(fig, "eda_correlation_heatmap.png")
    return corr


def plot_tenure_contract(df):
    d = df.copy()
    d["TenureGroup"] = tenure_group(d["tenure"])
    pivot = d.pivot_table(index="TenureGroup", columns="Contract", values=config.TARGET,
                          aggfunc="mean", observed=True) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Oranges", ax=ax,
                cbar_kws={"label": "Churn rate (%)"})
    ax.set_title("Churn rate (%) by tenure group and contract")
    _save(fig, "eda_tenure_contract_heatmap.png")
    return pivot


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_eda() -> dict:
    raw = load_raw_data()
    df = load_clean_data()
    df["TenureGroup"] = tenure_group(df["tenure"])
    df["MonthlyChargeBand"] = pd.qcut(df["MonthlyCharges"], 4,
                                      labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    plot_churn_distribution(df)
    plot_numeric_histograms(df)
    plot_boxplots(df)
    plot_churn_rate_bar(df, "Contract", "eda_churn_by_contract.png")
    plot_churn_rate_bar(df, "TenureGroup", "eda_churn_by_tenure_group.png")
    plot_churn_rate_bar(df, "MonthlyChargeBand", "eda_churn_by_monthly_band.png")
    plot_churn_rate_bar(df, "PaymentMethod", "eda_churn_by_payment.png", rotate=20)
    plot_service_grid(df)
    plot_scatter(df)
    corr = plot_correlation(df)
    pivot = plot_tenure_contract(df)

    summary = {
        "data_quality_raw": data_quality_report(raw),
        "n_customers": int(len(df)),
        "n_churned": int(df[config.TARGET].sum()),
        "churn_rate": float(df[config.TARGET].mean()),
        "descriptive_stats": descriptive_stats(df).round(4).to_dict(orient="index"),
        "iqr_outliers": iqr_outliers(df).round(4).to_dict(orient="records"),
        "numeric_by_churn": numeric_group_tests(df).to_dict(orient="records"),
        "chi_square": chi_square_tests(df).to_dict(orient="records"),
        "churn_rate_by": {
            col: churn_rate_by(df, col).round(4).reset_index().astype({col: str}).to_dict(orient="records")
            for col in config.CATEGORICAL_FEATURES + ["TenureGroup", "MonthlyChargeBand"]
        },
        "monthly_charge_band_edges": [float(x) for x in
                                      pd.qcut(df["MonthlyCharges"], 4, retbins=True)[1]],
        "tenure_contract_churn_pct": pivot.round(2).to_dict(),
        "correlation_with_churn": corr[config.TARGET].drop(config.TARGET).round(4).to_dict(),
        "corr_totalcharges_vs_tenure_x_monthly": float(
            np.corrcoef(df["TotalCharges"], df["tenure"] * df["MonthlyCharges"])[0, 1]),
    }
    with open(config.EDA_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    s = run_eda()
    print(f"Customers: {s['n_customers']}, churn rate: {s['churn_rate']:.2%}")
    print(f"Figures saved to {config.FIGURES_DIR}")
