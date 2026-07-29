"""Run exploratory data analysis and save reusable project visuals."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_data.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
SUMMARY_PATH = PROJECT_ROOT / "data" / "eda_summary.csv"

FEATURE_LABELS = {
    "annual_income": "Annual income",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "credit_utilization": "Credit utilization",
    "late_payments": "Late payments",
    "credit_history_years": "Credit history (years)",
    "loan_amount": "Loan amount",
    "interest_rate": "Interest rate",
    "employment_years": "Employment (years)",
    "default": "Default",
}

NAVY = "#17324D"
BLUE = "#3B82B8"
ORANGE = "#E07A3F"
LIGHT = "#E8EEF3"


def save_figure(filename: str) -> None:
    """Apply consistent spacing and save a high-resolution figure."""
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_class_balance(data: pd.DataFrame) -> None:
    counts = data["default"].value_counts().sort_index()
    labels = ["No default", "Default"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts.values, color=[BLUE, ORANGE], width=0.6)
    plt.title("Loan Outcome Distribution", weight="bold", pad=14)
    plt.ylabel("Number of applicants")
    plt.grid(axis="y", alpha=0.2)

    for bar, count in zip(bars, counts.values):
        percentage = count / len(data)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(data) * 0.012,
            f"{count:,}\n({percentage:.1%})",
            ha="center",
            va="bottom",
            weight="bold",
        )
    plt.ylim(0, counts.max() * 1.15)
    save_figure("class_balance.png")


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    correlations = data.corr(numeric_only=True).rename(
        index=FEATURE_LABELS, columns=FEATURE_LABELS
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Feature Correlation Matrix", weight="bold", pad=14)
    save_figure("correlation_heatmap.png")


def plot_default_rate_by_late_payments(data: pd.DataFrame) -> None:
    grouped = (
        data.assign(late_payment_group=data["late_payments"].clip(upper=4))
        .groupby("late_payment_group", as_index=False)
        .agg(default_rate=("default", "mean"), applicants=("default", "size"))
    )
    grouped["label"] = grouped["late_payment_group"].astype(str)
    grouped.loc[grouped["late_payment_group"] == 4, "label"] = "4+"

    plt.figure(figsize=(8, 5))
    bars = plt.bar(grouped["label"], grouped["default_rate"], color=ORANGE)
    plt.title("Default Rate Rises with Late Payments", weight="bold", pad=14)
    plt.xlabel("Late payments")
    plt.ylabel("Default rate")
    plt.gca().yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    plt.grid(axis="y", alpha=0.2)

    for bar, rate, applicants in zip(
        bars, grouped["default_rate"], grouped["applicants"]
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{rate:.1%}\nn={applicants:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylim(0, min(1, grouped["default_rate"].max() + 0.14))
    save_figure("default_rate_by_late_payments.png")


def plot_risk_profile(data: pd.DataFrame) -> None:
    features = [
        "late_payments",
        "credit_utilization",
        "debt_to_income_ratio",
        "interest_rate",
        "credit_history_years",
        "annual_income",
        "employment_years",
        "loan_amount",
    ]

    standardized = data[features].apply(
        lambda column: (column - column.mean()) / column.std()
    )
    standardized["default"] = data["default"]
    profile = standardized.groupby("default")[features].mean().T
    profile = profile.rename(index=FEATURE_LABELS, columns={0: "No default", 1: "Default"})

    profile.plot(
        kind="barh",
        figsize=(9, 6),
        color=[BLUE, ORANGE],
        width=0.72,
    )
    plt.axvline(0, color=NAVY, linewidth=0.8)
    plt.title("Average Applicant Risk Profile", weight="bold", pad=14)
    plt.xlabel("Standard deviations from the overall mean")
    plt.ylabel("")
    plt.legend(title="Outcome")
    plt.grid(axis="x", alpha=0.2)
    save_figure("applicant_risk_profile.png")


def save_summary(data: pd.DataFrame) -> None:
    summary = data.describe().T
    summary["missing"] = data.isna().sum()
    summary["correlation_with_default"] = data.corr(numeric_only=True)["default"]
    summary.to_csv(SUMMARY_PATH, index_label="feature")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run src/generate_data.py first."
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    data = pd.read_csv(DATA_PATH)

    plot_class_balance(data)
    plot_correlation_heatmap(data)
    plot_default_rate_by_late_payments(data)
    plot_risk_profile(data)
    save_summary(data)

    print(f"Analyzed {len(data):,} applicants")
    print(f"Default rate: {data['default'].mean():.2%}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved 4 charts to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
