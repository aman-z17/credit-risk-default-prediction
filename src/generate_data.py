"""Generate a reproducible synthetic credit-risk dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "credit_risk_data.csv"


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert log-odds into probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_intercept(risk_score: np.ndarray, target_rate: float) -> float:
    """Find an intercept that gives the requested average default probability."""
    low, high = -15.0, 15.0
    for _ in range(100):
        midpoint = (low + high) / 2
        if sigmoid(risk_score + midpoint).mean() < target_rate:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def generate_credit_data(
    n_applicants: int = 10_000,
    seed: int = 42,
    target_default_rate: float = 0.18,
) -> pd.DataFrame:
    """Create applicant features and sample a binary loan-default outcome."""
    if n_applicants < 100:
        raise ValueError("n_applicants must be at least 100.")
    if not 0 < target_default_rate < 1:
        raise ValueError("target_default_rate must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    # A hidden variable creates realistic correlation among observed risk factors.
    financial_strain = rng.normal(0, 1, n_applicants)

    annual_income = np.exp(
        rng.normal(np.log(68_000) - 0.16 * financial_strain, 0.48)
    )
    annual_income = np.clip(annual_income, 18_000, 250_000)

    debt_to_income_ratio = (
        0.08
        + 0.48 * rng.beta(2.2, 3.0, n_applicants)
        + 0.055 * financial_strain
    )
    debt_to_income_ratio = np.clip(debt_to_income_ratio, 0.03, 0.75)

    credit_utilization = (
        rng.beta(2.0, 2.8, n_applicants) + 0.11 * financial_strain
    )
    credit_utilization = np.clip(credit_utilization, 0.01, 1.0)

    late_payment_rate = np.exp(-0.9 + 0.58 * financial_strain)
    late_payments = rng.poisson(late_payment_rate)
    late_payments = np.clip(late_payments, 0, 10)

    credit_history_years = rng.gamma(3.0, 3.4, n_applicants)
    credit_history_years -= 0.65 * financial_strain
    credit_history_years = np.clip(credit_history_years, 0.5, 35)

    employment_years = rng.gamma(2.3, 2.8, n_applicants)
    employment_years -= 0.25 * financial_strain
    employment_years = np.clip(employment_years, 0, 30)

    loan_to_income = rng.uniform(0.08, 0.48, n_applicants)
    loan_to_income += 0.025 * financial_strain
    loan_amount = annual_income * np.clip(loan_to_income, 0.05, 0.60)
    loan_amount = np.clip(loan_amount, 2_000, 80_000)

    interest_rate = (
        5.0
        + 7.0 * credit_utilization
        + 4.0 * debt_to_income_ratio
        + 0.55 * late_payments
        - 0.07 * credit_history_years
        + rng.normal(0, 1.6, n_applicants)
    )
    interest_rate = np.clip(interest_rate, 3.5, 29.0)

    # Standardized terms make the relative effects easy to inspect and adjust.
    income_z = (np.log(annual_income) - np.log(68_000)) / 0.48
    dti_z = (debt_to_income_ratio - 0.30) / 0.13
    utilization_z = (credit_utilization - 0.43) / 0.25
    history_z = (credit_history_years - 10.0) / 7.0
    employment_z = (employment_years - 6.0) / 5.0
    loan_burden_z = (loan_amount / annual_income - 0.28) / 0.12
    interest_z = (interest_rate - 10.5) / 3.5

    risk_score = (
        -0.22 * income_z
        + 0.65 * dti_z
        + 0.82 * utilization_z
        + 1.05 * late_payments
        - 0.90 * history_z
        - 0.20 * employment_z
        + 0.34 * loan_burden_z
        + 0.28 * interest_z
        + 0.45 * (credit_utilization > 0.80)
        + 0.55 * ((debt_to_income_ratio > 0.45) & (late_payments >= 2))
        + rng.normal(0, 0.75, n_applicants)
    )

    intercept = calibrate_intercept(risk_score, target_default_rate)
    default_probability = sigmoid(risk_score + intercept)
    default = rng.binomial(1, default_probability)

    data = pd.DataFrame(
        {
            "annual_income": np.round(annual_income, 2),
            "debt_to_income_ratio": np.round(debt_to_income_ratio, 3),
            "credit_utilization": np.round(credit_utilization, 3),
            "late_payments": late_payments.astype(int),
            "credit_history_years": np.round(credit_history_years, 1),
            "loan_amount": np.round(loan_amount, 2),
            "interest_rate": np.round(interest_rate, 2),
            "employment_years": np.round(employment_years, 1),
            "default": default.astype(int),
        }
    )

    validate_data(data, n_applicants)
    return data


def validate_data(data: pd.DataFrame, expected_rows: int) -> None:
    """Fail immediately if generation produces invalid data."""
    expected_columns = {
        "annual_income",
        "debt_to_income_ratio",
        "credit_utilization",
        "late_payments",
        "credit_history_years",
        "loan_amount",
        "interest_rate",
        "employment_years",
        "default",
    }
    if len(data) != expected_rows:
        raise ValueError("Generated row count does not match the request.")
    if set(data.columns) != expected_columns:
        raise ValueError("Generated columns do not match the project schema.")
    if data.isna().any().any():
        raise ValueError("Generated data contains missing values.")
    if not set(data["default"].unique()).issubset({0, 1}):
        raise ValueError("Default must contain only 0 and 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-rate", type=float, default=0.18)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_credit_data(args.rows, args.seed, args.default_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)

    print(f"Saved {len(data):,} applicants to {args.output}")
    print(f"Observed default rate: {data['default'].mean():.2%}")
    print(f"Missing values: {int(data.isna().sum().sum())}")


if __name__ == "__main__":
    main()
