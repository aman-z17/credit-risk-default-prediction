"""Tests for the synthetic credit-risk dataset generator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_data import generate_credit_data


class CreditDataGenerationTests(unittest.TestCase):
    def test_schema_and_shape(self):
        data = generate_credit_data(n_applicants=500, seed=42)
        self.assertEqual(data.shape, (500, 9))
        self.assertEqual(
            list(data.columns),
            [
                "annual_income",
                "debt_to_income_ratio",
                "credit_utilization",
                "late_payments",
                "credit_history_years",
                "loan_amount",
                "interest_rate",
                "employment_years",
                "default",
            ],
        )
        self.assertFalse(data.isna().any().any())
        self.assertTrue(set(data["default"].unique()).issubset({0, 1}))

    def test_fixed_seed_is_reproducible(self):
        first = generate_credit_data(n_applicants=500, seed=7)
        second = generate_credit_data(n_applicants=500, seed=7)
        self.assertTrue(first.equals(second))

    def test_financial_ranges(self):
        data = generate_credit_data(n_applicants=500, seed=42)
        self.assertTrue(data["annual_income"].between(18_000, 250_000).all())
        self.assertTrue(data["debt_to_income_ratio"].between(0.03, 0.75).all())
        self.assertTrue(data["credit_utilization"].between(0.01, 1.0).all())
        self.assertTrue(data["late_payments"].between(0, 10).all())
        self.assertTrue(data["credit_history_years"].between(0.5, 35).all())


if __name__ == "__main__":
    unittest.main()

