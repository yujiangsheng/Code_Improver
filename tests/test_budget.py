"""Tests for the budget-cap helper in ada.budget."""
from __future__ import annotations

import os
import time
import unittest

from ada.budget import Budget, BudgetExceeded


class TestBudget(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot env so we can restore.
        self._saved = {k: os.environ.get(k) for k in
                       ("ADA_MAX_COST", "ADA_MAX_STEPS_HARD",
                        "ADA_MAX_TOKENS", "ADA_MAX_SECONDS")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_unset_means_no_caps(self) -> None:
        b = Budget.from_env()
        b.start()
        b.check(cost_usd=1e9, steps=10**9, tokens=10**9)  # no raise

    def test_cost_cap(self) -> None:
        os.environ["ADA_MAX_COST"] = "0.50"
        b = Budget.from_env()
        b.start()
        b.check(cost_usd=0.49)
        with self.assertRaises(BudgetExceeded) as cm:
            b.check(cost_usd=0.51)
        self.assertEqual(cm.exception.kind, "cost_usd")

    def test_step_cap(self) -> None:
        os.environ["ADA_MAX_STEPS_HARD"] = "3"
        b = Budget.from_env()
        b.start()
        b.check(steps=3)
        with self.assertRaises(BudgetExceeded):
            b.check(steps=4)

    def test_token_cap(self) -> None:
        os.environ["ADA_MAX_TOKENS"] = "100"
        b = Budget.from_env()
        b.start()
        with self.assertRaises(BudgetExceeded) as cm:
            b.check(tokens=101)
        self.assertEqual(cm.exception.kind, "tokens")

    def test_seconds_cap(self) -> None:
        os.environ["ADA_MAX_SECONDS"] = "0.001"
        b = Budget.from_env()
        b.start()
        time.sleep(0.01)
        with self.assertRaises(BudgetExceeded) as cm:
            b.check()
        self.assertEqual(cm.exception.kind, "seconds")

    def test_describe(self) -> None:
        os.environ["ADA_MAX_COST"] = "1.0"
        b = Budget.from_env()
        d = b.describe()
        self.assertEqual(d["cost_usd"], 1.0)
        self.assertIsNone(d["steps"])

    def test_malformed_env_ignored(self) -> None:
        os.environ["ADA_MAX_COST"] = "not-a-float"
        b = Budget.from_env()
        self.assertIsNone(b.max_cost_usd)


if __name__ == "__main__":
    unittest.main()
