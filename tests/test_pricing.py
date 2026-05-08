"""Unit tests for ada/pricing.py."""
from __future__ import annotations

import unittest

from ada.pricing import estimate_cost, set_price


class TestPricing(unittest.TestCase):
    def test_known_model_priced(self) -> None:
        c = estimate_cost("gpt-4o-mini", 1_000_000, 0)
        self.assertTrue(c["priced"])
        self.assertAlmostEqual(c["usd"], 0.15, places=5)

    def test_completion_costs_more(self) -> None:
        a = estimate_cost("gpt-4o", 0, 1_000_000)["usd"]
        b = estimate_cost("gpt-4o", 1_000_000, 0)["usd"]
        self.assertGreater(a, b)

    def test_prefix_match_for_dated_model(self) -> None:
        c = estimate_cost("claude-opus-4-20250514", 100_000, 100_000)
        self.assertTrue(c["priced"])
        self.assertGreater(c["usd"], 0)

    def test_local_model_zero_cost(self) -> None:
        c = estimate_cost("qwen3.5:9b", 1_000_000, 1_000_000)
        self.assertTrue(c["priced"])
        self.assertEqual(c["usd"], 0.0)

    def test_unknown_model_unpriced(self) -> None:
        c = estimate_cost("totally-fake-2099", 100, 100)
        self.assertFalse(c["priced"])
        self.assertIsNone(c["usd"])

    def test_set_price_runtime_override(self) -> None:
        set_price("test-stub", 1.0, 2.0)
        c = estimate_cost("test-stub-x", 1_000_000, 1_000_000)
        self.assertEqual(c["usd"], 3.0)


if __name__ == "__main__":
    unittest.main()
