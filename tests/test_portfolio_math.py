import json
import unittest

import numpy as np
import pandas as pd

from portfolio_config import default_holdings, validate_holdings, validate_backup
from portfolio_math import allocation, risk_metrics, simulate_order, max_buy_quantity, wilder_rsi, market_panels


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.q = pd.Series(10.0, index=[f"T{i}" for i in range(10)])
        self.p = pd.Series(100.0, index=self.q.index)
        self.sectors = {t: f"Secteur {i % 4}" for i, t in enumerate(self.q.index)}
        self.returns = pd.DataFrame(np.random.default_rng(42).normal(0.0002, .015, (252, 10)), columns=self.q.index)
        self.w = pd.Series(.08, index=self.q.index)

    def test_components_reconcile(self):
        risk = risk_metrics(self.returns, self.w, 12500)
        self.assertAlmostEqual(risk["var_components"].sum(), risk["param_var"], places=10)
        self.assertAlmostEqual(risk["es_components"].sum(), risk["hist_es"], places=10)
        self.assertAlmostEqual(risk["vol_shares"].sum(), 1)
        self.assertGreaterEqual(risk["param_es"], risk["param_var"])
        self.assertGreaterEqual(risk["hist_es"], risk["hist_var"])

    def test_normal_expected_shortfall_analytic(self):
        sample = np.tile([-1., 1.], 50)
        sample = sample / np.std(sample, ddof=1) * .01
        risk = risk_metrics(pd.DataFrame({"A": sample}), pd.Series({"A": 1.}), 10000)
        self.assertAlmostEqual(risk["param_var"], 164.485362695, places=7)
        self.assertAlmostEqual(risk["param_es"], 206.271280751, places=7)
        self.assertAlmostEqual(risk["var_components"]["A"], risk["param_var"])

    def test_permutation_and_cash_scaling(self):
        baseline = risk_metrics(self.returns, self.w, 12500)
        shuffled = risk_metrics(self.returns[self.returns.columns[::-1]], self.w, 12500)
        half = risk_metrics(self.returns, self.w / 2, 12500)
        for key in ["vol", "param_var", "param_es", "hist_var", "hist_es"]:
            self.assertAlmostEqual(shuffled[key], baseline[key])
            self.assertAlmostEqual(half[key], baseline[key] / 2)

    def test_all_cash_and_constant_prices(self):
        risk = risk_metrics(self.returns, self.w * 0, 12500)
        for key in ["vol", "hist_var", "hist_es", "param_var", "param_es", "max_drawdown"]:
            self.assertEqual(risk[key], 0)
        self.assertTrue(np.isnan(risk["sharpe"]))
        flat = risk_metrics(self.returns * 0, self.w, 12500)
        self.assertEqual(flat["vol"], 0)

    def test_initial_loss_in_drawdown(self):
        returns = pd.DataFrame({"A": [-.1] + [0.] * 29})
        risk = risk_metrics(returns, pd.Series({"A": 1.}), 100)
        self.assertAlmostEqual(risk["max_drawdown"], -.1)

    def test_hhi_excludes_cash(self):
        a = allocation(self.p, self.q, 0, self.sectors)
        b = allocation(self.p, self.q, 10000, self.sectors)
        self.assertAlmostEqual(a["hhi"], .1)
        self.assertAlmostEqual(a["effective_lines"], 10)
        self.assertEqual(a["hhi"], b["hhi"])
        self.assertFalse(a["violations"])

    def test_etf_not_extra_sector(self):
        sectors = dict.fromkeys(self.q.index, "Secteur A")
        sectors["T0"] = "ETF diversifié"
        self.assertEqual(allocation(self.p, self.q, 0, sectors)["n_sectors"], 1)

    def test_max_quantity_boundary_and_cash(self):
        maximum = max_buy_quantity(self.p, self.q, 2000, "T0")
        a = simulate_order(self.p, self.q, self.p, 2000, "T0", "Achat", maximum, 100, self.sectors)
        b = simulate_order(self.p, self.q, self.p, 2000, "T0", "Achat", maximum + 1, 100, self.sectors)
        self.assertLessEqual(a["after"]["weights"]["T0"], .15)
        self.assertGreater(b["after"]["weights"]["T0"], .15)
        self.assertEqual(max_buy_quantity(self.p, self.q, 50, "T0"), 0)

    def test_sale_zero_line_and_conservation(self):
        result = simulate_order(self.p, self.q, self.p, 2000, "T0", "Vente", 10, 105, self.sectors)
        self.assertEqual(result["quantities"]["T0"], 0)
        self.assertEqual(result["after"]["n_lines"], 9)
        self.assertTrue(any("minimum solo de 10" in s for s in result["after"]["violations"]))
        self.assertAlmostEqual(result["after"]["total"], result["before"]["total"])
        self.assertTrue(np.isnan(result["pru"]))

    def test_invalid_orders_do_not_mutate_holdings(self):
        for side, qty in [("Vente", 11), ("Achat", 21), ("Achat", .5)]:
            with self.assertRaises(ValueError):
                simulate_order(self.p, self.q, self.p, 2000, "T0", side, qty, 100, self.sectors)
        self.assertEqual(self.q["T0"], 10)

    def test_purchase_pru(self):
        result = simulate_order(self.p, self.q, self.p, 2000, "T0", "Achat", 2, 110, self.sectors)
        self.assertAlmostEqual(result["pru"], 1220 / 12)
        self.assertAlmostEqual(result["before"]["total"], result["after"]["total"])

    def test_no_silent_missing_price(self):
        prices = self.p.copy()
        prices["T0"] = np.nan
        with self.assertRaises(ValueError):
            allocation(prices, self.q, 2000, self.sectors)
        with self.assertRaises(ValueError):
            risk_metrics(self.returns.drop(columns="T0"), self.w, 12500)

    def test_raw_vs_adjusted_and_no_forward_fill(self):
        raw = pd.DataFrame({"A": [100., 102., np.nan, 105.]})
        adjusted = pd.DataFrame({"A": [90., 91.8, np.nan, 94.5]})
        p, _, history, rets = market_panels(pd.concat({"Close": raw, "Adj Close": adjusted}, axis=1), ["A"])
        self.assertEqual(p["A"], 105)
        self.assertEqual(history.iloc[-1, 0], 94.5)
        self.assertEqual(len(rets), 1)
        with self.assertRaises(ValueError):
            market_panels(pd.concat({"Close": raw, "Adj Close": adjusted}, axis=1), ["A", "B"])

    def test_rsi_wilder_seed_and_missing(self):
        close = pd.DataFrame({"up": np.arange(30.) + 10, "flat": np.ones(30) * 10, "down": 40 - np.arange(30.)})
        rsi = wilder_rsi(close)
        self.assertTrue(rsi.iloc[:14].isna().all().all())
        self.assertEqual(rsi.iloc[14]["up"], 100)
        self.assertEqual(rsi.iloc[14]["flat"], 50)
        self.assertEqual(rsi.iloc[14]["down"], 0)
        close.iloc[20, 0] = np.nan
        self.assertTrue(np.isnan(wilder_rsi(close).iloc[-1]["up"]))

    def test_config_and_backup_round_trip(self):
        frame = default_holdings()
        backup = dict(version=1, holdings=frame.to_dict("records"), capital=50959.17, cash=8662.64, rf=0., journal=[{"Thèse": "Test"}])
        restored = validate_backup(json.loads(json.dumps(backup)))
        pd.testing.assert_frame_equal(restored["holdings"], validate_holdings(frame))
        self.assertEqual(restored["journal"], backup["journal"])
        for field, invalid in [("cash", -1), ("capital", 0), ("rf", float("nan")), ("journal", "bad")]:
            with self.assertRaises(ValueError):
                validate_backup({**backup, field: invalid})
        duplicated = pd.concat([frame, frame.iloc[[0]]])
        with self.assertRaises(ValueError):
            validate_holdings(duplicated)


if __name__ == "__main__":
    unittest.main()
