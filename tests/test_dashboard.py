from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from portfolio_config import default_holdings, validate_backup


APP = str(Path(__file__).resolve().parents[1] / "dashboard.py")


def fixture():
    base = default_holdings().set_index("Code Yahoo")
    rng = np.random.default_rng(77)
    close = pd.DataFrame(index=pd.bdate_range(end=datetime.now(ZoneInfo("Europe/Paris")).date(), periods=260))
    for ticker in sorted(base.index):
        path = np.exp(rng.normal(0.0002, 0.01, len(close)).cumsum())
        close[ticker] = path / path[-1] * base.loc[ticker, "PRU (€)"] * .99
    close["^FCHI"] = 8000 * np.exp(rng.normal(0.0002, .009, len(close)).cumsum())
    return pd.concat({"Close": close, "Adj Close": close * .95}, axis=1)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.data = fixture()
        self.download = patch("yfinance.download", return_value=self.data)
        self.download.start()
        self.addCleanup(self.download.stop)

    def app(self):
        return AppTest.from_file(APP, default_timeout=30).run()

    def assert_clean(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(len(app.tabs), 6)

    def test_all_tabs_and_twenty_simulations(self):
        app = self.app()
        self.assert_clean(app)
        self.assertFalse(app.error, [e.value for e in app.error])
        for ticker in default_holdings()["Code Yahoo"]:
            for side in ["Achat", "Vente"]:
                app.selectbox(key="sim_asset").select(ticker)
                app.radio(key="sim_side").set_value(side)
                app.number_input(key="sim_quantity").set_value(1)
                app.run()
                self.assert_clean(app)
                self.assertFalse(app.error, [e.value for e in app.error])
                tables = [d.value for d in app.dataframe if "Variation" in d.value.columns]
                self.assertEqual(len(tables), 1)
                self.assertIn("CVaR historique 95 % (€)", tables[0].index)

    def test_full_sale_oversell_cash_and_limit(self):
        app = self.app()
        app.radio(key="sim_side").set_value("Vente")
        app.number_input(key="sim_quantity").set_value(29).run()
        self.assert_clean(app)
        self.assertTrue(any("9 valeurs détenues" in w.value for w in app.warning))
        app.number_input(key="sim_quantity").set_value(30).run()
        self.assertTrue(any("Vente impossible" in e.value for e in app.error))
        app.radio(key="sim_side").set_value("Achat")
        app.number_input(key="sim_quantity").set_value(1000).run()
        self.assertTrue(any("Liquidités insuffisantes" in e.value for e in app.error))
        app.number_input(key="sim_quantity").set_value(10).run()
        self.assertTrue(any("plafond solo de 15" in w.value for w in app.warning))
        self.assert_clean(app)

    def test_journal_and_restore(self):
        app = self.app()
        for label, value in [("Thèse : pourquoi cette décision ?", "Hypothèse de test"),
                             ("Qu’est-ce qui invaliderait ma thèse ?", "Résultats contraires"),
                             ("Sources consultées et dates", "Document annuel de test")]:
            next(w for w in app.text_area if w.label == label).set_value(value)
        next(b for b in app.button if b.label == "Ajouter à mon journal").click().run()
        self.assert_clean(app)
        self.assertEqual(len(app.session_state["journal"]), 1)
        backup = dict(version=1, holdings=default_holdings().to_dict("records"), cash=10000.,
                      capital=60000., rf=2., journal=app.session_state["journal"])
        app.session_state["pending_restore"] = validate_backup(backup)
        app.run()
        self.assert_clean(app)
        self.assertEqual(app.number_input(key="cash").value, 10000.)
        self.assertEqual(app.number_input(key="capital").value, 60000.)
        self.assertEqual(len(app.session_state["journal"]), 1)

    def test_missing_quote_and_short_history(self):
        self.download.stop()
        bad = self.data.copy()
        bad[("Close", "S500.PA")] = np.nan
        with patch("yfinance.download", return_value=bad):
            st.cache_data.clear()
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any("S500.PA" in e.value for e in app.error))
            self.assertTrue(any(b.label == "Corriger les positions" for b in app.button))
        with patch("yfinance.download", return_value=self.data.tail(10)):
            st.cache_data.clear()
            app = self.app()
            self.assert_clean(app)
            self.assertTrue(any("Historique commun insuffisant" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
