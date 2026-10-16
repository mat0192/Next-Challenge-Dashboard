from datetime import datetime
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from portfolio_config import default_holdings, validate_backup


APP = str(Path(__file__).resolve().parents[1] / "dashboard.py")


class BrowserStore:
    """Transport stub; the real JavaScript protocol is also tested with Node."""
    def __init__(self):
        self.raw = None
        self.writes = []
        self.error = False

    def __call__(self, *, data, **kwargs):
        if data["action"] == "idle":
            return SimpleNamespace(result=None)
        result = dict(id=data["id"], raw=self.raw)
        if self.error:
            result["status"] = "error"
        elif data["action"] == "read":
            result["status"] = "read"
        elif self.raw not in (data["expected"], data["raw"]):
            result["status"] = "conflict"
        else:
            self.raw = data["raw"]
            self.writes.append(self.raw)
            result.update(status="saved", raw=self.raw)
        return SimpleNamespace(result=result)


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
        self.download_mock = self.download.start()
        self.addCleanup(self.download.stop)
        self.browser = BrowserStore()
        self.storage = patch("portfolio_storage.bridge", side_effect=self.browser)
        self.storage.start()
        self.addCleanup(self.storage.stop)

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
            self.assertTrue(any(b.label == "Exporter ma sauvegarde (positions + journal)" for b in app.get("download_button")))
        with patch("yfinance.download", return_value=self.data.tail(10)):
            st.cache_data.clear()
            app = self.app()
            self.assert_clean(app)
            self.assertTrue(any("Historique commun insuffisant" in w.value for w in app.warning))

    def test_import_refresh_and_new_browser_session(self):
        app = self.app()
        backup = dict(version=1, holdings=default_holdings().to_dict("records"),
                      cash=12345.67, capital=60000., rf=1.5,
                      journal=[{"Thèse": "À conserver après fermeture"}])
        backup["holdings"][-1]["Quantité"] = 479
        app.session_state["pending_restore"] = validate_backup(backup)
        app.run()
        self.assert_clean(app)
        saved = json.loads(self.browser.raw)
        self.assertEqual(saved, backup)
        before = app.metric[0].value
        self.download_mock.return_value = self.data * 1.1
        next(b for b in app.button if b.label == "Actualiser les cours").click().run()
        self.assertNotEqual(app.metric[0].value, before)
        self.assertEqual(json.loads(self.browser.raw), backup)
        reopened = self.app()  # Empty Streamlit state, same browser storage.
        self.assert_clean(reopened)
        self.assertEqual(reopened.number_input(key="cash").value, 12345.67)
        self.assertEqual(reopened.session_state["holdings"].iloc[-1]["Quantité"], 479)
        self.assertEqual(reopened.session_state["journal"], backup["journal"])
        reopened.number_input(key="cash").set_value(12000.).run()
        self.assertEqual(json.loads(self.browser.raw)["cash"], 12000.)
        self.assertEqual(self.app().number_input(key="cash").value, 12000.)

    def test_invalid_storage_and_conflict_are_not_overwritten(self):
        self.browser.raw = '{"version": 1, "holdings": []}'
        original = self.browser.raw
        app = self.app()
        self.assert_clean(app)
        self.assertEqual(self.browser.raw, original)
        self.assertFalse(self.browser.writes)
        self.assertTrue(any("illisible" in w.value for w in app.warning))
        self.browser.raw = None
        app = self.app()
        external = json.loads(self.browser.raw)
        external["cash"] = 15000.
        self.browser.raw = json.dumps(external)
        app.number_input(key="cash").set_value(12000.).run()
        self.assertEqual(json.loads(self.browser.raw)["cash"], 15000.)
        self.assertTrue(any("autre onglet" in w.value for w in app.warning))
        self.assertEqual(self.app().number_input(key="cash").value, 15000.)

    def test_blocked_storage_keeps_manual_export_available(self):
        self.browser.error = True
        app = self.app()
        self.assert_clean(app)
        self.assertTrue(any("ne permet pas" in w.value for w in app.warning))
        self.assertFalse(self.browser.writes)
        self.assertTrue(app.get("download_button"))


if __name__ == "__main__":
    unittest.main()
