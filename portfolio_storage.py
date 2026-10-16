"""Restore before writing; keep positions independent from the market-data cache."""
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from portfolio_config import INITIAL_CAPITAL, INITIAL_CASH, default_holdings, validate_backup

STORAGE_KEY = "nextchallenge.solo.portfolio.v1"
bridge = st.components.v2.component(
    "portfolio_storage",
    html='<div role="status" aria-live="polite">Lecture de ta sauvegarde…</div>',
    css='div { font-size: 0.85rem; color: var(--st-text-color); }',
    js=Path(__file__).with_suffix(".js").read_text(encoding="utf-8"),
)


def snapshot(state):
    """The portable v1 format intentionally excludes downloaded prices."""
    return dict(version=1, capital=float(state["capital"]), cash=float(state["cash"]),
                rf=float(state["rf"]), holdings=state["holdings"].to_dict("records"),
                journal=state["journal"])


def encode(state):
    return json.dumps(snapshot(state), ensure_ascii=False, sort_keys=True, allow_nan=False)


def defaults(state):
    for key, value in dict(capital=INITIAL_CAPITAL, cash=INITIAL_CASH, rf=0.,
                           holdings=default_holdings(), journal=[]).items():
        if key not in state:
            state[key] = value


def sync_portfolio():
    """Mount once per run, serialize writes, and acknowledge before another write."""
    state = st.session_state
    if "_portfolio_store" not in state:
        state._portfolio_store = dict(ready=False, raw=None, blocked=None, command=None)
    store = state._portfolio_store
    if store["ready"]:
        defaults(state)
        if "pending_restore" in state:
            state.update(state.pop("pending_restore"))
    if store["command"] is None:
        if not store["ready"]:
            store["command"] = dict(action="read", id=uuid4().hex, key=STORAGE_KEY)
        elif not store["blocked"] and encode(state) != store["raw"]:
            store["command"] = dict(action="write", id=uuid4().hex, key=STORAGE_KEY,
                                     expected=store["raw"], raw=encode(state))
    command = store["command"] or dict(action="idle", message=(
        "Sauvegarde automatique suspendue" if store["blocked"] else "✓ Enregistré dans ce navigateur"))
    result = bridge(data=command, key="portfolio_browser_store", default={"result": None},
                    on_result_change=lambda: None).result
    if result and command.get("id") == result.get("id"):
        store["command"] = None
        status = result.get("status")
        if status == "read":
            store["raw"] = result["raw"]
            if result["raw"] is not None:
                try:
                    if len(result["raw"]) > 2_000_000:
                        raise ValueError("Sauvegarde trop volumineuse")
                    state.update(validate_backup(json.loads(result["raw"])))
                except (ValueError, KeyError, TypeError, OverflowError):
                    store["blocked"] = "La sauvegarde locale est illisible. Elle n’a pas été écrasée."
            store["ready"] = True
        elif status == "saved":
            store["raw"] = result["raw"]
        elif status == "conflict":
            store["raw"] = result["raw"]
            store["blocked"] = "Un autre onglet a modifié le portefeuille. Recharge cette page pour récupérer sa version."
        else:
            store["blocked"] = "Le navigateur ne permet pas l’enregistrement. Exporte ton JSON avant de fermer la page."
            store["ready"] = True
        st.rerun()
    if not store["ready"]:
        st.info("Chargement de tes positions enregistrées dans ce navigateur…")
        if st.button("Continuer sans sauvegarde automatique"):
            store.update(ready=True, command=None, blocked="Mode temporaire : exporte ton JSON avant de fermer la page.")
            st.rerun()
        st.stop()
    if store["blocked"]:
        st.warning(store["blocked"])
        if store["raw"] is not None:
            st.download_button("Récupérer la sauvegarde locale existante", store["raw"],
                               file_name="nextchallenge_sauvegarde_locale.json", mime="application/json")
        if st.button("Enregistrer les données affichées à la place", help="Remplace la sauvegarde locale par le portefeuille actuellement affiché."):
            store.update(blocked=None, command=dict(action="write", id=uuid4().hex,
                         key=STORAGE_KEY, expected=store["raw"], raw=encode(state)))
            st.rerun()
    st.caption("Positions, cash et journal conservés sur ce navigateur. Exporte un JSON pour changer d’appareil ou garder une copie.")


def backup_controls():
    """Available even when Yahoo quotes fail."""
    st.download_button("Exporter ma sauvegarde (positions + journal)",
                       json.dumps(snapshot(st.session_state), ensure_ascii=False, indent=2),
                       file_name="nextchallenge_sauvegarde.json", mime="application/json")
    imported = st.file_uploader("Restaurer une sauvegarde du dashboard", type=["json"])
    if imported is not None and st.button("Importer ces positions et ce cash"):
        try:
            if imported.size > 2_000_000:
                raise ValueError("Fichier trop volumineux (maximum 2 Mo).")
            st.session_state.pending_restore = validate_backup(json.loads(imported.getvalue()))
            st.rerun()
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            st.error(f"Sauvegarde non chargée : {exc}")
    st.caption("Le JSON remplace les positions, le cash et le journal. Les cours restent récupérés séparément : importer ne restaure aucun ancien cours.")
