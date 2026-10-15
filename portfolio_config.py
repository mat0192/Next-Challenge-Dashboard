"""Portefeuille de départ et validation des modifications de session."""
import re
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 50959.17
INITIAL_CASH = 8662.64
RULES_URL = "https://www.nextwise.fr/R%C3%A8glement-du-Next-Challenge?lng=fr"
RULES_CHECKED = "15/10/2026"
SECTORS = ["Énergie", "Finance", "Santé", "Consommation discrétionnaire", "Industrie",
           "Technologie", "Matériaux", "Télécommunications", "Consommation de base",
           "Services aux collectivités", "Immobilier", "ETF diversifié", "Non classé"]
DEFAULT_HOLDINGS = [
    ["S500.PA", "Amundi S&P 500 ETF", "ETF diversifié", 29, 205.05],
    ["TTE.PA", "TotalEnergies", "Énergie", 50, 79.31],
    ["ACA.PA", "Crédit Agricole", "Finance", 245, 16.31],
    ["SAN.PA", "Sanofi", "Santé", 55, 71.91],
    ["MC.PA", "LVMH", "Consommation discrétionnaire", 11, 377.90],
    ["SAF.PA", "Safran", "Industrie", 13, 318.40],
    ["ASML.AS", "ASML Holding", "Technologie", 3, 1552.20],
    ["AI.PA", "Air Liquide", "Matériaux", 22, 170.98],
    ["ORA.PA", "Orange", "Télécommunications", 280, 13.84],
    ["STLAP.PA", "Stellantis", "Consommation discrétionnaire", 980, 3.93],
]
COLUMNS = ["Code Yahoo", "Nom", "Secteur", "Quantité", "PRU (€)"]


def default_holdings():
    return pd.DataFrame(DEFAULT_HOLDINGS, columns=COLUMNS)


def validate_holdings(frame):
    if not set(COLUMNS).issubset(frame.columns) or not 1 <= len(frame) <= 60:
        raise ValueError("Le tableau doit contenir entre 1 et 60 lignes et toutes les colonnes attendues.")
    frame = frame[COLUMNS].copy()
    if frame[COLUMNS[:3]].isna().any().any():
        raise ValueError("Complète le code, le nom et le secteur de chaque ligne.")
    for column in COLUMNS[:3]:
        frame[column] = frame[column].astype(str).str.strip()
    frame["Code Yahoo"] = frame["Code Yahoo"].str.upper()
    if frame["Code Yahoo"].duplicated().any() or not frame["Code Yahoo"].map(lambda x: bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.=-]{0,24}", x))).all():
        raise ValueError("Chaque code Yahoo doit être valide et unique.")
    if not frame["Nom"].map(lambda x: 0 < len(x) <= 100).all():
        raise ValueError("Chaque nom doit contenir de 1 à 100 caractères.")
    if not frame["Secteur"].isin(SECTORS).all():
        raise ValueError("Choisis un secteur dans la liste proposée.")
    for column in COLUMNS[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError("Les quantités et PRU doivent être des nombres positifs ou nuls.")
    if (frame["Quantité"] % 1 != 0).any() or not (frame["Quantité"] > 0).any():
        raise ValueError("Utilise des quantités entières, avec au moins une ligne détenue.")
    frame["Quantité"] = frame["Quantité"].astype(int)
    return frame


def validate_backup(restored):
    if not isinstance(restored, dict) or restored.get("version") != 1:
        raise ValueError("Format de sauvegarde non reconnu.")
    frame = validate_holdings(pd.DataFrame(restored["holdings"]))
    capital, cash, rf = [float(restored[k]) for k in ("capital", "cash", "rf")]
    if not all(np.isfinite(v) for v in (capital, cash, rf)) or capital <= 0 or cash < 0 or not 0 <= rf <= 20:
        raise ValueError("Paramètres du compte invalides.")
    journal = restored.get("journal", [])
    if not isinstance(journal, list) or len(journal) > 1000 or not all(isinstance(e, dict) and all(isinstance(v, str) for v in e.values()) for e in journal):
        raise ValueError("Journal invalide.")
    return dict(holdings=frame, capital=capital, cash=cash, rf=rf, journal=journal)
