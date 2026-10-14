import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Risk Terminal", layout="wide")
st.title("Terminal de Risque - Front Office 📈")
st.subheader("Portefeuille Next Challenge (50 000 €)")

portfolio_positions = {
    "500.PA": 29, "TTE.PA": 50, "ACA.PA": 245, "SAN.PA": 55,
    "MC.PA": 11, "SAF.PA": 13, "ASML.AS": 3, "AI.PA": 22, 
    "ORA.PA": 280, "STLAP.PA": 1015
}

# Fonction sécurisée avec cache mémoire (ttl=60s) pour ne pas se faire bloquer par Yahoo
@st.cache_data(ttl=60)
def load_data():
    tickers = list(portfolio_positions.keys())
    # On télécharge seulement 5 jours pour aller très vite
    data = yf.download(tickers, period="5d", progress=False)
    return data['Close'].iloc[-1]

st.write("Actualisation des données du marché en cours...")

try:
    current_prices = load_data()
    quantities = pd.Series(portfolio_positions)
    
    # Calcul strict des valorisations
    position_values = current_prices * quantities
    total_portfolio_value = position_values.sum()

    st.metric(label="Capital Actuellement Investi", value=f"{total_portfolio_value:,.2f} €")
    
    st.write("### Répartition de tes positions")
    st.bar_chart(position_values)
except Exception as e:
    st.error("Les serveurs de Yahoo Finance limitent l'accès. Actualise la page dans quelques secondes.")
