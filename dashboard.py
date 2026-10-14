import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. Configuration de la page de ton site
st.set_page_config(page_title="Risk Terminal", layout="wide")
st.title("Terminal de Risque - Front Office 📈")
st.subheader("Portefeuille Next Challenge (50 000 €)")

# 2. Tes données (avec tes 10 valeurs)
portfolio_positions = {
    "500.PA": 29, "TTE.PA": 50, "ACA.PA": 245, "SAN.PA": 55,
    "MC.PA": 11, "SAF.PA": 13, "ASML.AS": 3, "AI.PA": 22, 
    "ORA.PA": 280, "STLAP.PA": 1015
}
tickers = list(portfolio_positions.keys())

# 3. Le moteur de calcul
st.write("Actualisation des données du marché en cours...")
data = yf.download(tickers, period="1y")['Close']
current_prices = data.iloc[-1]
quantities = pd.Series(portfolio_positions)
position_values = current_prices * quantities
total_portfolio_value = position_values.sum()

# 4. Affichage visuel sur le Dashboard
st.metric(label="Capital Actuellement Investi", value=f"{total_portfolio_value:,.2f} €")

st.write("### Répartition de tes positions")
st.bar_chart(position_values)
