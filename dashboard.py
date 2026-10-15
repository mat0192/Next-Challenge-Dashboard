import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Risk Terminal", layout="wide")
st.title("Terminal de Risque - Front Office 📈")
st.subheader("Portefeuille Next Challenge")

# 1. Ton inventaire brut
portfolio_positions = {
    "500.PA": 29, "TTE.PA": 50, "ACA.PA": 245, "SAN.PA": 55,
    "MC.PA": 11, "SAF.PA": 13, "ASML.AS": 3, "AI.PA": 22, 
    "ORA.PA": 280, "STLAP.PA": 1015
}

# 2. Le dictionnaire de traduction pour un affichage "Client"
company_names = {
    "500.PA": "S&P 500", "TTE.PA": "TotalEnergies", "ACA.PA": "Crédit Agricole",
    "SAN.PA": "Sanofi", "MC.PA": "LVMH", "SAF.PA": "Safran",
    "ASML.AS": "ASML", "AI.PA": "Air Liquide", "ORA.PA": "Orange",
    "STLAP.PA": "Stellantis"
}

@st.cache_data(ttl=300)
def load_data():
    tickers = list(portfolio_positions.keys())
    data = yf.download(tickers, period="1y", progress=False)['Close']
    return data

st.write("Actualisation des modèles de risque en cours...")

try:
    data = load_data()
    current_prices = data.iloc[-1]
    quantities = pd.Series(portfolio_positions)
    
    # Calcul des valorisations et des poids
    position_values = current_prices * quantities
    total_portfolio_value = position_values.sum()
    weights = position_values / total_portfolio_value
    
    # Modélisation de la Value at Risk (VaR 95%)
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    var_95 = np.percentile(portfolio_returns, 5)
    var_euro = total_portfolio_value * var_95
    
    # Remplacement des codes par les Vrais Noms d'entreprises
    position_values.index = position_values.index.map(company_names)
    returns.columns = returns.columns.map(company_names)
    
    # --- AFFICHAGE DASHBOARD ---
    col1, col2 = st.columns(2)
    col1.metric(label="Capital Actions Investi", value=f"{total_portfolio_value:,.2f} €")
    col2.metric(label="Value at Risk (95%) 1 Jour", value=f"{var_euro:,.2f} €", delta=f"{var_95*100:.2f} %", delta_color="inverse")
    
    st.write("---")
    
    col_chart, col_corr = st.columns(2)
    
    with col_chart:
        st.write("### Exposition par Actif")
        st.bar_chart(position_values)
        
    with col_corr:
        st.write("### Matrice de Corrélation (1 an)")
        corr_matrix = returns.corr()
        # Affichage brut et robuste de la matrice pour éviter le crash du serveur
        st.dataframe(corr_matrix)

except Exception as e:
    st.error(f"Le serveur a rencontré une erreur technique : {e}")
