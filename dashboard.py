import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Risk Terminal", layout="wide")
st.title("Terminal de Risque - Front Office 📈")
st.subheader("Portefeuille Next Challenge")

portfolio_positions = {
    "500.PA": 29, "TTE.PA": 50, "ACA.PA": 245, "SAN.PA": 55,
    "MC.PA": 11, "SAF.PA": 13, "ASML.AS": 3, "AI.PA": 22, 
    "ORA.PA": 280, "STLAP.PA": 1015
}

company_names = {
    "500.PA": "S&P 500", "TTE.PA": "TotalEnergies", "ACA.PA": "Crédit Agricole",
    "SAN.PA": "Sanofi", "MC.PA": "LVMH", "SAF.PA": "Safran",
    "ASML.AS": "ASML", "AI.PA": "Air Liquide", "ORA.PA": "Orange",
    "STLAP.PA": "Stellantis"
}

# ⚠️ Modifie ce chiffre avec les liquidités (cash) exactes qu'il te reste sur NextWise
liquidites_actuelles = 10378.07

@st.cache_data(ttl=300)
def load_data():
    tickers = list(portfolio_positions.keys())
    data = yf.download(tickers, period="1y", progress=False)['Close']
    return data

try:
    data = load_data()
    current_prices = data.iloc[-1]
    quantities = pd.Series(portfolio_positions)
    
    position_values = current_prices * quantities
    total_actions = position_values.sum()
    
    # Calcul de la NAV (Net Asset Value) réelle du portefeuille
    valeur_totale_portefeuille = total_actions + liquidites_actuelles
    
    # Poids mathématiques pour la VaR (uniquement sur la partie risquée)
    weights_risk = position_values / total_actions
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights_risk)
    var_95 = np.percentile(portfolio_returns, 5)
    var_euro = total_actions * var_95
    
    position_values.index = position_values.index.map(company_names)
    returns.columns = returns.columns.map(company_names)
    
    # --- DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Valeur Totale Portefeuille", value=f"{valeur_totale_portefeuille:,.2f} €")
    col2.metric(label="Capital Actions Investi", value=f"{total_actions:,.2f} €")
    col3.metric(label="Value at Risk (95%) 1 Jour", value=f"{var_euro:,.2f} €", delta=f"{var_95*100:.2f} %", delta_color="inverse")
    
    st.write("---")
    
    col_chart, col_corr = st.columns(2)
    with col_chart:
        st.write("### Exposition par Actif")
        st.bar_chart(position_values)
    with col_corr:
        st.write("### Matrice de Corrélation (1 an)")
        st.dataframe(returns.corr())

    st.write("---")
    
    # --- MODULE DE CONFORMITÉ (RÈGLE DES 15%) ---
    st.write("### 🚨 Radar de Conformité (Max 15% par ligne)")
    
    # Calcul du poids réel par rapport à la valeur TOTALE (avec cash)
    poids_reels = (position_values / valeur_totale_portefeuille) * 100
    
    df_conformite = pd.DataFrame({
        "Valeur Actuelle (€)": position_values,
        "Poids dans le Portefeuille (%)": poids_reels
    }).sort_values(by="Poids dans le Portefeuille (%)", ascending=False)
    
    # Formatage visuel (rouge si > 14%, jaune si > 12%, vert sinon)
    def colorer_limites(val):
        if val >= 15:
            return 'color: white; background-color: darkred'
        elif val >= 13:
            return 'color: black; background-color: orange'
        return 'color: white; background-color: darkgreen'

    st.dataframe(df_conformite.style.map(colorer_limites, subset=['Poids dans le Portefeuille (%)']).format("{:.2f}"))

except Exception as e:
    st.error(f"Erreur serveur : {e}")
