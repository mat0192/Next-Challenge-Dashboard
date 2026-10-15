import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Risk Terminal", layout="wide")
st.title("Terminal de Risque - Front Office 📈")
st.subheader("Portefeuille Next Challenge")

# 1. Inventaire exact (Quantités corrigées sur NextWise)
portfolio_positions = {
    "S500.PA": 29, "TTE.PA": 50, "ACA.PA": 245, "SAN.PA": 55,
    "MC.PA": 11, "SAF.PA": 13, "ASML.AS": 3, "AI.PA": 22, 
    "ORA.PA": 280, "STLAP.PA": 980
}

# 2. Prix d'achat unitaires (PRU) exacts relevés sur NextWise
prix_achat = {
    "S500.PA": 205.05, "TTE.PA": 79.31, "ACA.PA": 16.31, "SAN.PA": 71.91,
    "MC.PA": 377.90, "SAF.PA": 318.40, "ASML.AS": 1552.20, "AI.PA": 170.98, 
    "ORA.PA": 13.84, "STLAP.PA": 3.93
}

company_names = {
    "S500.PA": "Amundi S&P 500 ETF", "TTE.PA": "TotalEnergies", "ACA.PA": "Crédit Agricole",
    "SAN.PA": "Sanofi", "MC.PA": "LVMH", "SAF.PA": "Safran",
    "ASML.AS": "ASML Holding", "AI.PA": "Air Liquide", "ORA.PA": "Orange",
    "STLAP.PA": "Stellantis"
}

# Paramètres de compte NextWise
apport_initial = 50959.17
liquidites_actuelles = 8662.64

@st.cache_data(ttl=300)
def load_data():
    tickers = list(portfolio_positions.keys())
    data = yf.download(tickers, period="1y", progress=False)['Close']
    return data

try:
    data = load_data()
    current_prices = data.iloc[-1]
    quantities = pd.Series(portfolio_positions)
    buy_prices = pd.Series(prix_achat)
    
    # Valorisations et PnL
    position_values = current_prices * quantities
    initial_values = buy_prices * quantities
    pnl_euro = position_values - initial_values
    pnl_pct = ((current_prices - buy_prices) / buy_prices) * 100
    
    total_actions = position_values.sum()
    valeur_totale_portefeuille = total_actions + liquidites_actuelles
    perf_globale_euro = valeur_totale_portefeuille - apport_initial
    perf_globale_pct = (perf_globale_euro / apport_initial) * 100
    
    # Modélisation de la Value at Risk (VaR 95%)
    weights_risk = position_values / total_actions
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights_risk)
    var_95 = np.percentile(portfolio_returns, 5)
    var_euro = total_actions * var_95
    
    # Traduction des tickers en noms d'entreprises
    position_values.index = position_values.index.map(company_names)
    current_prices.index = current_prices.index.map(company_names)
    buy_prices.index = buy_prices.index.map(company_names)
    quantities.index = quantities.index.map(company_names)
    pnl_euro.index = pnl_euro.index.map(company_names)
    pnl_pct.index = pnl_pct.index.map(company_names)
    returns.columns = returns.columns.map(company_names)
    
    # --- MÉTRIQUES HAUT DE PAGE ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Valorisation Totale", 
        value=f"{valeur_totale_portefeuille:,.2f} €", 
        delta=f"{perf_globale_pct:.2f} % ({perf_globale_euro:+.2f} €)"
    )
    col2.metric(label="Valorisation des Titres", value=f"{total_actions:,.2f} €")
    col3.metric(label="Liquidités Disponibles", value=f"{liquidites_actuelles:,.2f} €")
    col4.metric(
        label="Value at Risk (95%) 1J", 
        value=f"{var_euro:,.2f} €", 
        delta=f"{var_95*100:.2f} %", 
        delta_color="inverse"
    )
    
    st.write("---")
    
    # --- GRAPHIQUE & CORRÉLATION ---
    col_chart, col_corr = st.columns(2)
    with col_chart:
        st.write("### Exposition par Actif (€)")
        st.bar_chart(position_values)
    with col_corr:
        st.write("### Matrice de Corrélation (1 an)")
        st.dataframe(returns.corr().round(2))

    st.write("---")
    
    # --- RADAR DE CONFORMITÉ & SUIVI PNL ---
    st.write("### 🚨 Radar de Conformité (Max 15%) & Suivi PnL")
    
    poids_reels = (position_values / valeur_totale_portefeuille) * 100
    plafond_15_pct = valeur_totale_portefeuille * 0.15
    marge_achat_euro = plafond_15_pct - position_values
    
    df_suivi = pd.DataFrame({
        "Quantité": quantities,
        "PRU (€)": buy_prices,
        "Cours Actuel (€)": current_prices,
        "Capital Actuel (€)": position_values,
        "PnL (€)": pnl_euro,
        "PnL (%)": pnl_pct,
        "Poids Portefeuille (%)": poids_reels,
        "Marge avant 15% (€)": marge_achat_euro
    }).sort_values(by="Poids Portefeuille (%)", ascending=False)
    
    def colorer_limites(val):
        if val >= 15:
            return 'color: white; background-color: darkred'
        elif val >= 13:
            return 'color: black; background-color: orange'
        return 'color: white; background-color: darkgreen'
        
    def colorer_pnl(val):
        if val > 0:
            return 'color: #00ff88'
        elif val < 0:
            return 'color: #ff4b4b'
        return ''

    st.dataframe(
        df_suivi.style
        .map(colorer_limites, subset=['Poids Portefeuille (%)'])
        .map(colorer_pnl, subset=['PnL (€)', 'PnL (%)'])
        .format({
            "Quantité": "{:.0f}",
            "PRU (€)": "{:.2f} €",
            "Cours Actuel (€)": "{:.2f} €",
            "Capital Actuel (€)": "{:,.2f} €",
            "PnL (€)": "{:+.2f} €",
            "PnL (%)": "{:+.2f} %",
            "Poids Portefeuille (%)": "{:.2f} %",
            "Marge avant 15% (€)": "{:,.2f} €"
        }),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Erreur serveur : {e}")
