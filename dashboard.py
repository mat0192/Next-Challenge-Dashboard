import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Risk Terminal - Front Office", layout="wide")
st.title("Terminal de Risque & Aide à la Décision 📈")
st.subheader("Portefeuille Next Challenge — Euronext")

# 1. Inventaire et PRU NextWise
portfolio_positions = {
    "S500.PA": 29, "TTE.PA": 50, "ACA.PA": 245, "SAN.PA": 55,
    "MC.PA": 11, "SAF.PA": 13, "ASML.AS": 3, "AI.PA": 22, 
    "ORA.PA": 280, "STLAP.PA": 980
}

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

apport_initial = 50959.17
liquidites_actuelles = 8662.64

@st.cache_data(ttl=300)
def load_data():
    tickers = list(portfolio_positions.keys()) + ["^FCHI"]
    data = yf.download(tickers, period="1y", progress=False)['Close']
    return data

try:
    raw_data = load_data()
    cac_data = raw_data["^FCHI"].dropna()
    data = raw_data[list(portfolio_positions.keys())]
    
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
    
    # Modélisation Quantitative (VaR, CVaR, Volatilité, Bêta vs CAC 40)
    weights_risk = position_values / total_actions
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights_risk)
    
    var_95 = np.percentile(portfolio_returns, 5)
    var_euro = total_actions * var_95
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    cvar_euro = total_actions * cvar_95
    vol_annuelle = portfolio_returns.std() * np.sqrt(252) * 100
    
    # Calcul du Bêta par rapport au CAC 40
    cac_returns = cac_data.pct_change().dropna()
    aligned = pd.concat([portfolio_returns, cac_returns], axis=1).dropna()
    aligned.columns = ['Portefeuille', 'CAC40']
    cov_matrix = np.cov(aligned['Portefeuille'], aligned['CAC40'])
    beta_portefeuille = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # Traduction des tickers en noms d'entreprises
    position_values.index = position_values.index.map(company_names)
    current_prices.index = current_prices.index.map(company_names)
    buy_prices.index = buy_prices.index.map(company_names)
    quantities.index = quantities.index.map(company_names)
    pnl_euro.index = pnl_euro.index.map(company_names)
    pnl_pct.index = pnl_pct.index.map(company_names)
    returns.columns = returns.columns.map(company_names)
    
    # --- BANDEAU DE MÉTRIQUES SUPÉRIEUR ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Valorisation Totale", f"{valeur_totale_portefeuille:,.2f} €", f"{perf_globale_pct:.2f} % ({perf_globale_euro:+.2f} €)")
    col2.metric("Titres Investis", f"{total_actions:,.2f} €", f"{(total_actions/valeur_totale_portefeuille)*100:.1f} % du total")
    col3.metric("Cash Disponible", f"{liquidites_actuelles:,.2f} €", f"{(liquidites_actuelles/valeur_totale_portefeuille)*100:.1f} % du total")
    col4.metric("VaR (95%) 1J", f"{var_euro:,.2f} €", f"{var_95*100:.2f} %", delta_color="inverse")
    col5.metric("CVaR (Krach 5%) 1J", f"{cvar_euro:,.2f} €", f"{cvar_95*100:.2f} %", delta_color="inverse")
    col6.metric("Bêta (vs CAC 40)", f"{beta_portefeuille:.2f}", f"Vol. 1A : {vol_annuelle:.1f} %", delta_color="off")
    
    st.write("---")
    
    # --- NAVIGATION PAR ONGLETS ---
    tab1, tab2, tab3 = st.tabs([
        "🚨 Conformité, PnL & Capacité d'Achat", 
        "📊 Analyse de Risque & Corrélation", 
        "🛠️ Simulateur d'Ordres (Pre-Trade)"
    ])
    
    with tab1:
        st.write("### Suivi des Positions & Limite Réglementaire (Max 15%)")
        
        poids_reels = (position_values / valeur_totale_portefeuille) * 100
        plafond_15_pct = valeur_totale_portefeuille * 0.15
        marge_achat_euro = np.maximum(0, plafond_15_pct - position_values)
        
        # Nombre maximal d'actions achetables (limité par le plafond 15% ET le cash disponible)
        budget_max_ligne = np.minimum(marge_achat_euro, liquidites_actuelles)
        actions_max_achetables = np.floor(budget_max_ligne / current_prices)
        
        df_suivi = pd.DataFrame({
            "Quantité": quantities,
            "PRU (€)": buy_prices,
            "Cours Actuel (€)": current_prices,
            "Capital Actuel (€)": position_values,
            "PnL (€)": pnl_euro,
            "PnL (%)": pnl_pct,
            "Poids (%)": poids_reels,
            "Marge avant 15% (€)": marge_achat_euro,
            "Actions Achetables Max": actions_max_achetables
        }).sort_values(by="Poids (%)", ascending=False)
        
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
            .map(colorer_limites, subset=['Poids (%)'])
            .map(colorer_pnl, subset=['PnL (€)', 'PnL (%)'])
            .format({
                "Quantité": "{:.0f}",
                "PRU (€)": "{:.2f} €",
                "Cours Actuel (€)": "{:.2f} €",
                "Capital Actuel (€)": "{:,.2f} €",
                "PnL (€)": "{:+.2f} €",
                "PnL (%)": "{:+.2f} %",
                "Poids (%)": "{:.2f} %",
                "Marge avant 15% (€)": "{:,.2f} €",
                "Actions Achetables Max": "+{:.0f} titres"
            }),
            use_container_width=True
        )

    with tab2:
        col_chart, col_corr = st.columns(2)
        with col_chart:
            st.write("### Contribution au Risque (Volatilité Annuelle par Actif)")
            vol_indiv = (returns.std() * np.sqrt(252) * 100).sort_values(ascending=False)
            st.bar_chart(vol_indiv)
            
        with col_corr:
            st.write("### Matrice de Corrélation Thermique (1 an)")
            corr_matrix = returns.corr()
            
            def colorer_corr(val):
                if val >= 0.99:
                    return 'color: #888888; background-color: #1e1e1e'
                elif val >= 0.45:
                    return 'color: white; background-color: #8b0000'
                elif val >= 0.25:
                    return 'color: white; background-color: #b22222'
                elif val <= -0.15:
                    return 'color: white; background-color: #0047ab'
                return ''
                
            st.dataframe(
                corr_matrix.style.map(colorer_corr).format("{:.2f}"),
                use_container_width=True
            )
            
        st.write("### Trajectoire Comparée sur 6 Mois : Ton Allocation vs CAC 40 (Base 100)")
        aligned_6m = aligned.tail(126)
        base100 = (1 + aligned_6m).cumprod() * 100
        st.line_chart(base100)

    with tab3:
        st.write("### Testeur d'Ordre Pré-Exécution (Vérification Règle des 15%)")
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            actif_choisi = st.selectbox("Choisis la ligne à renforcer ou alléger :", list(company_names.values()))
            sens_ordre = st.radio("Sens de l'opération :", ["Achat (Renforcement)", "Vente (Allègement)"], horizontal=True)
            qte_sim = st.number_input("Quantité de titres à traiter :", min_value=1, value=10, step=1)
            
        with col_sim2:
            prix_unitaire = current_prices[actif_choisi]
            montant_ordre = qte_sim * prix_unitaire
            val_actuelle_ligne = position_values[actif_choisi]
            
            if "Achat" in sens_ordre:
                nouvelle_val_ligne = val_actuelle_ligne + montant_ordre
                nouveau_cash = liquidites_actuelles - montant_ordre
            else:
                nouvelle_val_ligne = max(0, val_actuelle_ligne - montant_ordre)
                nouveau_cash = liquidites_actuelles + montant_ordre
                
            nouveau_poids = (nouvelle_val_ligne / valeur_totale_portefeuille) * 100
            
            st.metric("Montant de l'ordre estimé", f"{montant_ordre:,.2f} €")
            st.metric("Nouveau Poids de la ligne", f"{nouveau_poids:.2f} %", f"{nouveau_poids - (val_actuelle_ligne/valeur_totale_portefeuille)*100:+.2f} %")
            
            if nouveau_cash < 0:
                st.error(f"❌ Ordre rejeté : Liquidités insuffisantes (Il manque {abs(nouveau_cash):,.2f} €).")
            elif nouveau_poids > 15.0:
                st.error(f"🚨 ALERTE CONFORMITÉ : Cet achat fait passer {actif_choisi} à {nouveau_poids:.2f} % (Plafond autorisé : 15,00 %).")
            else:
                st.success(f"✅ Ordre valide ! Nouveau solde espèces après exécution : {nouveau_cash:,.2f} €.")

except Exception as e:
    st.error(f"Erreur serveur : {e}")
