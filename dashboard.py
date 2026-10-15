import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# optional: try import for normal ppf
try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

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
    data = raw_data[list(portfolio_positions.keys())].ffill()

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

    # Rendements journaliers des actions
    returns = data.pct_change().dropna()

    # Poids
    weights_actions = position_values / total_actions
    weights_total = position_values / valeur_totale_portefeuille  # includes cash implicitly (cash weight = cash/total)

    # Rendements portfolios
    portfolio_returns_actions = returns.dot(weights_actions)
    portfolio_returns_total = returns.dot(weights_total)  # cash return = 0 implicitly

    # Historical VaR/CVaR (1-day) on total portfolio (actions + cash)
    alpha = 0.95
    q_loss = np.percentile(portfolio_returns_total, 100 * (1 - alpha))
    hist_var_pct = -q_loss
    hist_var_euro = hist_var_pct * valeur_totale_portefeuille

    cvar_mask = portfolio_returns_total <= q_loss
    if cvar_mask.sum() > 0:
        hist_cvar_pct = -portfolio_returns_total[cvar_mask].mean()
    else:
        hist_cvar_pct = 0.0
    hist_cvar_euro = hist_cvar_pct * valeur_totale_portefeuille

    # Parametric VaR (variance-covariance) using normal assumption
    mu_assets = returns.mean()
    mu_portfolio = portfolio_returns_total.mean()
    Sigma = returns.cov()  # daily covariance

    # portfolio daily std (including cash as zero-return)
    w = weights_total.values
    Sigma_mat = Sigma.values
    port_var_daily = float(w @ Sigma_mat @ w)
    port_std_daily = np.sqrt(port_var_daily)
    # annualize
    port_std_annual = port_std_daily * np.sqrt(252)

    # get z for alpha
    if _HAS_SCIPY:
        z = -norm.ppf(1 - alpha)
    else:
        # fallback approximate for 95%
        z = 1.6448536269514722 if alpha == 0.95 else 2.3263478740408408

    # parametric VaR pct (positive number)
    param_var_pct = -(mu_portfolio + (-z) * port_std_daily)
    # since z above is positive, use more stable formula: -(mu + z * std)
    param_var_pct = -(mu_portfolio + (-z) * port_std_daily) if False else -(mu_portfolio - z * port_std_daily)
    # Correct formula: VaR = -(mu_portfolio + z_alpha * port_std_daily) where z_alpha is negative quantile
    # Using z_pos = -norm.ppf(1-alpha) so z_pos>0, VaR_pct = -(mu_portfolio - z_pos * port_std_daily)
    z_pos = z
    param_var_pct = -(mu_portfolio - z_pos * port_std_daily)
    param_var_euro = param_var_pct * valeur_totale_portefeuille

    # Parametric CVaR (Gaussian) approximation: cvar = -(mu - (pdf(z)/(1-alpha)) * std) where z = norm.ppf(1-alpha)
    if _HAS_SCIPY:
        z_q = norm.ppf(1 - alpha)
        pdf_z = norm.pdf(z_q)
        param_cvar_pct = -(mu_portfolio + (pdf_z / (1 - alpha)) * port_std_daily)
    else:
        # fallback: approximate parametric CVaR with historical CVaR
        param_cvar_pct = hist_cvar_pct
    param_cvar_euro = param_cvar_pct * valeur_totale_portefeuille

    # Vol annualisée (total portfolio)
    vol_annuelle = port_std_daily * np.sqrt(252) * 100

    # Beta vs CAC40 (use portfolio of actions only)
    cac_returns = cac_data.pct_change().dropna()
    aligned = pd.concat([portfolio_returns_actions, cac_returns], axis=1).dropna()
    aligned.columns = ['Portefeuille', 'CAC40']
    cov_matrix = np.cov(aligned['Portefeuille'], aligned['CAC40'])
    beta_portefeuille = cov_matrix[0, 1] / cov_matrix[1, 1]

    # RSI (Wilder) 14J
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    period = 14
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50
    rsi_14 = rsi.iloc[-1]

    sma_20 = data.rolling(window=20).mean().iloc[-1]
    ecart_sma20 = ((current_prices - sma_20) / sma_20) * 100
    perf_5j = ((current_prices - data.iloc[-6]) / data.iloc[-6]) * 100

    # Risk decomposition: parametric marginal contributions (Euler) and historical contributions
    # Parametric marginal contribution (pct): MC_i = -(w_i * mu_i + z_pos * (Sigma@w)_i / port_std_daily)
    mu_i = mu_assets
    Sigma_w = Sigma_mat @ w
    # avoid divide by zero
    if port_std_daily == 0:
        mc_param_pct = np.zeros_like(w)
    else:
        mc_param_pct = -(w * mu_i.values + z_pos * (Sigma_w / port_std_daily))
    mc_param_euro = mc_param_pct * valeur_totale_portefeuille

    # Historical marginal contributions via tail average
    tail_idx = portfolio_returns_total <= q_loss
    if tail_idx.sum() > 0:
        contribs_assets = returns[tail_idx].multiply(weights_total, axis=1)
        hist_mc_pct = -contribs_assets.mean().values  # positive contributions to loss
    else:
        hist_mc_pct = np.zeros(len(w))
    hist_mc_euro = hist_mc_pct * valeur_totale_portefeuille

    # Volatility contributions (variance decomposition)
    # variance contribution daily: w_i * (Sigma@w)_i
    var_contrib_daily = w * Sigma_w
    total_var_daily = port_var_daily if port_var_daily > 0 else 1.0
    share_var = var_contrib_daily / total_var_daily
    # contribution in euros approximated as share_var * portfolio annual std * portfolio value
    contrib_vol_euro = share_var * (port_std_daily * np.sqrt(252)) * valeur_totale_portefeuille

    # Prepare display series with company names
    assets = list(company_names.values())
    mc_param_ser = pd.Series(mc_param_euro, index=assets)
    hist_mc_ser = pd.Series(hist_mc_euro, index=assets)
    contrib_vol_ser = pd.Series(contrib_vol_euro, index=assets)
    weights_ser = (weights_total * 100).rename(index=company_names)
    vol_indiv = (returns.std() * np.sqrt(252) * 100).rename(index=company_names)

    # --- BANDEAU DE MÉTRIQUES SUPÉRIEUR ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Valorisation Totale", f"{valeur_totale_portefeuille:,.2f} €", f"{perf_globale_pct:.2f} % ({perf_globale_euro:+.2f} €)")
    col2.metric("Titres Investis", f"{total_actions:,.2f} €", f"{(total_actions/valeur_totale_portefeuille)*100:.1f} % du total")
    col3.metric("Cash Disponible", f"{liquidites_actuelles:,.2f} €", f"{(liquidites_actuelles/valeur_totale_portefeuille)*100:.1f} % du total")
    col4.metric("VaR Hist. (95%) 1J", f"{hist_var_euro:,.0f} €", f"{hist_var_pct*100:.2f} %", delta_color="inverse")
    col5.metric("VaR Param. (95%) 1J", f"{param_var_euro:,.0f} €", f"{param_var_pct*100:.2f} %", delta_color="inverse")
    col6.metric("CVaR Hist. (5%) 1J", f"{hist_cvar_euro:,.0f} €", f"{hist_cvar_pct*100:.2f} %", delta_color="inverse")

    st.write("---")

    # --- NAVIGATION PAR ONGLETS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Conformité, PnL & Capacité d'Achat",
        "🎯 Signaux Quantitatifs & Timing",
        "📊 Analyse de Risque & Corrélation",
        "🛠️ Simulateur d'Ordres (Pre-Trade)",
        "📉 Stress & Décomposition du Risque"
    ])

    with tab1:
        st.write("### Suivi des Positions & Limite Réglementaire (Max 15%)")

        poids_reels = (position_values / valeur_totale_portefeuille) * 100
        plafond_15_pct = valeur_totale_portefeuille * 0.15
        marge_achat_euro = np.maximum(0, plafond_15_pct - position_values)

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
        st.write("### Radar de Momentum & Signaux d'Arbitrage sur tes 10 Valeurs")

        signaux = []
        for actif in rsi_14.index:
            r = rsi_14[actif]
            m = ecart_sma20[actif]
            p = poids_reels[actif]
            if p >= 14.0:
                signaux.append("⚠️ Plafond 15% proche (Bloqué)")
            elif r < 35:
                signaux.append("🟢 Survente (Opportunité Rebonds)")
            elif r > 70:
                signaux.append("🟠 Surachat (Prise de profit partielle)")
            elif m > 1.5 and 45 <= r <= 68:
                signaux.append("🚀 Momentum Haussier (Renforcer)")
            elif m < -2.0:
                signaux.append("🔻 Sous Moyenne 20J (Attendre)")
            else:
                signaux.append("⚪ Neutre (Conserver)")

        df_signaux = pd.DataFrame({
            "Cours Actuel (€)": current_prices,
            "Perf 5 Jours (%)": perf_5j,
            "Écart vs MM20 (%)": ecart_sma20,
            "RSI (14 Jours)": rsi_14,
            "Poids Actuel (%)": poids_reels,
            "Signal Tactique": signaux
        }, index=rsi_14.index).sort_values(by="RSI (14 Jours)", ascending=True)

        def colorer_rsi(val):
            if val < 35:
                return 'color: white; background-color: #006400'
            elif val > 70:
                return 'color: black; background-color: #ffa500'
            return ''

        st.dataframe(
            df_signaux.style
            .map(colorer_rsi, subset=['RSI (14 Jours)'])
            .map(colorer_pnl, subset=['Perf 5 Jours (%)', 'Écart vs MM20 (%)'])
            .format({
                "Cours Actuel (€)": "{:.2f} €",
                "Perf 5 Jours (%)": "{:+.2f} %",
                "Écart vs MM20 (%)": "{:+.2f} %",
                "RSI (14 Jours)": "{:.1f}",
                "Poids Actuel (%)": "{:.2f} %"
            }),
            use_container_width=True
        )

    with tab3:
        col_chart, col_corr = st.columns(2)
        with col_chart:
            st.write("### Contribution au Risque (Volatilité Annuelle par Actif)")
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

    with tab4:
        st.write("### Testeur d'Ordre Pré-Exécution (Vérification Règle des 15% & Nouveau PRU)")
        col_sim1, col_sim2 = st.columns(2)

        with col_sim1:
            actif_choisi = st.selectbox(
                "Choisis la ligne à renforcer ou alléger :",
                list(portfolio_positions.keys()),
                format_func=lambda ticker: company_names.get(ticker, ticker)
            )
            sens_ordre = st.radio("Sens de l'opération :", ["Achat (Renforcement)", "Vente (Allègement)"], horizontal=True)
            qte_sim = st.number_input("Quantité de titres à traiter :", min_value=1, value=10, step=1)

        with col_sim2:
            prix_unitaire = current_prices[actif_choisi]
            qte_actuelle = quantities[actif_choisi]
            pru_actuel = buy_prices[actif_choisi]
            montant_ordre = qte_sim * prix_unitaire
            val_actuelle_ligne = position_values[actif_choisi]

            if "Achat" in sens_ordre:
                nouvelle_qte = qte_actuelle + qte_sim
                nouvelle_val_ligne = val_actuelle_ligne + montant_ordre
                nouveau_cash = liquidites_actuelles - montant_ordre
                if nouvelle_qte > 0:
                    nouveau_pru = ((qte_actuelle * pru_actuel) + montant_ordre) / nouvelle_qte
                else:
                    nouveau_pru = pru_actuel
            else:
                nouvelle_qte = max(0, qte_actuelle - qte_sim)
                nouvelle_val_ligne = max(0, val_actuelle_ligne - montant_ordre)
                nouveau_cash = liquidites_actuelles + montant_ordre
                nouveau_pru = pru_actuel

            nouvelle_valeur_portefeuille = valeur_totale_portefeuille + (nouvelle_val_ligne - val_actuelle_ligne) - (liquidites_actuelles - nouveau_cash)
            if nouvelle_valeur_portefeuille == 0:
                nouveau_poids = 0.0
            else:
                nouveau_poids = (nouvelle_val_ligne / nouvelle_valeur_portefeuille) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Montant de l'ordre", f"{montant_ordre:,.2f} €")
            c2.metric("Nouveau Poids", f"{nouveau_poids:.2f} %", f"{nouveau_poids - (val_actuelle_ligne/valeur_totale_portefeuille)*100:+.2f} %")
            c3.metric("Nouveau PRU Moyen", f"{nouveau_pru:.2f} €", f"{nouvelle_qte:.0f} titres au total", delta_color="off")

            if nouveau_cash < 0:
                st.error(f"❌ Ordre rejeté : Liquidités insuffisantes (Il manque {abs(nouveau_cash):,.2f} €).")
            elif nouveau_poids > 15.0:
                st.error(f"🚨 ALERTE CONFORMITÉ : Cet achat fait passer {company_names.get(actif_choisi, actif_choisi)} à {nouveau_poids:.2f} % (Plafond autorisé : 15,00 %).")
            else:
                st.success(f"✅ Ordre valide ! Nouveau solde espèces après exécution : {nouveau_cash:,.2f} €.")

    with tab5:
        st.write("### Stress Tests & Décomposition du Risque")
        st.write("Scénarios simples (impact direct sur cours des actions)")
        shocks = st.slider("Choisis un choc de marché (uniform pour tous les titres) :", -50, 0, -10, step=5)
        shocks_pct = shocks / 100.0

        # Scenario impact
        shocked_values = position_values * (1 + shocks_pct)
        impact_euro = position_values.sum() - shocked_values.sum()
        impact_pct = (impact_euro / valeur_totale_portefeuille) * 100
        st.metric(f"Impact du choc {shocks}% sur la valeur", f"-{impact_euro:,.2f} €", f"-{impact_pct:.2f} %")

        st.write("---")
        st.write("### Décomposition du risque (contributions à la VaR)")
        df_contrib = pd.DataFrame({
            "Poids (%)": weights_ser,
            "Vol. Ann. (%)": vol_indiv,
            "Contrib VaR Param (€)": mc_param_ser,
            "Contrib VaR Hist (€)": hist_mc_ser,
            "Contrib Vol (€ approx.)": contrib_vol_ser
        })
        df_contrib = df_contrib.sort_values(by="Contrib VaR Param (€)", ascending=False)
        st.dataframe(df_contrib.style.format({
            "Poids (%)": "{:.2f} %",
            "Vol. Ann. (%)": "{:.2f} %",
            "Contrib VaR Param (€)": "{:,.0f} €",
            "Contrib VaR Hist (€)": "{:,.0f} €",
            "Contrib Vol (€ approx.)": "{:,.0f} €"
        }), use_container_width=True)

        st.write("### Graphique : Contributions à la VaR (paramétrique vs historique)")
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(mc_param_ser.sort_values(ascending=False))
        with c2:
            st.bar_chart(hist_mc_ser.sort_values(ascending=False))

        st.write("---")
        st.write("### Détails supplémentaires : Parametric vs Historical VaR/CVaR")
        st.write(f"VaR historique (95%) : {hist_var_euro:,.2f} € ({hist_var_pct*100:.2f} %)")
        st.write(f"CVaR historique (5%) : {hist_cvar_euro:,.2f} € ({hist_cvar_pct*100:.2f} %)")
        st.write(f"VaR paramétrique (95%) : {param_var_euro:,.2f} € ({param_var_pct*100:.2f} %)")
        st.write(f"CVaR paramétrique (5%) : {param_cvar_euro:,.2f} € ({param_cvar_pct*100:.2f} %)")

except Exception as e:
    st.error(f"Erreur serveur : {e}")

