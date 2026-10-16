import json
from hashlib import sha256
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from portfolio_config import RULES_URL, RULES_CHECKED, SECTORS, validate_holdings
from portfolio_math import allocation, market_panels, risk_metrics, max_buy_quantity, simulate_order, wilder_rsi
from portfolio_storage import sync_portfolio, backup_controls, reference_portfolio, is_legacy_portfolio

st.set_page_config(page_title="NextChallenge Solo · Pilotage", page_icon="📊", layout="wide")


def euro(value):
    return f"{value:,.2f} €".replace(",", " ") if np.isfinite(value) else "Non disponible"


def number(value, digits=2):
    return f"{value:.{digits}f}" if np.isfinite(value) else "Non disponible"


def display_frame(frame, formats=None):
    st.dataframe(frame.style.format(formats or {}, na_rep="—"), width="stretch")


@st.cache_data(ttl=300, show_spinner=False)
def load_data(tickers):
    result = yf.download(list(tickers), period="1y", interval="1d", auto_adjust=False,
                         progress=False, timeout=15, group_by="column")
    return result, datetime.now(ZoneInfo("Europe/Paris")).isoformat()


st.title("NextChallenge Solo")
st.caption("Mes positions → les cours actuels → mes décisions · Interface simplifiée du 16 octobre")
with st.sidebar:
    st.header("Sauvegarde")
    sync_portfolio()
    # Cash is now account data, not an immediately saved sidebar widget.
    st.session_state.cash = float(st.session_state.cash)
    with st.expander("Réglages avancés"):
        capital = st.number_input("Capital de référence (€)", min_value=0.01,
                              step=100.0, key="capital", help="Apport initial affiché par NextWise : ta capture indique 50 959,17 €.")
        rf_pct = st.number_input("Taux de référence du Sharpe (%/an)", min_value=0.0, max_value=20.0,
                            step=0.25, key="rf", help="Hypothèse, pas un taux récupéré ni le paramètre officiel du concours. Cash supposé sans rémunération.")
    with st.expander("Sauvegarde et transfert"):
        backup_controls()
    st.link_button("Règlement officiel", RULES_URL, width="stretch")
    st.caption(f"Consulté le {RULES_CHECKED} · Fin annoncée : 9 avril 2027.")

reference = reference_portfolio()
legacy = is_legacy_portfolio(st.session_state, reference)
if legacy:
    st.warning("Tu vois encore l’ancienne base : 980 Stellantis et 8 662,64 € de cash. Ton fichier du 16 octobre indique 480 Stellantis et 10 654,39 € de cash.")
    if st.button("Charger mes chiffres du 16 octobre", type="primary", width="stretch"):
        st.session_state.pending_restore = dict(holdings=reference["holdings"], cash=reference["cash"])
        st.rerun()

st.subheader("Gérer mon portefeuille")
st.write("**Après un achat ou une vente exécuté sur NextWise :** ouvre le formulaire ci-dessous, corrige les quantités et le cash, puis enregistre.")
with st.expander("Modifier mon portefeuille — quantités et cash"):
    st.caption("Recopie les chiffres de NextWise. Les cours de marché et la valeur totale se recalculent ensuite ; tu ne les saisis pas.")
    # Reset the draft only when the saved account changes (import, edit or restore).
    editor_version = sha256((st.session_state.holdings.to_json() + str(st.session_state.cash)).encode()).hexdigest()[:12]
    with st.form("edit_portfolio"):
        draft_cash = st.number_input("Liquidités actuelles sur NextWise (€)", min_value=0.0,
                                     value=float(st.session_state.cash), step=1.0,
                                     key=f"edit_cash_{editor_version}")
        edited = st.data_editor(st.session_state.holdings, num_rows="dynamic", hide_index=True,
                                key=f"edit_holdings_{editor_version}",
                                column_config={"Secteur": st.column_config.SelectboxColumn(options=SECTORS, required=True),
                                               "Quantité": st.column_config.NumberColumn(min_value=0, step=1, required=True),
                                               "PRU (€)": st.column_config.NumberColumn(min_value=0, required=True)}, width="stretch")
        apply_changes = st.form_submit_button("Enregistrer mon portefeuille", type="primary", width="stretch")
    if apply_changes:
        try:
            updated = validate_holdings(edited)
            if not np.isfinite(draft_cash):
                raise ValueError("Le cash doit être un nombre valide.")
            st.session_state.pending_restore = dict(holdings=updated, cash=float(draft_cash))
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.caption("Attends « Enregistré dans ce navigateur » avant de fermer. Vérifie l’éligibilité NextWise et la cotation en EUR pour toute nouvelle valeur.")

st.button("Actualiser les cours", on_click=load_data.clear, width="stretch")
st.caption("Pour consulter le marché : ce bouton récupère les prix Yahoo sans changer tes quantités ni ton cash. Il n’importe aucun ordre NextWise. Cours en cache 5 minutes.")
cash = st.session_state.cash

holdings = validate_holdings(st.session_state.holdings).set_index("Code Yahoo")
names, sectors = holdings["Nom"].to_dict(), holdings["Secteur"].to_dict()
quantities, buy_prices = holdings["Quantité"], holdings["PRU (€)"]
tickers = tuple(holdings.index)
label = lambda ticker: f"{names.get(ticker, ticker)} ({ticker})"

try:
    with st.spinner("Chargement des cours et de l’historique…"):
        downloaded, fetched_at = load_data(tickers + ("^FCHI",))
        prices, quote_dates, adjusted, returns = market_panels(downloaded, tickers)
    book = allocation(prices, quantities, cash, sectors)
except Exception as exc:
    st.error(f"Données indisponibles : {exc}")
    st.info("Aucune position n’est exclue du total pour masquer un cours manquant. Réessaie avec Actualiser les cours.")
    st.info("Pour corriger une ligne, ouvre « Modifier mon portefeuille — quantités et cash » en haut de la page. Ta sauvegarde reste disponible à gauche.")
    st.stop()

now = datetime.now(ZoneInfo("Europe/Paris"))
quote_age = quote_dates.map(lambda timestamp: (now.date() - pd.Timestamp(timestamp).date()).days)
stale = quote_age[quote_age > 7]
total, values, weights = book["total"], book["values"], book["weights"]
pnl = values - quantities * buy_prices
performance = total / capital - 1
metrics = st.columns(4)
metrics[0].metric("Valorisation estimée", euro(total), f"{performance:+.2%} vs capital de référence",
                  help="Derniers cours Yahoo + cash saisi. Ce n’est pas la valorisation officielle NextWise.")
metrics[1].metric("Liquidités", euro(cash), f"{cash / total:.1%} du portefeuille", delta_color="off")
metrics[2].metric("Valeurs détenues", book["n_lines"], "Minimum solo : 10", delta_color="off")
metrics[3].metric("Plus grande ligne", f"{weights.max():.2%}", names[weights.idxmax()], delta_color="off")
st.caption(f"Récupération : {datetime.fromisoformat(fetched_at).strftime('%d/%m/%Y à %H:%M')} (Paris) · "
           f"Dernières observations : {pd.Timestamp(quote_dates.min()).strftime('%d/%m/%Y')} – {pd.Timestamp(quote_dates.max()).strftime('%d/%m/%Y')}. "
           "Une barre quotidienne peut être en cours de séance ; aucun cours d’exécution n’est garanti.")
if len(stale):
    st.warning("Cours de plus de 7 jours : " + ", ".join(stale.index) + ". Simulations suspendues jusqu’à réception de données récentes.")
try:
    risk = risk_metrics(returns, weights, total, rf_pct / 100)
except ValueError as exc:
    risk = None
    st.warning(f"Analyse de risque indisponible : {exc}")

overview, positions_tab, signals_tab, risk_tab, simulation_tab, journal_tab = st.tabs([
    "🧭 Pilotage", "💼 Positions", "🔎 Signaux", "📊 Risque", "🧪 Simulateur", "📝 Journal & concours"
])

with overview:
    st.subheader("À vérifier avant de décider")
    if book["violations"]:
        for violation in book["violations"]:
            st.warning(violation)
    else:
        st.success("Les trois repères solo sont respectés avec les données et secteurs saisis.")
    if book["n_lines"] == 10:
        st.info("Tu détiens exactement 10 valeurs : solder une ligne ferait passer ce nombre à 9. Compare l’ensemble de la réallocation avant de valider tes ordres.")
    if weights.max() >= 0.13:
        st.info(f"{names[weights.idxmax()]} approche du plafond : {weights.max():.2%}. Une hausse du cours peut aussi augmenter ce poids.")
    st.caption("Secteurs indicatifs à rapprocher de NextWise ; l’ETF diversifié n’ajoute pas un secteur au compteur. Le classement officiel n’est pas recalculé ici.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Secteurs identifiés", book["n_sectors"], "Minimum solo : 4", delta_color="off")
    c2.metric("HHI des titres", number(book["hhi"] * 10000, digits=0),
              help="Somme des poids au carré, normalisés sur les titres, × 10 000. Dix lignes égales = 1 000. Cash exclu ; la convention officielle peut différer.")
    c3.metric("Nombre effectif de lignes", number(book["effective_lines"], digits=1),
              help="Inverse de la somme des poids normalisés au carré. Mesure la concentration par ligne, sans détailler les constituants de l’ETF.")
    if risk:
        c1, c2, c3 = st.columns(3)
        c1.metric("Volatilité annualisée estimée", f"{risk['vol']:.2%}", help="Variabilité de l’allocation constante sur l’historique disponible, cash inclus.")
        c2.metric("Perte moyenne en queue · CVaR 95 %", euro(risk["hist_es"]), help="Moyenne des pertes de la queue historique défavorable. Ce n’est pas une perte maximale.")
        c3.metric("Ratio de diversification", number(risk["diversification"]), help="Moyenne pondérée des volatilités individuelles / volatilité du portefeuille.")
        largest = risk["es_components"].idxmax()
        st.write(f"**À examiner :** {names[largest]} contribue le plus à la perte moyenne de queue simulée ({euro(risk['es_components'][largest])}). Ce classement ne prédit pas la prochaine baisse.")
    left, right = st.columns(2)
    with left:
        st.write("**Répartition par ligne (% du total)**")
        allocation_chart = (weights * 100).rename(index=label)
        allocation_chart.loc["Liquidités"] = cash / total * 100
        st.bar_chart(allocation_chart.sort_values(ascending=False))
    with right:
        st.write("**Répartition par secteur (% du total)**")
        st.bar_chart(book["sector_weights"] * 100)
        st.caption("ETF affiché séparément : pas de détail de ses constituants ni de leur recouvrement avec tes actions.")
    with st.expander("Comment utiliser cette page"):
        st.write("1. Vérifie positions et cash. 2. Repère les concentrations. 3. Compare l’ordre dans le simulateur. 4. Écris ta thèse et ce qui te ferait changer d’avis.")
        st.write("Une faible VaR, un Sharpe élevé ou davantage de lignes ne suffisent pas à sélectionner un investissement. Ces repères rendent les compromis visibles.")

with positions_tab:
    st.subheader("Positions & capacité d’achat")
    max_qty = pd.Series({t: max_buy_quantity(prices, quantities, cash, t) for t in tickers})
    follow = pd.DataFrame({"Valeur": [names[t] for t in tickers], "Secteur": [sectors[t] for t in tickers],
                           "Quantité": quantities, "PRU (€)": buy_prices, "Cours Yahoo (€)": prices,
                           "Capital (€)": values, "PnL latent (€)": pnl,
                           "Poids (%)": weights * 100, "Achat max (titres)": max_qty,
                           "Date cours": quote_dates.map(lambda d: pd.Timestamp(d).strftime('%d/%m/%Y'))})
    display_frame(follow.sort_values("Poids (%)", ascending=False), {"Quantité": "{:.0f}", "PRU (€)": "{:.2f}",
                  "Cours Yahoo (€)": "{:.2f}", "Capital (€)": "{:,.2f}", "PnL latent (€)": "{:+,.2f}",
                  "Poids (%)": "{:.2f}", "Achat max (titres)": "{:.0f}"})
    st.caption("Achat max : cash et plafond de 15 % pour cette seule ligne. Ces capacités ne s’additionnent pas : chaque achat utilise le même cash. PnL latent hors gains/pertes réalisés.")
    st.caption("Pour modifier tes positions et le cash : utilise « Modifier mon portefeuille — quantités et cash » tout en haut de la page.")

with signals_tab:
    st.subheader("Lecture technique · constats, pas ordres d’achat")
    rsi = wilder_rsi(adjusted).iloc[-1]
    latest = adjusted.iloc[-1]
    sma20 = adjusted.rolling(20).mean().iloc[-1]
    sma50 = adjusted.rolling(50).mean().iloc[-1]
    sma200 = adjusted.rolling(200).mean().iloc[-1]
    perf5 = adjusted.pct_change(5, fill_method=None).iloc[-1] * 100
    perf20 = adjusted.pct_change(20, fill_method=None).iloc[-1] * 100
    readings = []
    for ticker in tickers:
        if not np.isfinite(rsi[ticker]):
            reading = "Historique insuffisant"
        elif rsi[ticker] < 35:
            reading = "RSI bas : faiblesse récente, rebond non confirmé"
        elif rsi[ticker] > 70:
            reading = "RSI élevé : hausse soutenue, repli possible"
        else:
            reading = "RSI intermédiaire : pas de conclusion isolée"
        readings.append(reading)
    signal_frame = pd.DataFrame({"Valeur": [names[t] for t in tickers], "RSI 14J": rsi,
                                "Perf. 5 séances (%)": perf5, "Perf. 20 séances (%)": perf20,
                                "Écart MM20 (%)": (latest / sma20 - 1) * 100,
                                "Écart MM50 (%)": (latest / sma50 - 1) * 100,
                                "Écart MM200 (%)": (latest / sma200 - 1) * 100,
                                "Lecture": readings})
    display_frame(signal_frame, {c: "{:+.2f}" for c in signal_frame if c not in ["Valeur", "Lecture"]})
    st.info("Confronte le mouvement aux informations d’entreprise, au calendrier des résultats et à ta thèse. Une survente peut durer ; une tendance forte peut rester en surachat.")
    st.caption("Cours ajustés ; RSI de Wilder amorcé par une moyenne simple sur 14 variations. Un historique manquant ne devient pas un signal de survente.")

with risk_tab:
    st.subheader("Risque de l’allocation actuelle")
    st.info("Simulation historique à poids constants, cash inclus à rendement nul. Les graphiques et le Sharpe ci-dessous ne sont ni ton historique réel de transactions ni ton score NextWise.")
    if risk is None:
        st.write("Ces indicateurs nécessitent un historique commun suffisant.")
    else:
        st.caption(f"{risk['n']} rendements communs · {risk['tail_n']} observations dans la queue à 5 % · {returns.index.min().date()} – {returns.index.max().date()}.")
        if risk["n"] < 100:
            st.warning("Échantillon court : mesures de queue particulièrement instables.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VaR historique · 95 % · 1 séance", euro(risk["hist_var"]))
        c2.metric("CVaR historique · 95 % · 1 séance", euro(risk["hist_es"]))
        c3.metric("VaR normale · 95 % · 1 séance", euro(risk["param_var"]))
        c4.metric("CVaR normale · 95 % · 1 séance", euro(risk["param_es"]))
        c1, c2 = st.columns(2)
        c1.metric("Sharpe de l’allocation simulée", number(risk["sharpe"]), help="Excès de rendement journalier / volatilité × √252. Le taux de référence est paramétrable à gauche.")
        c2.metric("Repli maximal simulé", f"{risk['max_drawdown']:.2%}", help="Plus forte baisse depuis un sommet, capital initial inclus.")
        with st.expander("VaR, CVaR et Sharpe en langage simple"):
            st.write("La VaR à 95 % est un seuil de perte dépassé dans les 5 % de scénarios les plus défavorables du modèle. La CVaR est la perte moyenne de cette queue. Aucune n’est une perte maximale.")
            st.write("La méthode historique reprend les observations passées. La méthode normale suppose une distribution gaussienne et peut sous-estimer les extrêmes. Le Sharpe rapporte l’excès de rendement à la volatilité ; son niveau passé ne garantit rien pour la suite.")
            st.write("Quantiles conservés avec leur signe : une estimation négative correspond à un gain à ce quantile. Les conventions de calcul ici ne reproduisent pas nécessairement celles de NextWise.")
        contrib = pd.DataFrame({"Poids (%)": weights * 100, "Vol. individuelle (%/an)": risk["individual_vol"] * 100,
                                "Contribution VaR normale (€)": risk["var_components"],
                                "Contribution CVaR historique (€)": risk["es_components"],
                                "Part de variance (%)": risk["vol_shares"] * 100}).rename(index=label)
        st.write("**Quelle ligne contribue au risque du portefeuille ?**")
        display_frame(contrib.sort_values("Contribution CVaR historique (€)", ascending=False), "{:.2f}")
        st.caption("Les contributions totalisent leur mesure respective. Une contribution négative traduit une compensation dans cet échantillon. La volatilité individuelle seule n’est pas une contribution au risque.")
        st.bar_chart(contrib[["Contribution VaR normale (€)", "Contribution CVaR historique (€)"]])
        left, right = st.columns(2)
        with left:
            st.write("**Trajectoire simulée, base 100**")
            chart = pd.DataFrame({"Allocation actuelle + cash": risk["wealth"] * 100})
            benchmark = None
            if "^FCHI" in downloaded["Close"]:
                benchmark = downloaded["Close"]["^FCHI"].pct_change(fill_method=None).reindex(chart.index)
                if benchmark.notna().all():
                    chart["CAC 40 prix (hors dividendes)"] = (1 + benchmark).cumprod() * 100
            chart.loc[chart.index[0] - pd.offsets.BDay(1)] = 100
            st.line_chart(chart.sort_index())
            st.caption("Comparaison illustrative : allocation ajustée des dividendes face au CAC 40 prix. Ne mesure pas une surperformance homogène. Poids rééquilibrés chaque séance, sans frais.")
            if benchmark is not None:
                pair = pd.concat([risk["returns"].rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
                if len(pair) >= 20 and pair["benchmark"].var() > 0:
                    beta = pair.cov().loc["portfolio", "benchmark"] / pair["benchmark"].var()
                    st.caption(f"Bêta vs CAC 40 : {beta:.2f}, cash inclus. Sensibilité historique, pas une prévision.")
        with right:
            st.write("**Replis depuis un sommet (%)**")
            st.area_chart(risk["drawdown"] * 100)
        with st.expander("Corrélations et recouvrements"):
            display_frame(returns.corr().rename(index=label, columns=label), "{:.2f}")
            pairs = [(a, b, returns[a].corr(returns[b])) for i, a in enumerate(tickers) for b in tickers[i + 1:] if quantities[a] > 0 and quantities[b] > 0]
            high = sorted([p for p in pairs if np.isfinite(p[2]) and p[2] >= 0.75], key=lambda x: -x[2])[:5]
            for a, b, corr in high:
                st.write(f"{names[a]} / {names[b]} : {corr:.2f}. Vérifie si ces lignes apportent des expositions différentes.")
            if not high:
                st.write("Aucune paire détenue au-dessus de 0,75. Les corrélations peuvent changer en période de stress.")
    st.write("**Stress déterministe**")
    shock = st.slider("Baisse hypothétique de tous les titres (%)", -50, 0, -10, step=1)
    impact = book["invested"] * shock / 100
    c1, c2 = st.columns(2)
    c1.metric("Impact du scénario", euro(impact), f"{impact / total:.2%}", delta_color="off")
    c2.metric("Valeur après ce scénario", euro(total + impact))
    st.caption("Même choc sur chaque titre ; cash inchangé. Aucune probabilité n’est associée à ce scénario.")

with simulation_tab:
    st.subheader("Comparer un ordre avant de le saisir sur NextWise")
    st.caption("Simulation uniquement, sans transmission d’ordre. Frais nuls selon l’article 4 ; cours d’exécution à vérifier sur NextWise.")
    left, right = st.columns([1, 2])
    with left:
        chosen = st.selectbox("Valeur à simuler", list(tickers), format_func=label, key="sim_asset")
        side = st.radio("Sens", ["Achat", "Vente"], horizontal=True, key="sim_side")
        quantity = st.number_input("Quantité à simuler", min_value=1, value=1, step=1, key="sim_quantity")
        execution = st.number_input("Cours de simulation (€)", min_value=0.0001, value=float(prices[chosen]),
                                    step=0.01, format="%.4f", key=f"execution_{chosen}")
        local_prices = prices.copy()
        local_prices[chosen] = execution
        st.write(f"Titres détenus : **{quantities[chosen]:.0f}**")
        st.write(f"Achat max cash / 15 % : **{max_buy_quantity(local_prices, quantities, cash, chosen)} titres**")
        st.caption("Le cours saisi valorise cette ligne avant et après l’ordre. Le capital reste identique avant/après, hors changement d’hypothèse de cours.")
    with right:
        if len(stale):
            st.warning("Simulation suspendue : cours trop ancien. Actualise les données.")
        else:
            try:
                result = simulate_order(prices, quantities, buy_prices, cash, chosen, side, quantity, execution, sectors)
            except ValueError as exc:
                st.error(str(exc))
            else:
                before, after = result["before"], result["after"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Montant simulé", euro(result["amount"]))
                c2.metric("Cash après", euro(result["cash"]))
                c3.metric("PRU après", euro(result["pru"]))
                comparison = pd.DataFrame({"Avant": [before["weights"][chosen] * 100, before["n_lines"], before["n_sectors"], before["hhi"] * 10000],
                                           "Après": [after["weights"][chosen] * 100, after["n_lines"], after["n_sectors"], after["hhi"] * 10000]},
                                          index=["Poids de la ligne (%)", "Valeurs détenues", "Secteurs identifiés", "HHI titres"])
                if risk is not None:
                    risk_before = risk_metrics(returns, before["weights"], before["total"], rf_pct / 100)
                    risk_after = risk_metrics(returns, after["weights"], after["total"], rf_pct / 100)
                    for title, key, factor in [("Volatilité (%/an)", "vol", 100), ("VaR historique 95 % (€)", "hist_var", 1),
                                               ("CVaR historique 95 % (€)", "hist_es", 1), ("Ratio de diversification", "diversification", 1)]:
                        comparison.loc[title] = [risk_before[key] * factor, risk_after[key] * factor]
                comparison["Variation"] = comparison["Après"] - comparison["Avant"]
                display_frame(comparison, {"Avant": "{:.2f}", "Après": "{:.2f}", "Variation": "{:+.2f}"})
                if after["violations"]:
                    for violation in after["violations"]:
                        st.warning(violation)
                else:
                    st.success("Les trois repères solo restent respectés après simulation, selon les données saisies.")
                st.caption("Un risque estimé plus faible ne prouve pas que le rendement ou le classement s’améliorera. Confronte aussi l’ordre à ta thèse et aux données NextWise.")

with journal_tab:
    st.subheader("Préparer une décision que tu peux expliquer")
    st.caption("Notes personnelles de session, exportables. Aucune publication ou interaction automatique sur NextWise.")
    with st.form("decision_journal", clear_on_submit=True):
        journal_asset = st.selectbox("Valeur étudiée", list(tickers), format_func=label)
        journal_action = st.selectbox("Décision envisagée", ["Observer", "Conserver", "Acheter / renforcer", "Vendre / alléger"])
        thesis = st.text_area("Thèse : pourquoi cette décision ?", max_chars=3000)
        catalyst = st.text_input("Catalyseur et horizon", max_chars=500)
        invalidation = st.text_area("Qu’est-ce qui invaliderait ma thèse ?", max_chars=1500)
        sources = st.text_area("Sources consultées et dates", placeholder="Résultats, communiqué, document d’entreprise…", max_chars=2000)
        ai_note = st.text_input("Aide IA utilisée, le cas échéant", max_chars=500)
        if st.form_submit_button("Ajouter à mon journal"):
            if not thesis.strip() or not invalidation.strip() or not sources.strip():
                st.warning("Complète la thèse, son invalidation et les sources.")
            else:
                st.session_state.journal.append({"Date": now.isoformat(timespec="minutes"), "Valeur": label(journal_asset),
                                                 "Décision": journal_action, "Thèse": thesis, "Catalyseur / horizon": catalyst,
                                                 "Invalidation": invalidation, "Sources": sources, "Aide IA": ai_note})
                st.rerun()
    if st.session_state.journal:
        st.dataframe(pd.DataFrame(st.session_state.journal), hide_index=True, width="stretch")
        st.download_button("Exporter le journal en JSON", json.dumps(st.session_state.journal, ensure_ascii=False, indent=2),
                           file_name="nextchallenge_journal.json", mime="application/json")
    st.write("**Rendez-vous de travail sur NextWise**")
    st.caption("Repères personnels pour ta revue : ni points acquis ni synchronisation avec la plateforme.")
    st.checkbox("Consulter les informations et le dashboard NextWise", key="review_info")
    st.checkbox("Faire les quiz disponibles", key="review_quiz")
    st.checkbox("Préparer une contribution argumentée pour le chat individuel", key="review_chat")
    st.checkbox("Réexaminer les thèses après les nouvelles d’entreprise", key="review_thesis")
    st.write(f"[Règles et critères officiels]({RULES_URL}) · Rendement, risque, diversification et participation sont évalués. Les pondérations détaillées du barème n’ont pas été identifiées dans la page consultée.")
    with st.expander("Méthode et limites"):
        st.write("Positions et cash sont saisis manuellement. Les prix bruts servent à la valorisation ; les historiques ajustés servent au risque et aux signaux. Le portefeuille ne se synchronise pas après un ordre NextWise.")
        st.write("Sans historique quotidien réel de valorisation et de flux, ton Sharpe réalisé au concours ne peut pas être calculé ici. Le parcours affiché simule l’allocation actuelle rééquilibrée chaque jour, sans frais, sur l’historique commun disponible.")
        st.write("Secteurs indicatifs, sans détail des constituants de l’ETF. Les contrôles portent sur le mode solo ; la règle 5/10/40 du règlement concerne les clubs.")
        st.markdown("Sources techniques : [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), "
                    "[Sharpe](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm), "
                    "[allocation du risque](https://arxiv.org/abs/0708.2542), "
                    "[ratio de diversification](https://www.tobam.fr/wp-content/docs/GROUND_RULES/en/TOBAM%20Maximum%20Diversification%20Equity%20Index%20Series%20Ground%20Rules_March2022.pdf).")
