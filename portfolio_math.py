"""Calculs indépendants de l'interface, en euros et rendements décimaux."""
from statistics import NormalDist

import numpy as np
import pandas as pd


def validate_positions(prices, quantities, cash):
    prices = pd.Series(prices, dtype=float)
    quantities = pd.Series(quantities, dtype=float)
    if not prices.index.is_unique or not quantities.index.is_unique:
        raise ValueError("Codes de valeurs dupliqués.")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Les liquidités doivent être positives ou nulles.")
    if not np.isfinite(quantities).all() or (quantities < 0).any() or (quantities % 1 != 0).any():
        raise ValueError("Les quantités doivent être des nombres entiers positifs ou nuls.")
    prices = prices.reindex(quantities.index)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Un cours est manquant, non fini ou non positif : valorisation interrompue.")
    values = prices * quantities
    total = float(values.sum() + cash)
    if total <= 0:
        raise ValueError("La valeur totale doit être strictement positive.")
    return prices, quantities, values, total


def allocation(prices, quantities, cash, sectors):
    prices, quantities, values, total = validate_positions(prices, quantities, cash)
    active = quantities > 0
    weights = values / total
    invested = float(values.sum())
    equity_weights = values / invested if invested > 0 else values * 0
    hhi = float((equity_weights ** 2).sum()) if invested > 0 else np.nan
    labels = pd.Series(sectors).reindex(quantities.index).fillna("Non classé")
    # Un ETF multisection n'est pas compté comme un secteur supplémentaire.
    known = labels[active & ~labels.isin(["ETF diversifié", "Non classé"])]
    sector_weights = weights.groupby(labels).sum().sort_values(ascending=False)
    violations = []
    if int(active.sum()) < 10:
        violations.append(f"{int(active.sum())} valeurs détenues : minimum solo de 10.")
    if known.nunique() < 4:
        violations.append(f"{known.nunique()} secteurs identifiés : minimum solo de 4.")
    over = weights[weights > 0.15 + 1e-10]
    for ticker, weight in over.items():
        violations.append(f"{ticker} pèse {weight:.2%} : plafond solo de 15 %.")
    return dict(values=values, total=total, invested=invested, weights=weights,
                equity_weights=equity_weights, hhi=hhi,
                effective_lines=1 / hhi if hhi > 0 else np.nan,
                n_lines=int(active.sum()), n_sectors=int(known.nunique()),
                sector_weights=sector_weights, violations=violations)


def market_panels(downloaded, tickers):
    """Ne jamais supprimer silencieusement une position dont le cours manque."""
    if downloaded is None or downloaded.empty:
        raise ValueError("Yahoo Finance n'a renvoyé aucun cours. Réessaie avec Actualiser les cours.")
    if not isinstance(downloaded.columns, pd.MultiIndex):
        raise ValueError("Format de données de marché inattendu.")
    fields = downloaded.columns.get_level_values(0)
    if "Close" not in fields or "Adj Close" not in fields:
        raise ValueError("Cours bruts ou historiques ajustés indisponibles.")
    close = downloaded["Close"].sort_index().reindex(columns=tickers)
    adjusted = downloaded["Adj Close"].sort_index().reindex(columns=tickers)
    close = close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    adjusted = adjusted.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    close = close.where(close > 0)
    adjusted = adjusted.where(adjusted > 0)
    missing = [ticker for ticker in tickers if close[ticker].dropna().empty]
    if missing:
        raise ValueError("Cours manquants pour " + ", ".join(missing) + ". Aucun total incomplet n'est affiché.")
    prices = close.apply(lambda series: series.dropna().iloc[-1])
    dates = close.apply(lambda series: series.last_valid_index())
    # Sans remplissage implicite : une absence de cotation n'est pas un rendement nul.
    returns = adjusted.pct_change(fill_method=None).dropna(how="any")
    return prices, dates, adjusted, returns


def risk_metrics(returns, weights, total, rf_annual=0.0, confidence=0.95):
    """Allocation actuelle à poids constants, cash supposé sans rendement.

    Les contributions Euler totalisent la VaR paramétrique ; les contributions
    de queue totalisent la CVaR historique (et non la VaR historique).
    """
    if not 0 < confidence < 1 or rf_annual <= -1:
        raise ValueError("Hypothèses de risque invalides.")
    weights = pd.Series(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() > 1 + 1e-8:
        raise ValueError("Poids invalides pour un portefeuille sans levier.")
    absent = weights[(weights > 0) & ~weights.index.isin(returns.columns)]
    if len(absent):
        raise ValueError("Historique absent : " + ", ".join(absent.index))
    sample = returns.reindex(columns=weights.index).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 20:
        raise ValueError(f"Historique commun insuffisant : {len(sample)} séances, minimum technique de 20.")
    w = weights.to_numpy()
    mu = sample.mean().to_numpy()
    covariance = sample.cov().to_numpy()
    sigma_w = covariance @ w
    variance = max(0.0, float(w @ sigma_w))
    sigma = np.sqrt(variance)
    portfolio_returns = sample.dot(weights)
    mean = float(portfolio_returns.mean())
    z = NormalDist().inv_cdf(confidence)
    density = NormalDist().pdf(z)
    param_var = z * sigma - mean
    param_es = density / (1 - confidence) * sigma - mean
    component = -w * mu
    if sigma > 1e-15:
        component = component + z * w * sigma_w / sigma
    quantile = float(portfolio_returns.quantile(1 - confidence))
    tail = portfolio_returns <= quantile
    hist_var = -quantile
    hist_es = -float(portfolio_returns[tail].mean())
    hist_components = -sample.loc[tail].multiply(weights, axis=1).mean() * total
    vol_share = w * sigma_w / variance if variance > 1e-30 else np.zeros_like(w)
    individual_sigma = sample.std().to_numpy()
    diversification = float(w @ individual_sigma / sigma) if sigma > 1e-15 else np.nan
    rf_day = (1 + rf_annual) ** (1 / 252) - 1
    sharpe = float((portfolio_returns.mean() - rf_day) / sigma * np.sqrt(252)) if sigma > 1e-15 else np.nan
    wealth = (1 + portfolio_returns).cumprod()
    # Inclure le capital de départ pour ne pas ignorer une perte dès le premier jour.
    peaks = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / peaks - 1
    return dict(returns=portfolio_returns, n=len(sample), tail_n=int(tail.sum()),
                vol=sigma * np.sqrt(252), sharpe=sharpe, diversification=diversification,
                drawdown=drawdown, max_drawdown=float(drawdown.min()), wealth=wealth,
                hist_var=hist_var * total, hist_es=hist_es * total,
                param_var=param_var * total, param_es=param_es * total,
                var_components=pd.Series(component * total, index=weights.index),
                es_components=hist_components,
                vol_shares=pd.Series(vol_share, index=weights.index),
                individual_vol=pd.Series(individual_sigma * np.sqrt(252), index=weights.index))


def max_buy_quantity(prices, quantities, cash, ticker, limit=0.15):
    prices, quantities, values, total = validate_positions(prices, quantities, cash)
    room = max(0.0, total * limit - values[ticker])
    return max(0, int(np.floor(min(room, cash) / prices[ticker] + 1e-10)))


def simulate_order(prices, quantities, buy_prices, cash, ticker, side, quantity, execution_price, sectors):
    if ticker not in quantities or side not in ("Achat", "Vente"):
        raise ValueError("Valeur ou sens de l'ordre invalide.")
    if not np.isfinite(quantity) or quantity < 1 or quantity % 1 != 0:
        raise ValueError("La quantité doit être un entier supérieur ou égal à 1.")
    if not np.isfinite(execution_price) or execution_price <= 0:
        raise ValueError("Le cours de simulation doit être positif.")
    prices = pd.Series(prices, dtype=float).copy()
    prices[ticker] = execution_price
    prices, quantities, _, _ = validate_positions(prices, quantities, cash)
    before = allocation(prices, quantities, cash, sectors)
    amount = float(quantity * execution_price)
    old_qty = quantities[ticker]
    if side == "Vente" and quantity > old_qty:
        raise ValueError(f"Vente impossible : {old_qty:.0f} titres détenus, {quantity:.0f} demandés.")
    if side == "Achat" and amount > cash + 1e-8:
        raise ValueError(f"Liquidités insuffisantes : il manque {amount - cash:,.2f} €.")
    quantities = quantities.copy()
    quantities[ticker] += quantity if side == "Achat" else -quantity
    new_cash = max(0.0, cash - amount) if side == "Achat" else cash + amount
    old_pru = float(buy_prices[ticker])
    new_pru = (old_qty * old_pru + amount) / quantities[ticker] if side == "Achat" else (old_pru if quantities[ticker] else np.nan)
    after = allocation(prices, quantities, new_cash, sectors)
    return dict(before=before, after=after, quantities=quantities, cash=new_cash,
                amount=amount, pru=new_pru, price=execution_price)


def wilder_rsi(prices, period=14):
    """Amorçage par moyenne simple puis lissage de Wilder ; NaN sans historique."""
    output = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for ticker in prices:
        series = prices[ticker]
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = avg_loss = np.nan
        for i in range(period, len(series)):
            if not np.isfinite(delta.iloc[i]):
                avg_gain = avg_loss = np.nan
                continue
            if not np.isfinite(avg_gain):
                window = delta.iloc[i - period + 1:i + 1]
                if window.isna().any():
                    continue
                avg_gain = float(gain.iloc[i - period + 1:i + 1].mean())
                avg_loss = float(loss.iloc[i - period + 1:i + 1].mean())
            else:
                avg_gain = ((period - 1) * avg_gain + gain.iloc[i]) / period
                avg_loss = ((period - 1) * avg_loss + loss.iloc[i]) / period
            output.iloc[i, output.columns.get_loc(ticker)] = (
                50 if avg_gain == avg_loss == 0 else
                100 if avg_loss == 0 else
                100 - 100 / (1 + avg_gain / avg_loss)
            )
    return output
