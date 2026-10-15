# NextChallenge Solo

Dashboard Streamlit de suivi de portefeuille et de simulation d'ordres.

## Démarrage

```sh
pip install -r requirements.txt
streamlit run dashboard.py
```

## Utilisation

- **Pilotage** : concentrations, répartition et points à examiner.
- **Positions** : quantités, PRU, cours, capacité d'achat et édition des lignes.
- **Signaux** : constats de momentum, moyennes mobiles et RSI de Wilder.
- **Risque** : mesures historiques et gaussiennes, contributions réconciliables, corrélations et stress.
- **Simulateur** : comparaison avant/après, contrôles de cash et de quantité, repères de diversification.
- **Journal & concours** : raisonnement, invalidation, sources et revue personnelle.

Le compte initial reprend les positions saisies et la base NextWise de 50 959,17 € avec 8 662,64 € de cash. Les modifications de positions, paramètres et notes restent propres à chaque session. Exporter le JSON dans Positions pour les conserver, puis restaurer ce fichier lors d'une prochaine session. Rien n'est écrit dans un fichier partagé du serveur ; aucune interaction ou transaction n'est envoyée à NextWise.

## Conventions de calcul

Les derniers `Close` non ajustés valorisent les titres. Les variations de `Adj Close` servent aux calculs historiques. Les rendements manquants ne sont pas remplis par zéro ; l'échantillon commun est indiqué. Un cours manquant interrompt la valorisation ; les données de plus de sept jours désactivent les simulations. Une barre quotidienne peut être encore en formation et ne garantit pas le cours d'exécution.

Les poids historiques sont ceux de l'allocation actuelle, maintenus constants à chaque séance. Cash sans rendement. Il ne s'agit pas du parcours réel du participant. Le Sharpe utilise un taux de référence annuel paramétrable (0 % par défaut, hypothèse uniquement) et 252 séances par an. Sans historique réel de valorisation et de flux, aucun Sharpe réalisé ou score de concours n'est produit. Le CAC 40 prix est un repère illustratif hors dividendes, différent des cours ajustés du portefeuille.

Pour des rendements de moyenne `mu`, covariance `Sigma`, poids `w` et écart-type `sigma` :

- VaR normale à 95 % : `(z * sigma - mu_portfolio) * valeur`.
- CVaR normale : `(phi(z) / 0.05 * sigma - mu_portfolio) * valeur`.
- Contribution Euler à la VaR : `(-w_i * mu_i + z * w_i * (Sigma @ w)_i / sigma) * valeur`.
- Contributions historiques : pertes moyennes pondérées sur les observations de queue ; elles totalisent la **CVaR**, pas la VaR.

Les quantiles gardent leur signe. Les contributions totalisent leur mesure, y compris les effets compensateurs négatifs. Le HHI porte sur les titres seuls, normalisés à 100 %, et exclut le cash. Le ratio de diversification est la moyenne pondérée des volatilités divisée par la volatilité totale. Les replis intègrent le capital avant le premier rendement.

Le simulateur valorise la ligne au cours hypothétique choisi avant et après l'ordre ; il conserve ainsi le capital hors frais. Il refuse de vendre plus que la quantité détenue ou d'acheter avec un cash insuffisant. Les alertes de diversification ne prétendent pas reproduire le moteur officiel de validation.

Les secteurs sont indicatifs et modifiables. L'ETF diversifié est exclu du décompte des secteurs ; ses constituants ne sont pas détaillés. Pour toute valeur ajoutée, vérifier son éligibilité sur NextWise et sa cotation en EUR. Les règles solo et la date de consultation sont accessibles depuis l'interface ; le barème officiel n'est pas reconstruit.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests déterministes sans connexion Yahoo : valeurs analytiques de VaR/CVaR, additivité des contributions, ordre des indices, proportionnalité au cash, cas sans volatilité, repli initial, bornes de capacité d'achat, ventes totales et excessives, RSI, cours manquants, sauvegardes et parcours Streamlit.

## Sources

- [Règlement NextChallenge, consulté le 15 octobre 2026](https://www.nextwise.fr/R%C3%A8glement-du-Next-Challenge?lng=fr)
- [yfinance.download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [William F. Sharpe, The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
- [Dirk Tasche, Capital Allocation to Business Units and Sub-Portfolios](https://arxiv.org/abs/0708.2542)
- [TOBAM, Diversification Ratio](https://www.tobam.fr/wp-content/docs/GROUND_RULES/en/TOBAM%20Maximum%20Diversification%20Equity%20Index%20Series%20Ground%20Rules_March2022.pdf)
