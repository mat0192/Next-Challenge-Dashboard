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

Le compte initial reprend les positions saisies et la base NextWise de 50 959,17 € avec 10 654,39 € de cash et 480 Stellantis. `portfolio_config.py` est uniquement le point de départ lorsqu'aucune sauvegarde n'existe.

Positions, PRU, liquidités, capital de référence, taux et journal sont automatiquement enregistrés dans le stockage local du navigateur. Attendre **« Enregistré dans ce navigateur »** avant de fermer la page. Au prochain chargement, la sauvegarde est lue et validée avant toute écriture ; elle prend priorité sur la configuration initiale. « Actualiser les cours » ne touche qu'au cache Yahoo.

Le JSON contient le portefeuille, pas les cours. Restaurer une ancienne sauvegarde restaure donc les anciennes **quantités et liquidités**, valorisées avec les cours récupérés actuellement. Utiliser la sauvegarde la plus récente après un achat ou une vente. L'import remplace aussi le journal ; ce n'est pas une fusion.

**Sauvegarde et transfert**, dans la barre latérale, permet d'exporter une copie ou de l'importer sur un autre appareil. Le stockage est propre au navigateur et à l'adresse de l'application : navigation privée, suppression des données du site ou changement d'appareil peuvent nécessiter cet import. Ces commandes restent disponibles même si Yahoo ne répond pas. Une erreur de stockage est affichée ; une sauvegarde illisible ou modifiée par un autre onglet n'est pas écrasée automatiquement. Modifier le portefeuille dans un seul onglet à la fois.

Aucune donnée personnelle modifiée n'est enregistrée dans un fichier serveur partagé ou dans GitHub. Les données sont transmises à la session Streamlit pour les calculs. Aucune interaction ou transaction n'est envoyée à NextWise : reporter manuellement les positions et le cash après exécution des ordres.

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
node --test tests/test_portfolio_storage.mjs
```

Tests déterministes sans connexion Yahoo : valeurs analytiques de VaR/CVaR, additivité des contributions, ordre des indices, proportionnalité au cash, cas sans volatilité, repli initial, bornes de capacité d'achat, ventes totales et excessives, RSI, cours manquants, sauvegardes et parcours Streamlit.

## Sources

- [Règlement NextChallenge, consulté le 15 octobre 2026](https://www.nextwise.fr/R%C3%A8glement-du-Next-Challenge?lng=fr)
- [yfinance.download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [William F. Sharpe, The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
- [Dirk Tasche, Capital Allocation to Business Units and Sub-Portfolios](https://arxiv.org/abs/0708.2542)
- [TOBAM, Diversification Ratio](https://www.tobam.fr/wp-content/docs/GROUND_RULES/en/TOBAM%20Maximum%20Diversification%20Equity%20Index%20Series%20Ground%20Rules_March2022.pdf)
