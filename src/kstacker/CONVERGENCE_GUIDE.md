# Guide de lecture des diagnostics MCMC

Ce guide concerne les figures du dossier `values/figures/convergence/`. Les figures du dossier `science/` décrivent les contraintes orbitales physiques et doivent être interprétées séparément des diagnostics d'échantillonnage.

## 1. `cold_chain_traces`

Chaque panneau correspond à un paramètre de l'orbite (`a`, `lambda_ref`, masse stellaire, `h`, `k`, `p`, `q`, et éventuellement `fp`). Les traits transparents sont des marcheurs de la chaîne froide ; la courbe foncée représente la médiane instantanée des marcheurs.

**À rechercher :** des fluctuations autour d'une distribution relativement stationnaire, des échanges entre régions plausibles et l'absence de dérive persistante. Des marcheurs durablement séparés peuvent révéler une multimodalité physique ou une exploration incomplète. Une trace visuellement stable ne suffit jamais à prouver la convergence.

## 2. `sampler_overview`

**Log posterior :** les trajectoires de log posterior doivent explorer des valeurs comparables au cours de la production. Une augmentation durable ou l'apparition tardive de valeurs nettement supérieures suggère une exploration inachevée.

**Fraction d'acceptation :** il n'existe pas de seuil universel. Des valeurs proches de zéro signalent souvent des propositions très rarement acceptées ; une acceptation très élevée peut correspondre à des déplacements inefficaces. Il faut la lire avec les traces et les temps d'autocorrélation, en tenant compte du type de move `emcee`.

**ACF :** la courbe de corrélation affichée est celle de la **médiane des marcheurs pour `a`**. Elle illustre la persistance temporelle, mais **ne remplace pas** les autocorrélations intégrées des chaînes ni le calcul du nombre d'échantillons effectifs (ESS).

## 3. Pour `emcee`

- Vérifier la dispersion et le recouvrement des marcheurs, idéalement après plusieurs initialisations physiquement distinctes.
- Comparer la durée de production au temps d'autocorrélation intégré (`tau`) pour chaque paramètre, avec une attention particulière aux paramètres angulaires et aux modes dégénérés.
- Recalculer les statistiques sur des fractions successives de la chaîne ; des intervalles crédibles qui changent fortement indiquent une incertitude d'échantillonnage encore importante.
- Ne pas utiliser le thinning comme un remède à une faible convergence. Il réduit principalement la taille du fichier de sortie.

## 4. Pour `reddemcee` / APT

`adaptive_tempering_diagnostics` ajoute des éléments spécifiques au tempérage parallèle adaptatif :

**Échelle des températures.** Le panneau `beta` représente l'inverse de température ; la chaîne froide correspond à `beta = 1`. Les températures plus chaudes (`beta` plus petit) explorent un paysage de vraisemblance aplati. Une échelle très étirée ne garantit pas des échanges efficaces.

**Mobilité entre températures.** Les trajectoires médianes du demi-grand axe par température renseignent sur l'exploration des régions orbitales. Des chaînes chaudes mobiles mais une chaîne froide bloquée peuvent signaler des échanges insuffisants.

**Historique de swaps et de l'adaptation.** Quand le moteur expose les historiques `get_tsw()` et `get_smd()`, ils sont tracés tels que fournis, sans attribuer arbitrairement une signification probabiliste à chaque élément. Leur normalisation et leur forme exacte sont définies par `reddemcee`. Un historique indisponible apparaît comme tel, sans valeur inventée.

**Chaîne utilisée pour l'inférence :** les postérieurs physiques et les intervalles proviennent de la **chaîne froide**. Mélanger directement toutes les températures fausserait la distribution cible.

**Vérification supplémentaire souhaitable :** suivre la dynamique des températures au cours de l'adaptation, vérifier la circulation des marcheurs entre zones chaudes et froides, et comparer plusieurs réalisations indépendantes.

## 5. Interprétation scientifique et publication

Une bonne convergence signifie que l'algorithme explore correctement **la distribution cible choisie** ; elle ne valide ni le modèle instrumental ni l'hypothèse d'indépendance des instruments. Un mauvais modèle de bruit ou une calibration incompatible peut produire des postérieurs étroits même lorsque les traces semblent parfaites.

Pour un résultat publiable : documenter les données et unités, les masques, le modèle de bruit, les prior, le choix de likelihood par instrument, le nombre de marcheurs et d'itérations, les diagnostics d'autocorrélation/ESS, ainsi que la robustesse aux initialisations et aux hypothèses instrumentales. Le graphique d'empilement recalé sur l'orbite est **illustratif** et ne doit jamais servir de preuve quantitative indépendante de détection.
