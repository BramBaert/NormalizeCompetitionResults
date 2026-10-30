# Un classement de club équitable, tous âges et disciplines confondus

[English](README.md) | [Nederlands](README.nl.md) | **Français**

Dans le tir sportif, les catégories d'âge garantissent une compétition équitable. Un club local compte toutefois rarement assez de tireurs par catégorie pour que cela fonctionne : une poignée de tireurs au pistolet et quelques tireurs à la carabine, répartis sur sept groupes d'âge, ne font pas une compétition. Sans compétition, peu de raisons de participer, et sans tireurs, pas de compétition.

Ce projet propose de laisser tomber les catégories d'âge dans les compétitions de club **sans rendre la comparaison injuste**. Il s'appuie sur des années de résultats du championnat de Belgique pour tenir compte de l'âge, et même de la différence entre carabine et pistolet à air comprimé, afin que tous les membres d'un club puissent s'affronter dans un seul classement.

## L'idée : un score sur 100

Chaque résultat est converti en score : *imaginez 100 tireurs de votre discipline et de votre groupe d'âge au championnat de Belgique (BOA). Combien d'entre eux battriez-vous ?* Comme cette question a le même sens dans chaque groupe, un tireur au pistolet de 55 ans peut être comparé à un tireur à la carabine de 25 ans.

![Deux tireurs d'exemple sur les distributions réelles du BOA](pictures/score_explainer_fr.png?raw=true)

Le tireur A fait 590,0 à la carabine en Senior 1 et bat 57 tireurs sur 100 de ce groupe. Le tireur B fait 536 au pistolet en Senior 2 : 54 points de moins, mais meilleur que 86 tireurs sur 100 de ce groupe. **Au classement du club, B termine devant A.**

Remarquez à quel point l'échelle du score est irrégulière par rapport aux points : au milieu d'un groupe, là où les tireurs se bousculent, quelques points suffisent pour en dépasser beaucoup ; en bas, chaque place coûte beaucoup de points.

Un score peut dépasser légèrement 100 quand un résultat est meilleur que tout ce qui figure dans les données. C'est voulu : plus de points doit toujours donner un score plus élevé. Une qualification de niveau olympique de 633,0 à la carabine obtient 100,3, une qualification de 587 au pistolet 103,0.

## En pratique

Tous les tireurs du championnat de Belgique 2025, les deux disciplines et tous les âges, dans un seul classement :

| # | Tireur | Discipline | Catégorie | Résultat (coups) | Score |
| --: | :-- | :-- | :-- | --: | --: |
| 1 | DEBO Elise | Pistolet | Cadet | 375 (40) | 101,1 |
| 2 | DRUART Jean-Michel | Pistolet | D3 & S3 | 553 (60) | 100,0 |
| 3 | LIEBENS Bart | Pistolet | D1 & S1 | 568 (60) | 99,6 |
| 4 | ADRIAENSSEN Ruben | Carabine | Cadet | 409,5 (40) | 99,4 |
| 5 | HENNES Sandra | Carabine | Cadet | 409,4 (40) | 99,4 |
| 6 | SCHEPENS Nicolas | Carabine | D1 & S1 | 623,1 (60) | 98,5 |
| 7 | VANDEVYVERE Emma | Carabine | D1 & S1 | 623,0 (60) | 98,5 |
| 8 | HERBILLON Lana | Carabine | Junior | 620,5 (60) | 98,4 |
| 9 | VANDROOGENBROECK Océane | Carabine | Junior | 618,0 (60) | 97,4 |
| 10 | MAHAUT Claudy | Carabine | D3 & S3 | 584,6 (60) | 97,4 |

Cadets, juniors, seniors et vétérans des deux disciplines atteignent tous le sommet. Le classement complet se trouve dans [`BOA 2025 Normalized.csv`](BOA%202025%20Normalized.csv). Que ce soit équitable pour chaque groupe, et pas seulement au sommet, c'est ce que montre [la preuve par neuf](#la-preuve-par-neuf).

## Utilisation

### Classement de club dans un tableur

`club/Clubklassement St. Michiel.xlsx` est une proposition de classement, préparée pour le format de la compétition de club de St. Michiel Roeselare : une saison de 10 compétitions de 30 coups dont les 5 meilleures comptent. Il fonctionne dans Excel, LibreOffice Calc et Google Sheets, sans macros. Utilisez-le comme exemple : un autre format demande des adaptations dans `src/build_club_sheet.py` (et une table de correction pour un autre nombre de coups). Le classeur est en néerlandais (le club est flamand) :

- **Uitleg** : les paramètres (saison, nombre de compétitions qui comptent) et le mode d'emploi.
- **Resultaten** : par tireur le nom, la discipline, la catégorie et les résultats. Le score (« op 100 schutters », sur 100 tireurs) est calculé pour chaque résultat.
- **Klassement** : le classement imprimable sur une page en paysage, trié sur la somme des 5 meilleurs scores. Chaque tireur a deux lignes : les résultats tirés avec leurs 5 meilleurs et leur moyenne, et en dessous les scores.
- **Coëfficiënten** : la table de correction dans laquelle les scores sont recherchés.

### Lancer les scripts

Installez [uv](https://docs.astral.sh/uv/) et lancez les scripts depuis la racine du dépôt. Au premier lancement, uv installe Python 3.11 et les paquets nécessaires.

```
uv run src/process.py                    # ajuster les distributions, écrire la table de correction et le classement de la dernière année
uv run src/process.py --year 2024        # classement d'une année donnée
uv run src/process.py --no-plots         # ne pas ouvrir les figures plotly
uv run src/build_club_sheet.py           # construire le classeur du club à partir de la table de correction (--rows 200, --output <fichier>)
uv run src/plot_explainer.py             # dessiner la figure explicative en anglais, néerlandais et français (--html pour des versions interactives)
uv run src/build_poster.py               # affiche A4 en néerlandais et en français dans poster/ (imprimée avec Microsoft Edge)
```

| Option de `process.py` | Défaut | Signification |
| :-- | :-- | :-- |
| `--use-shot-count-normalized` | 1 | Ajuster sur les résultats ramenés à `--normalized-shot-count` coups |
| `--normalized-shot-count` | 30 | Nombre de coups de référence, p. ex. celui d'une compétition de club |
| `--use-unisex` | 1 | Regrouper dames et messieurs du même âge |
| `--score-scale` | 100 | L'échelle du score : « combien en battriez-vous sur autant de tireurs » |

| Sortie | Contenu |
| :-- | :-- |
| `normalization_tables/LogGamma_CDF_30shots.csv` | Table de correction de toutes les disciplines et tous les groupes |
| `BOA <année> Normalized.csv` | Tous les résultats de cette année classés selon leur score |
| `club/Clubklassement St. Michiel.xlsx` | Le classeur du club |
| `pictures/score_explainer_<en, nl, fr>.png` | La figure explicative |

**Mise à jour annuelle :** ajoutez les nouveaux résultats du BOA à `input_data/BE_national.csv`, puis lancez `process.py --no-plots`, `build_club_sheet.py` et `plot_explainer.py`.

## Comment ça marche

### Les données

[`input_data/BE_national.csv`](input_data/BE_national.csv) contient 3 096 résultats de qualification à la carabine 10 m (LK, *luchtkarabijn*) et au pistolet 10 m (LP, *luchtpistool*). Presque tous viennent du championnat de Belgique (BOA) : carabine 2018 - 2025, pistolet 2018 - 2025 plus quelques années antérieures, et quelques dizaines de résultats au pistolet de championnats provinciaux. Les résultats marqués comme anomalie sont écartés. La Belgique connaît ces catégories d'âge :

| Catégorie | Abréviation | Âge | Coups par compétition |
| :-: | :-: | :-: | :-: |
| Poussin | DUV | < 12 | 30 |
| Benjamin | BEN | 12 - 13 | 30 |
| Cadet | CAD | 14 - 15 | 40 |
| Junior Dame / Homme | JD / JH | 16 - 20 | 60 |
| Dame 1 / Senior 1 | D1 / S1 | 21 - 49 | 60 |
| Dame 2 / Senior 2 | D2 / S2 | 50 - 59 | 60 |
| Dame 3 / Senior 3 | D3 / S3 | 60+ | 60 |

Chaque discipline et chaque groupe d'âge est traité séparément, en quatre étapes.

### Étape 1 : le même nombre de coups

Les catégories tirent 30, 40 ou 60 coups, une compétition de club 30. Tous les résultats sont d'abord ramenés au format du club (un résultat de 600 sur 60 coups compte pour 300), pour que la table s'applique directement aux résultats du club. Le score ne dépend pas de cette conversion : 590,0 sur 60 coups et 295,0 sur 30 coups obtiennent tous deux 57. C'est pourquoi les figures montrent les points habituels sur 60 coups.

### Étape 2 : dames et messieurs ensemble

Autrefois les dames tiraient 40 coups et les messieurs 60 ; aujourd'hui tous tirent 60 coups. En Belgique, les dames obtiennent systématiquement de meilleurs scores que les messieurs à la carabine, mais les catégories D2 et D3 ont trop peu de résultats pour ajuster seules une distribution fiable. Au sommet du sport, dames et messieurs font jeu égal, comme le montrent les finales olympiques de Tokyo 2020 :

![Finales olympiques de Tokyo 2020](pictures/olympic_final.png?raw=true)

Les dames et messieurs du même âge forment donc un seul groupe : DUV, BEN, CAD, JUN, D1 & S1, D2 & S2 et D3 & S3. Les scripts peuvent toujours travailler par sexe (`--use-unisex 0`).

### Étape 3 : ajuster une distribution

Les résultats d'un groupe sont décrits par une distribution statistique, pour que le score ne dépende pas du hasard de quelques résultats. Une bonne candidate doit rendre la chute brutale en haut : il y a un score maximal et les meilleurs tireurs se pressent juste en dessous.

La première idée était une loi normale asymétrique, qui s'ajustait mal. La loi bêta ([trouvée ici](http://zoonek2.free.fr/UNIX/48_R/07.html)) peut produire une telle chute brutale :

![Fonctions bêta théoriques](pictures/beta_function.png?raw=true)

Sur 3 ans de données elle s'ajustait encore mal, sur 5 ans mieux, mais certaines catégories restaient récalcitrantes. La bibliothèque [distfit](https://erdogant.github.io/distfit/pages/html/index.html) a ensuite testé de nombreuses distributions sur toutes les catégories ; voici le classement selon le score cumulé de toutes les catégories (plus bas, c'est mieux) :

![Résultats d'ajustement de différentes distributions](pictures/dist_fitting_unisex.png?raw=true)

Loggamma et genextreme ont ainsi rejoint la comparaison (carabine, première analyse) :

![Fonctions de densité de probabilité](pictures/pdf_unisex.png?raw=true)

La loi bêta se comporte bizarrement pour les cadets (et pour D2 quand les sexes sont séparés). Un ajustement idéal passe par le milieu de chaque barre de l'histogramme :

![Histogramme idéal](pictures/ideal_histogram.jpg?raw=true)

Loggamma rend le mieux le pic et la chute brutale, et distfit la classe toujours première sur les données actuelles des deux disciplines. `process.py` prévient si cela change un jour.

### Étape 4 : de la distribution au score

La fonction de répartition (CDF) d'un groupe donne, pour chaque résultat, la part de ce groupe qui fait moins. Le score est cette part multipliée par 100.

Pour l'utiliser dans un tableur, la CDF est stockée sous forme de table : 1 000 résultats entre le point à 1 % et le point à 99 % de chaque groupe, chacun avec sa CDF. Un résultat est multiplié par le facteur `100 x CDF / points` de la ligne la plus proche. Entre 1 % et 99 %, l'écart avec la part exacte reste sous 0,2. En dehors, le facteur du bord s'applique, si bien que le score continue de croître avec le résultat et que les résultats exceptionnels dépassent 100. La table est [`normalization_tables/LogGamma_CDF_30shots.csv`](normalization_tables/LogGamma_CDF_30shots.csv) ; le classeur du club la reprend dans l'onglet *Coëfficiënten*.

### Comment on en est arrivé là

La première version n'avait pas de score sur 100. Elle convertissait chaque résultat en celui qu'il aurait été dans une **catégorie de référence** (Senior 1, 60 coups). Deux façons de le faire ont été comparées :

- **Méthode linéaire** : découper chaque catégorie entre son point à 1 % et son point à 99 % en pas égaux et prendre le résultat au même pas dans la catégorie de référence.
- **Méthode CDF** : prendre le résultat de la catégorie de référence qui a la même CDF, c'est-à-dire celui qui bat la même part de son groupe.

![Méthodes de normalisation](pictures/normalization_methods.png?raw=true)

Sur l'échelle de 600 points de la catégorie de référence, les deux semblaient bonnes, la linéaire même un peu meilleure :

![Box plot de la première version](pictures/result_box_plot_unisex.png?raw=true)

Mathématiquement, la méthode linéaire est pourtant fausse : la pente de la CDF diffère selon la catégorie, donc des pas égaux en points contiennent une part différente de tireurs dans chaque catégorie. Deux éléments ont ensuite rendu la catégorie de référence obsolète :

- **Les données du pistolet ont été ajoutées.** Une catégorie de référence à la carabine ne peut pas classer des tireurs au pistolet, et une référence par discipline donne toujours deux classements.
- **Le score devait devenir compréhensible.** « Vous battriez 57 tireurs de votre âge sur 100 » parle à chaque membre ; « en senior, vous auriez fait 287,4 » non.

Avec la part de tireurs comme score, les deux méthodes ont pu être comparées sur l'échelle qui compte. Le choix était alors fait, comme le montre le chapitre suivant.

## La preuve par neuf

Si les scores sont équitables, chaque groupe d'âge des deux disciplines a la même allure sur l'échelle du score : la moitié centrale des tireurs entre 25 et 75, la moitié au-dessus de 50. Les boîtes rouges, la méthode CDF utilisée, montrent exactement cela pour les 3 086 résultats des données.

![Score par discipline et groupe d'âge](pictures/score_box_plot.png?raw=true)

Les boîtes bleues sont la méthode linéaire : elle donne des scores trop élevés (médianes de 65 à 78) et varie selon le groupe. Elle n'est gardée que pour comparaison.

<details>
<summary>Comment lire un box plot</summary>

![Lire un box plot](pictures/reading_boxplots.png?raw=true)

La boîte contient la moitié centrale des scores, le trait plein dans la boîte est la médiane (la moitié des tireurs fait mieux) et le trait pointillé la moyenne. Les moustaches montrent le quart le plus bas et le quart le plus haut.
</details>

## Pistes pour la suite

- En 2024, la catégorie junior est passée de 16 - 17 à 16 - 20 ans, alors que la plupart des données utilisent encore les anciens âges. Les tireurs pourraient être suivis par leur nom pour reporter de 3 ans leur passage en D1 / S1. Idéalement sur 8 ans de données, en écartant les 3 plus anciennes pour garder assez de données corrigées.
- Une commande pour attribuer un score à un CSV de résultats avec une table de correction existante, sans refaire l'ajustement.
