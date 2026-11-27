# Contraste des bactéries par rapport au fond diffus, sur les hologrammes denses

27 novembre 2026. Script : `etude_contraste_dense.py`. Les données sont dans
`contraste_bacteries.csv` (une ligne par bactérie et par focus) et `fond_hologrammes.csv`.

## Ce qui a été mesuré

**Données.** `SIMULATED HOLO/` : 8 densités, de N = 10 à 500 bactéries par hologramme, avec
10 hologrammes par densité, soit 25 000 bactéries environ. Les orientations sont tirées de
façon isotrope.

**Chaîne de traitement :**

1. la lecture de `benchmark_detection.lire_simulation` (division par la moyenne de
   l'hologramme, X sur les colonnes, Y sur les lignes, Z depuis le capteur) ;
2. un filtre **passe-haut seul (15)** ;
3. un volume de 240 plans de 0,5 µm depuis le capteur, avec un pixel objet de 0,175 µm ;
4. `sum_size` 15 et les cinq fonctions de focus élémentaires.

**Définitions :**

- **signal** : la moyenne du focus sur les **3 × 3 × 3 voxels centrés sur la position vraie**
  de chaque bactérie, soit 0,5 × 0,5 × 1,5 µm ;
- **fond diffus** : la **médiane du volume de focus**. Les bactéries occupent moins de
  0,1 % des voxels, donc la médiane ne les voit pas. Elle mesure le bruit et les franges
  superposées de toutes les bactéries ;
- **contraste** = signal / fond diffus.

On note aussi (signal − fond) / (1,4826 × MAD du volume), l'échelle du K du seuil MAD.

Seules les bactéries dont le cube touche un bord du volume sont exclues, soit moins de
1,5 %.

## Résultats (`contraste_vs_densite.png`, `contraste_normalise_vs_densite.png`)

**Contraste médian (signal / fond diffus) :**

| Focus | N = 10 | 50 | 100 | 200 | 500 | À N = 500, rapporté à N = 10 |
|---|---:|---:|---:|---:|---:|---:|
| SUM_OF_VARIANCE | 89 | 59 | 43 | 27 | 12 | 0,14 |
| SUM_OF_GRADIENT | 60 | 36 | 26 | 17 | 8,3 | 0,14 |
| SUM_OF_INTENSITY | 33 | 22 | 16 | 10 | 5,3 | 0,16 |
| TENEGRAD | 5,7 | 4,4 | 3,8 | 3,1 | 2,3 | 0,41 |
| SUM_OF_LAPLACIAN | 3,8 | 3,4 | 3,3 | 3,1 | 2,6 | 0,68 |

**Le contraste baisse parce que le fond monte, pas parce que la bactérie s'affaiblit.**
Entre N = 10 et 500, le fond diffus est multiplié par 6,2 pour SUM_OF_VARIANCE, alors que le
contraste est divisé par 7. Le signal sur la bactérie ne change donc presque pas. Les franges
de toutes les bactéries s'additionnent dans le volume de focus.

**Comparer les variations entre focus** :

- **Les trois critères « au carré »** (VARIANCE, GRADIENT, INTENSITY) perdent la même
  proportion, environ 85 %.
- **TENEGRAD** semble moins perdre (0,41). Mais c'est une racine de gradient, et la racine
  de 0,14 vaut 0,37 : c'est la même perte, comprimée, comme dans l'étude de la rotation.
- **SUM_OF_LAPLACIAN est réellement moins sensible au fond diffus**, avec 0,68 pour un
  critère pourtant au carré. Son fond n'est multiplié que par 1,8. Le laplacien ne garde
  que les hautes fréquences, peu présentes dans les franges hors focus. Mais son contraste
  de départ est très bas (3,8), et 7 à 29 % des bactéries ont un contraste inférieur à 2
  selon la densité. Pour SUM_OF_VARIANCE, cette part reste sous 0,3 % même à N = 500.

**Contraste en unités de MAD du fond**, l'échelle du K du seuil local MAD :

| Focus | N = 10 | 100 | 500 |
|---|---:|---:|---:|
| SUM_OF_VARIANCE | 564 | 153 | 34 |
| SUM_OF_GRADIENT | 304 | 88 | 25 |
| SUM_OF_INTENSITY | 244 | 55 | 13 |
| TENEGRAD | 47 | 19 | 8,9 |
| SUM_OF_LAPLACIAN | 20 | 16 | 9,3 |

Sur cette échelle, qui décide de la détection, **SUM_OF_VARIANCE reste en tête à toutes
les densités**. À N = 500, la bactérie médiane est encore à 34 MAD du fond pour
SUM_OF_VARIANCE, contre 9 pour TENEGRAD et SUM_OF_LAPLACIAN. Le K = 20 par défaut du GUI
garde donc la bactérie médiane avec SUM_OF_VARIANCE, mais pas avec ces deux critères.

**Les bactéries couchées** (φ > 60°) ont un contraste un peu plus faible : pour
SUM_OF_VARIANCE à N ≤ 50, 56 contre 73. C'est cohérent avec l'étude de la rotation.

## Limites

- **Le fond est la médiane du volume entier**, alors que le seuil local la calcule par
  maille de 64 × 64 × 25 voxels. Le fond réel autour d'une bactérie dans une région dense
  est plus élevé que la médiane globale.
- **Le cube est centré sur la position vraie, pas sur la détection.** Un décalage d'un
  voxel du sommet du focus abaisse le signal mesuré.
- **Le rapport signal / fond ne se compare pas d'un focus à l'autre** (racine contre carré).
  Les variations normalisées et l'échelle MAD se comparent.

---

# Détection par seuil global optimisé (`etude_detection_dense.py`)

Mêmes hologrammes et même reconstruction : passe-haut seul (15), 240 plans de 0,5 µm,
`sum_size` 15. La détection est faite comme dans HOLOTRACKER : seuil global
(moyenne + K × σ du volume), CCL3D, puis barycentre de chaque composante.

Les détections sont appariées à la vérité avec une tolérance de 3 µm en XY et de 5 µm en Z.
On balaie 16 valeurs de K, de 2,5 à 250. Pour chaque focus, on retient **un seul K pour
toutes les densités** : celui qui maximise le F1 moyen sur les densités.

Les données sont dans `detection_balayage_K.csv` et `detection_K_optimal.csv`. Graphes :
`detection_vs_densite.png`, `rappel_orientation_vs_densite.png`,
`erreur_XYZ_vs_densite.png`, `detection_F1_vs_K.png`.

## Précision, rappel, F1

| Focus | K | F1 moyen | F1 à N = 10 | 100 | 500 | Rappel à N = 500 |
|---|---:|---:|---:|---:|---:|---:|
| SUM_OF_VARIANCE | 8,5 | **0,94** | 0,93 | 0,98 | **0,89** | **0,81** |
| SUM_OF_GRADIENT | 11,6 | 0,89 | 0,79 | 0,97 | 0,81 | 0,69 |
| SUM_OF_INTENSITY | 8,5 | 0,84 | 0,88 | 0,94 | 0,62 | 0,46 |
| TENEGRAD | 8,5 | 0,68 | 0,18 | 0,92 | 0,63 | 0,48 |
| SUM_OF_LAPLACIAN | 15,8 | 0,58 | 0,78 | 0,70 | 0,30 | 0,18 |

**SUM_OF_VARIANCE a le meilleur F1 aux huit densités.** Avec le seuil global, c'est aussi
celui qui perd le moins de bactéries quand la densité monte.

**TENEGRAD ne trouve pas de K qui convienne à toutes les densités.** Au K retenu, sa
précision n'est que de 0,10 à N = 10 et de 0,27 à N = 20 : il y a une dizaine de fausses
détections par bactérie. Son pic de F1 en fonction de K est étroit, et l'optimum se déplace
avec la densité. C'est la même fragilité que dans l'étude de la rotation, où la plage de K
utilisable n'était que de ×3.

## Les bactéries couchées

| Rappel à N = 500 | Couchées (φ > 60°) | Debout (φ < 30°) | Couchées / debout |
|---|---:|---:|---:|
| SUM_OF_VARIANCE | **0,75** | **0,92** | **0,82** |
| SUM_OF_GRADIENT | 0,60 | 0,84 | 0,71 |
| TENEGRAD | 0,40 | 0,70 | 0,57 |
| SUM_OF_INTENSITY | 0,37 | 0,69 | 0,54 |
| SUM_OF_LAPLACIAN | 0,12 | 0,42 | 0,29 |

Jusqu'à N = 100, toutes les fonctions sauf LAPLACIAN retrouvent les bactéries couchées
presque aussi bien que les debout. Au-delà, elles perdent d'abord les couchées.
**SUM_OF_VARIANCE est la fonction qui les perd le moins** : son rapport couchées / debout
est le plus proche de 1 à toutes les densités, et son rappel des couchées est le plus haut.

**TENEGRAD perd davantage les couchées que SUM_OF_VARIANCE.** Sa faible variation du
contraste avec l'angle (`contraste_normalise_vs_phi.png`, étude de la rotation) ne se
traduit pas en détection.

## Erreur de position

- **Biais en X et Y : nul**, à quelques nanomètres près, pour toutes les fonctions. Cela
  confirme encore la convention des axes.
- **Écart-type en X et Y** : SUM_OF_VARIANCE et SUM_OF_GRADIENT sont les plus précis, de
  0,03 µm à N = 10 à 0,12–0,13 µm à N = 500, soit moins d'un pixel (0,175 µm).
  SUM_OF_INTENSITY et TENEGRAD montent jusqu'à 0,19–0,25 µm.
- **Écart-type en Z** : SUM_OF_LAPLACIAN est le plus faible (0,13 à 0,29 µm), mais sur les
  seules bactéries qu'il trouve, les plus faciles. Viennent ensuite SUM_OF_VARIANCE et
  SUM_OF_GRADIENT (0,15 à 0,37 µm), puis TENEGRAD (environ 0,38 µm). SUM_OF_INTENSITY est
  nettement le moins précis (0,5 à 1 µm), avec un biais de −0,47 µm à N = 10.
- **Biais en Z** : il est négatif pour toutes les fonctions, de −0,04 à −0,10 µm hors
  SUM_OF_INTENSITY, comme dans l'étude de la rotation. Ce biais était de −0,11 à −0,19 µm
  dans `benchmark_detection.py`, qui utilise le filtre 15–125 et K = 11. Le passe-bas
  pourrait donc y contribuer, mais K diffère aussi entre les deux mesures : ce n'est pas
  établi.

## Conclusion sur le choix du focus

Sur tous les critères mesurés, **SUM_OF_VARIANCE est le meilleur choix** : F1, rappel des
bactéries couchées, uniformité selon l'orientation et précision en XY. SUM_OF_GRADIENT est
un second proche. TENEGRAD et SUM_OF_LAPLACIAN sont à écarter pour la détection.

**Limites** :

- On n'a testé que le seuil global. Le GUI utilise par défaut le seuil local MAD, qui
  devrait mieux tenir à forte densité. L'ordre entre fonctions de focus reste à vérifier
  avec ce seuil.
- La grille de K est grossière, ×1,36 entre deux valeurs, et trois focus ont leur optimum
  au même point (8,5). Une grille plus fine autour de 6 à 16 préciserait les K sans changer
  le classement.
