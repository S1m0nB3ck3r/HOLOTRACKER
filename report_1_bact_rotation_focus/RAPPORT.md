# Une bactérie qui tourne : localisation selon la fonction de focus

27 novembre 2026. Script : `etude_focus_rotation.py`. Les résultats chiffrés sont dans les CSV
de ce répertoire et le résumé dans `resume_resultats.txt`.

## Ce qui a été mesuré

**Données.** `SIMU_1_BACT_ROTATION/`, un hologramme par angle φ, de 0° (bactérie dans l'axe
optique) à 90° (couchée) par pas de 5°, soit 19 hologrammes :

- un bâtonnet de 1,5 × 0,8 µm, incliné dans le plan XZ (θ = 0) ;
- centré au pixel (512, 512), à 50 µm du capteur ;
- pixel caméra 5,5 µm, grossissement ×40, soit 0,1375 µm au plan objet ;
- bruit capteur 0,015.

**Chaîne de traitement :**

1. l'hologramme brut, **sans division par l'hologramme de référence** ;
2. un filtre spectral **passe-haut seul (15)**, sans passe-bas ;
3. un volume de module reconstruit de 30 à 69,8 µm, au pas de 0,2 µm (200 plans) ;
4. le volume de focus, avec une fenêtre `sum_size` de 15 pixels (2 µm), pour chacune des
   cinq fonctions élémentaires : SUM_OF_VARIANCE, TENEGRAD, SUM_OF_LAPLACIAN,
   SUM_OF_GRADIENT et SUM_OF_INTENSITY. Les deux moyennes MEAN_ARITH_ALL et MEAN_GEO_ALL
   sont exclues ;
5. un seuil global (moyenne + K × écart-type du volume), puis CCL3D et le barycentre de
   chaque composante, pondéré par le focus ;
6. l'appariement à la position vraie, avec une tolérance de 3 µm en XY et de 5 µm en Z.

**Choix du K.** On a balayé 37 valeurs de K de 3 à 1000, en échelle logarithmique. Pour
chaque fonction de focus, on retient **un seul K pour les 19 angles**, comme HOLOTRACKER
applique un seul K à tout un film. Ce K est **le centre géométrique du plateau** où le F1
moyen est maximal. Ce critère laisse la plus grande marge des deux côtés : vers les fausses
détections quand K est trop bas, vers la perte de l'objet quand K est trop haut. L'erreur
de position change peu le long du plateau (0,03 à 0,06 µm en 3D), elle ne sert donc pas à
choisir.

**Contraste.** Le contraste est le max du volume de focus divisé par la moyenne du fond. Le
fond est tout le volume, sauf une colonne de 10 × 10 µm centrée sur la bactérie, sur toute
la profondeur.

## Résultats

| Focus | K retenu | Plateau de F1 = 1 | Trouvée | \|eX\| moyen | \|eY\| moyen | \|eZ\| moyen | Temps du focus |
|---|---:|---|---:|---:|---:|---:|---:|
| SUM_OF_VARIANCE | 64 | 10,9 à 323 (**×30**) | 19/19 | 0,004 µm | 0,003 µm | 0,039 µm | 399 ms |
| SUM_OF_GRADIENT | 89 | 20,8 à 323 (×15) | 19/19 | 0,006 µm | 0,003 µm | 0,037 µm | 279 ms |
| SUM_OF_INTENSITY | 47 | 12,8 à 144 (×11) | 19/19 | 0,005 µm | 0,004 µm | 0,040 µm | 185 ms |
| TENEGRAD | 40 | 20,8 à 64 (×3,1) | 19/19 | 0,005 µm | 0,004 µm | 0,034 µm | 245 ms |
| SUM_OF_LAPLACIAN | 21 | 12,8 à 34 (×2,6) | 19/19 | 0,026 µm | 0,012 µm | 0,031 µm | 233 ms |

Au K retenu, chaque fonction de focus trouve la bactérie aux 19 angles, avec une seule
détection par hologramme : F1 = 1 partout.

### Erreur en X et Y (`erreur_XY_vs_phi.png`)

L'erreur reste **très inférieure au pixel**, qui mesure 0,1375 µm. Pour quatre des cinq
fonctions, elle ne dépasse pas 0,02 µm. SUM_OF_LAPLACIAN est la seule qui se détache : jusqu'à
0,08 µm en X et 0,05 µm en Y. Son laplacien au carré amplifie les hautes fréquences, donc le
bruit, et peu de voxels passent son seuil. Pour les autres fonctions, l'orientation n'a pas
d'effet visible en XY. L'erreur en X, l'axe le long duquel la bactérie s'incline, n'est pas
plus grande que l'erreur en Y.

### Erreur en Z (`erreur_Z_vs_phi.png`)

L'erreur reste **inférieure au demi-plan** (0,1 µm) à tous les angles. Elle montre un **biais
négatif systématique** : de −0,02 à −0,04 µm en moyenne, jusqu'à −0,10 µm. La détection est
donc un peu trop près du capteur. Ce biais est plus marqué entre 5° et 65° et presque nul
pour une bactérie couchée (75° à 90°).

Les cinq fonctions de focus suivent la même courbe. Le biais ne vient donc pas du critère de
focus, mais du volume reconstruit lui-même, c'est-à-dire de la manière dont l'objet de phase
se refocalise. C'est sans doute le même biais que celui mesuré sur les simulations denses de
`SIMULATED HOLO` (−0,11 à −0,19 µm, pour un pas de 0,5 µm).

### Contraste (`contraste_vs_phi.png`, `contraste_normalise_vs_phi.png`, `contraste_sigma_vs_phi.png`)

- **Max / fond.** Toutes les fonctions perdent du contraste quand la bactérie se couche : le
  rapport est divisé par environ 2 entre 0° et 90°, et par 1,3 seulement pour TENEGRAD. Une
  bactérie dans l'axe traverse 1,5 µm de matière au lieu de 0,8 µm, donc déphase davantage
  l'onde.
- **Variation relative** (`contraste_normalise_vs_phi.png`, rapport divisé par sa valeur à
  φ = 0) : de 0° à 90°, SUM_OF_VARIANCE et SUM_OF_GRADIENT tombent à 0,47, SUM_OF_LAPLACIAN
  à 0,40, SUM_OF_INTENSITY à 0,54 et TENEGRAD à 0,77 seulement. L'écart de TENEGRAD vient
  surtout de sa forme : c'est une racine de gradient, alors que les autres sont des carrés.
  Or la racine de 0,47 vaut 0,69. TENEGRAD n'est donc pas réellement moins sensible à
  l'orientation, il comprime la même perte.
- **Comparer les fonctions entre elles avec ce rapport induit en erreur.** Il dépend de la
  forme de chaque critère : TENEGRAD est une racine de gradient, le laplacien est élevé au
  carré. SUM_OF_VARIANCE atteint 206 à 434, et TENEGRAD 8 à 10 seulement.
- **Le contraste en écarts-types, (max − moyenne) / σ, se compare directement au K.** C'est
  le K au-delà duquel le seuil global perd la bactérie. Pour SUM_OF_VARIANCE (≈ 360) et
  SUM_OF_GRADIENT (340 à 390), il reste plat avec l'orientation. SUM_OF_LAPLACIAN chute de 93 à
  35 : c'est le critère qui perd en premier une bactérie couchée, et ce qui borne son plateau
  à K = 34.

### Choix du K (`F1_vs_K.png`)

Sous le plateau, le F1 baisse parce que le bruit et les franges hors focus produisent des
fausses détections. Au-dessus, la bactérie disparaît. **SUM_OF_VARIANCE a de loin la plus
large plage de K sûrs**, un facteur 30. Le défaut actuel d'HOLOTRACKER, SUM_OF_VARIANCE avec
K = 15 en global, tombe dans ce plateau, mais près de son bas.

## Ce qu'il faut en retenir

1. **Sur une bactérie isolée, les cinq fonctions de focus localisent toutes correctement**,
   avec une erreur sous le voxel à toutes les orientations. Ce test ne les départage donc pas
   par l'erreur.
2. **Elles se départagent par la marge sur K.** SUM_OF_VARIANCE (×30), puis SUM_OF_GRADIENT
   (×15) et SUM_OF_INTENSITY (×11), tolèrent un K mal réglé. TENEGRAD (×3) et SUM_OF_LAPLACIAN
   (×2,6) demandent un réglage fin, et le laplacien perd en premier les bactéries couchées.
   Ce résultat conforte le choix de SUM_OF_VARIANCE comme focus par défaut.
3. **Le biais en Z** d'environ −0,04 µm, qui dépend de l'orientation, est commun à tous les
   critères. Il est petit devant le pas de reconstruction usuel (0,5 µm).

## Limites

- **Une bactérie seule, à faible bruit** (0,015) : c'est le cas le plus favorable. La largeur
  du plateau, en particulier sa borne basse, se réduira en présence de voisines et de bruit,
  comme dans `SIMULATED HOLO`.
- **Une seule position** (50 µm du capteur, au centre du champ) et un seul azimut (θ = 0).
- **Le pas de 0,2 µm est plus fin que l'usage** (0,5 µm). L'erreur en Z mesurée ici est
  celle d'un échantillonnage serré.
- **Le passe-haut seul et l'absence de division par l'hologramme de référence** sont les
  réglages demandés pour cette étude. Ils diffèrent de ceux du GUI et des scripts, qui sont
  encore à 15–125.
