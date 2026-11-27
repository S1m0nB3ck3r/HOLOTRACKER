# Types de seuil sur hologrammes denses, avec SUM_OF_GRADIENT

27 novembre 2026. Script : `etude_seuils_dense.py`. Les données sont dans les CSV de ce
répertoire et le résumé chiffré dans `journal_execution.txt`.

## Ce qui a été mesuré

**Données.** `SIMULATED HOLO/` : 8 densités, de N = 10 à 500 bactéries par hologramme, avec
10 hologrammes par densité. Les orientations sont tirées de façon isotrope.

**Chaîne de traitement.** Elle reprend celle du GUI, à trois réglages près, demandés pour
cette étude :

- le critère de focus est **SUM_OF_GRADIENT**, avec `sum_size` 15 ;
- le filtre spectral est un **passe-haut seul (15)** ;
- l'hologramme est divisé par sa moyenne, faute d'hologramme de référence.

Le volume a 240 plans de 0,5 µm depuis le capteur. Chaque seuil est suivi de CCL3D et du
barycentre de chaque composante, puis d'un appariement à la vérité (3 µm en XY, 5 µm en Z).

**Les quatre seuils** viennent des fonctions de `CCL3D.py`, avec les mailles du GUI
(25 plans sur 64 × 64 pixels, soit 12,5 × 11,2 × 11,2 µm) :

| Seuil | Formule | K balayés (16 valeurs) |
|---|---|---|
| global | moyenne + K × σ du volume | 2,5 à 150 |
| local sigma | fond écrêté (3 passes à ±3σ) + K × max(σ, plancher), par maille | 2,5 à 150 |
| local MAD | médiane + K × 1,4826 × MAD, par maille (**défaut du GUI**) | 2,5 à 150 |
| local médiane | médiane × (1 + K), par maille | 0,3 à 40 |

Pour chaque type de seuil, le K est **optimisé pour chaque densité** : celui qui maximise le
F1 moyen des 10 hologrammes. Pour comparaison, on note aussi le **meilleur K unique** pour
toutes les densités, celui qu'il faudrait choisir pour un film dont on ignore la densité.

## Résultats

### F1 (`seuils_detection_vs_densite.png`)

| Seuil | F1 moyen, K par densité | F1 moyen, K unique | F1 à N = 500, K par densité | F1 à N = 500, K unique |
|---|---:|---:|---:|---:|
| local sigma | **0,966** | **0,951** (K = 16,9) | **0,927** | **0,890** |
| local médiane | 0,965 | 0,945 (K = 4,1) | 0,926 | 0,892 |
| local MAD | 0,964 | 0,935 (K = 22,2) | 0,928 | 0,822 |
| global | 0,944 | 0,895 (K = 12,9) | 0,877 | 0,779 |

**1. Quand K est réglé pour chaque densité, les trois seuils locaux se valent.** Leur F1
moyen va de 0,964 à 0,966, et leurs courbes se superposent à toutes les densités. Le seuil
global n'est en retrait qu'à forte densité : 0,88 contre 0,93 à N = 500. Sa précision tombe
à 0,91 à N = 400 et 500, parce que pour garder le rappel il faut baisser K, ce qui laisse
passer le fond.

**2. Avec un K unique, l'écart se creuse.** Le seuil global perd 0,05 de F1 en moyenne et
0,10 à N = 500. Parmi les seuils locaux, local sigma et local médiane tiennent le mieux.
Local MAD, le défaut du GUI, perd davantage à forte densité : 0,82 contre 0,89 à N = 500,
car son K unique (22) est trop haut pour les densités fortes.

Cette dernière différence est à prendre avec prudence. La grille de K avance par pas de
×1,31, et le K = 20 du GUI tombe entre deux points (16,9 et 22,2). Un K de 17 à 20 pour
local MAD réduirait probablement l'écart.

### Le K optimal dépend de la densité, pour tous les seuils (`seuils_K_optimal_vs_densite.png`)

| Seuil | K optimal à N = 10 | à N = 500 | Rapport |
|---|---:|---:|---:|
| global | 22,2 | 7,5 | ×3,0 |
| local sigma | 29,2 | 12,9 | ×2,3 |
| local MAD | 29,2 | 12,9 | ×2,3 |
| local médiane (1 + K) | 8,8 | 3,9 | ×2,3 |

On aurait pu attendre du seuil local qu'il rende K indépendant de la densité. Il ne le fait
qu'en partie : le K optimal varie d'un facteur 2,3 au lieu de 3. La raison est mesurée dans
`report_focus_dense_holo` : quand la densité monte, le fond diffus monte **et fluctue
davantage**, alors que le signal d'une bactérie reste le même. Exprimé en unités de
dispersion du fond, son contraste baisse. Pour la garder, il faut baisser K, et un seuil
local n'y change rien, puisqu'il mesure justement cette dispersion.

Le seuil local corrige ce qui varie **dans** un hologramme : avec la profondeur, ou entre
régions. Il ne corrige pas ce qui varie **d'un hologramme à l'autre** avec la densité.
Dans ces simulations, les bactéries sont réparties uniformément, donc l'hétérogénéité dans
un même hologramme est faible. Sur un échantillon réel avec des amas, l'avantage du local
sur le global serait probablement plus grand qu'ici.

### Bactéries couchées (`seuils_rappel_orientation.png`)

À N = 500, les seuils locaux retrouvent 85 à 87 % des bactéries couchées, contre 80 % pour
le global. Les debout sont retrouvées à 92 à 96 %. Le rapport couchées / debout reste
proche de 0,9 pour tous les seuils. Le type de seuil influe peu sur le biais d'orientation.

### Erreur de position (`seuils_erreur_XYZ_vs_densite.png`)

Le type de seuil ne change presque rien à la précision :

- **biais en X et Y** : moins de 0,007 µm ;
- **écart-type en X et Y** : de 0,04 à 0,14 µm selon la densité ;
- **biais en Z** : de −0,03 à −0,08 µm, le même biais que dans les études précédentes ;
- **écart-type en Z** : de 0,12 à 0,45 µm.

À forte densité, local sigma et local MAD sont un peu plus précis en Z (0,35 µm à N = 500)
que global et local médiane (0,42 à 0,44 µm).

### Temps de calcul par hologramme (`seuils_temps.csv`)

| Étape | ms |
|---|---:|
| mailles local médiane | 10 |
| mailles local MAD | 19 |
| mailles local sigma | **258** |
| interpolation du seuil local au volume (commune aux trois) | 167 |
| seuil global (moyenne et écart-type du volume) | 415 |

Le temps du seuil global est mesuré **avec CUB désactivé**, ce qu'impose `nvcc` sur cette
machine. Dans le GUI, il est probablement plus court. Les statistiques par maille
dépendent peu de ce réglage, car elles réduisent de petits tableaux.

## Ce qu'il faut en retenir

1. **Seuil local plutôt que global** à partir d'une centaine de bactéries, et surtout si
   l'on garde un K unique.
2. **Entre les trois seuils locaux**, l'écart ne compte qu'avec un K unique :
   - local sigma tient le mieux, mais ses statistiques sont 13 fois plus lentes que
     celles de local MAD ;
   - local médiane est presque aussi robuste, avec la statistique la plus rapide (10 ms) et
     un seul paramètre ;
   - local MAD, le défaut actuel, se vaut avec les deux autres quand K est adapté, mais son
     K unique optimal est mal cerné par cette grille.
3. **Le K optimal baisse avec la densité, d'un facteur 2 à 3, quel que soit le seuil.** Un
   film dense demande un K plus bas qu'un film clairsemé. Le défaut du GUI (local MAD,
   K = 20) correspond à une densité de 50 à 100 bactéries par hologramme. À N = 300–500,
   K = 13 à 17 serait meilleur.

## Limites

- **Une seule fonction de focus**, SUM_OF_GRADIENT. Dans `report_focus_dense_holo`, avec le
  seuil global, c'était la deuxième derrière SUM_OF_VARIANCE.
- **La grille de K est grossière** (×1,31 par pas). Les différences de F1 inférieures à
  0,01 ne sont pas significatives.
- **Des mailles fixes** de 64 × 64 × 25 voxels, sans recouvrement. Leur taille n'est pas
  optimisée ici.
- **Les simulations ont un fond homogène dans le plan.** Elles avantagent peu le seuil
  local par rapport à un échantillon réel.
