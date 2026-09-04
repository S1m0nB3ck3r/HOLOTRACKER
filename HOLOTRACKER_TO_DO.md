# HOLOTRACKER — constats mesurés et pistes de correction

Document destiné au projet **HOLOTRACKER** (pipeline classique de localisation
holographique). Il rassemble des mesures faites le 2026-09-04 sur le simulateur
`Simu-Bacteria-Holograms`, avec la même chaîne optique et les mêmes modules
(`libs/propagation.py`, `libs/focus.py`, `libs/CCL3D.py`) que ceux d'HOLOTRACKER.

Les conclusions portent sur le pipeline **propagation → focus TENEGRAD → seuil →
CCL3D**, et expliquent trois comportements observés expérimentalement par Simon :
l'influence de l'orientation des bactéries, la dispersion énorme du nombre de
voxels par objet, et l'échec de la séparation en milieu dense.

---

## 1. Ce qui a été mesuré

Conditions : bactérie L = 3 µm, e = 1 µm, Δn = 0,02, λ = 660 nm, n = 1,33,
grossissement 40, pixel caméra 5,5 µm (voxel objet 0,1375 µm), pas Z 0,5 µm,
focus TENEGRAD `SUM_SIZE = 15`, CCL3D connectivité 26.

### 1.1 L'orientation change le signal d'un facteur 3,6

| φ | trajet optique traversé selon z | pic du focus |
|---:|---:|---:|
| 0° (axe parallèle à l'illumination) | 3,50 µm | **0,4106** |
| 30° | 2,00 µm | 0,2852 |
| 60° | 1,50 µm | 0,2098 |
| 90° (bactérie couchée) | 0,75 µm | **0,1133** |

Décroissance monotone, **rapport 3,63×** entre les deux extrêmes.

**Cause** : le déphasage accumulé vaut `2π·Δn·(trajet dans l'objet)/λ`. Un bâtonnet
aligné sur l'axe optique présente 3 µm de matière au rayon qui le traverse ; couché,
seulement 1 µm d'épaisseur. L'amplitude diffusée est proportionnelle à ce déphasage.

C'est un effet **purement projectif**. Il ne vient ni de l'image jumelle, ni de la
diffusion multiple. Un montage hors-axe ne le supprimerait pas.

### 1.2 Le nombre de voxels segmentés amplifie cet écart à la puissance 3

Même balayage, **un seul seuil global figé** :

| φ | pic / seuil | nb voxels retenus | plus grosse composante connexe |
|---:|---:|---:|---:|
| 0° | 7,41× | 459 399 | **459 399** |
| 30° | 5,15× | 243 256 | 243 256 |
| 60° | 3,79× | 140 215 | 140 215 |
| 90° | 2,04× | 20 951 | **8 777** |

**3,63× sur le pic → 21,9× sur le nombre de voxels → 52,3× sur la plus grosse
composante.** Loi empirique : `V ∝ (pic/seuil)^2,4`, plus grosse composante en `^3,1`.

Noter la dernière ligne : à φ = 90°, les 20 951 voxels retenus se **fragmentent**, la
plus grosse composante n'en fait que 8 777. Une bactérie couchée n'est pas détectée
comme un objet faible — elle est **éclatée en morceaux**, dont beaucoup tombent sous
`nb_vox_min`.

> **Conséquence directe : `nb_pix` n'est pas un descripteur d'objet.** C'est une mesure
> du dépassement de seuil élevée au cube. `CCL_filter(nb_vox_min, nb_vox_max)` est donc
> un filtre d'orientation déguisé : il élimine préférentiellement les bactéries
> perpendiculaires à l'axe optique.

### 1.3 Le seuil global se couple à la densité — c'est le défaut le plus grave

`calc_threshold()` calcule `moyenne + n·écart-type` sur le **volume entier, objets
compris**. Une bactérie témoin faible (couchée) est placée toujours au même endroit,
et la charge autour d'elle augmente :

| nb bactéries | seuil 10 σ | pic du témoin | pic / seuil | témoin détecté ? |
|---:|---:|---:|---:|:--|
| 1 | 0,0554 | 0,1133 | 2,04× | oui |
| 6 | 0,1779 | 0,1247 | 0,70× | **PERDUE** |
| 16 | 0,2635 | 0,1515 | 0,58× | **PERDUE** |
| 41 | 0,4057 | 0,2679 | 0,66× | **PERDUE** |

**Le seuil est multiplié par 7,3 entre 1 et 41 bactéries.** La même bactérie, au même
endroit, avec le même signal, bascule de détectée à perdue **dès qu'il y en a 6 dans le
volume**. Mécanisme à rétroaction positive : plus le milieu est dense, plus le seuil
monte, plus les objets faibles disparaissent.

### 1.4 Un estimateur robuste ne suffit pas

Remplacer `moyenne + 10σ` par `médiane + 15·MAD` :

| nb bactéries | moy + 10σ | témoin | méd + 15·MAD | témoin |
|---:|---:|:--|---:|:--|
| 1 | 0,0554 | oui | 0,0162 | oui |
| 6 | 0,1779 | PERDUE | 0,1118 | **oui** |
| 16 | 0,2635 | PERDUE | 0,2130 | PERDUE |
| 41 | 0,4057 | PERDUE | 0,3502 | PERDUE |

Un palier gagné, pas plus. **La raison est physique, pas statistique** : le fond monte
réellement. Avec une réponse axiale de 12 µm de large plus les franges, les zones
« vides » sont remplies de diffraction hors foyer venue des voisines. La médiane ne
corrige pas un fond qui augmente pour de bonnes raisons.

> **Aucun seuil global, si robuste soit-il, ne peut fonctionner en milieu dense.**
> Ce n'est pas un problème d'estimateur, c'est une impossibilité de principe.

### 1.5 La réponse est très anisotrope

Largeurs à mi-hauteur du volume de focus, bactérie isolée couchée :

| axe | largeur à mi-hauteur |
|---|---:|
| X | 3,85 µm |
| Y (axe de la bactérie) | 3,30 µm |
| **Z** | **12,00 µm** |

Anisotropie **3,1×**. C'est l'axe Z qui commande les fusions : deux bactéries proches
axialement ont des réponses largement superposées, et CCL3D les relie en une seule
composante. Vérifié séparément : le pic du focus passe de 1,01× (écart 20 µm) à
**1,81×** (écart 1 µm) pour deux bactéries empilées selon z, par addition cohérente des
amplitudes diffusées. Latéralement, l'effet est nul.

---

## 2. À faire, par ordre de gain

### 2.1 Remplacer le seuil global par un seuil local — les deux objections ont une réponse

Objection « ça coûte trop cher » → calculer moyenne et écart-type sur un volume
**décimé** (facteur 8 en XY, 4 en Z), puis ré-interpoler en trilinéaire. Le fond varie
lentement, rien n'est perdu, et le coût est divisé par ~256.

Objection « ça crée des faux positifs dans les zones vides » → c'est le point clé, et le
remède tient en un `max` :

```python
T(x) = moyenne_locale(x) + k * max(ecart_type_local(x), sigma_plancher)
```

`sigma_plancher` est calibré **une fois** sur un volume sans objet. Sans ce garde-fou,
on divise par un écart-type minuscule dans les régions vraiment vides et le bruit
explose ; avec lui, le seuil local est stable.

C'est aussi le bon choix physiquement : près d'un amas le fond est élevé et le seuil
monte **localement** ; à l'écart il reste bas, et une bactérie isolée faible reste
détectable. Le seuil global, lui, pénalise tout le volume pour une densité locale.

### 2.2 Détecter par maxima locaux plutôt que par seuil + composantes connexes

Pour localiser, il ne faut pas une segmentation mais **un point par bactérie**.

```python
from cupyx.scipy.ndimage import maximum_filter
# empreinte a la taille de la resolution mesuree :
#   XY 3,85 um / 0,1375 um ≈ 28 px      Z 12 um / 0,5 um ≈ 24 plans
vmax = maximum_filter(focus_vol, size=(24, 28, 28))
candidats = (focus_vol == vmax) & (focus_vol > seuil_local)
```

Un maximum par objet, que la bactérie soit couchée ou debout. On classe ensuite par
**proéminence** du maximum, pas par volume — immun à l'amplification en puissance 3.
Deux objets proches restent séparés tant qu'il subsiste un creux entre eux, là où CCL3D
fusionne.

### 2.3 Remplacer `nb_pix` par la proéminence dans `CCA_CUDA_float` et `CCL_filter`

Tant que le descripteur est un volume au-dessus d'un seuil, il reste ininterprétable et
biaisé par l'orientation. Descripteurs recommandés : valeur du pic, proéminence
(pic − col le plus bas menant à un pic plus haut), et intensité intégrée au-dessus du
fond local.

### 2.4 Compenser l'anisotropie axiale avant de seuiller

La réponse fait 3,85 µm en XY contre 12 µm en Z. Une corrélation avec la réponse
attendue (filtre adapté allongé en z), ou une normalisation par l'enveloppe axiale,
resserre la dimension où se produisent les fusions.

### 2.5 Ne pas partir sur un montage hors-axe en espérant régler le problème

L'off-axis supprime l'image jumelle et l'ordre zéro. Mais les deux mécanismes mesurés
ici y survivraient tous les deux :

- l'effet d'orientation est projectif, le trajet optique ne change pas ;
- l'anisotropie axiale est fixée par l'ouverture numérique, pas par la géométrie
  inline/off-axis.

L'image jumelle n'est **pas** le moteur des « grosses patates ».

---

## 3. Performance de référence à préserver

Le pipeline classique segmente un hologramme 1024×1024 en **≈ 400 ms** sur GPU standard.
Mesures faites sur Quadro RTX 3000 (6 Go) pour situer les coûts :

| étape | temps |
|---|---:|
| rétro-propagation 1024×1024×200 | 111 ms |
| rétro-propagation 1024×1024×100 | 53 ms |
| rétro-propagation 1024×1024×50 | 24,5 ms |

**Le volume est massivement sur-échantillonné en Z** : le pas est de 0,5 µm pour une
résolution axiale mesurée à 12 µm, soit un facteur 24. Passer à un pas de 2 µm
(50 plans au lieu de 200) divise par 4 le coût de toute la chaîne aval sans perdre
d'information de résolution — seulement un peu de précision de centroïde. C'est le
levier le plus simple, et il vaut pour le pipeline classique comme pour l'IA.

---

## 4. Comment reproduire ces mesures

Les scripts de mesure ont été écrits hors dépôt (scratchpad de session) et s'appuient
uniquement sur `libs/` :

| mesure | protocole |
|---|---|
| effet d'orientation | une bactérie, φ ∈ {0…90}, relever `focus.max()` et le trajet projeté `mask.sum(axis=2).max()·dz` |
| amplification en voxels | même balayage, seuil **figé** calibré sur le cas φ = 90°, compter `bin_volume.sum()` et la plus grosse composante |
| couplage seuil/densité | une bactérie témoin fixe + N bactéries aléatoires, comparer `calc_threshold(vol, 10)` au pic local du témoin |
| anisotropie | bactérie isolée, largeurs à mi-hauteur des profils passant par l'argmax du volume de focus |
| addition cohérente | deux bactéries à écart variable selon X puis Z, relever `focus.max()` rapporté au cas d'une seule |

**Écueil rencontré** : analyser un profil 1D du volume reconstruit pour trouver la
distance de fusion ne marche pas — les franges de diffraction produisent 11 à 13 maxima
locaux et l'argmax saute d'un lobe à l'autre. Il faut calibrer le seuil sur le bruit,
filtrer les composantes par taille et apparier les blobs aux positions vraies, plutôt
que de compter des pics.

**Écueil d'environnement** : sur cette machine, `cupy.median` et `cupy.bincount`
déclenchent une compilation `nvcc` qui échoue (version de MSVC non supportée par
CUDA 11.8). Contourner en ramenant un sous-échantillon sur CPU avec `numpy`. Les noyaux
`jit.rawkernel` du dépôt ne sont pas affectés.
