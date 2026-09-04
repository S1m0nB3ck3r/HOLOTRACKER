# Tests

## Lancer les tests

```bash
python tests/test_mean_hologram.py   # l'hologramme moyen (justesse, sans GPU)
python tests/test_pipeline.py        # la detection: pipeline Locate
python tests/test_link.py            # la couture: chaine Locate -> Link
```

Les deux derniers demandent un GPU CUDA; le premier non.

Aucune dépendance en plus de celles du projet. `pytest tests/test_pipeline.py` fonctionne
aussi si pytest est installé (il ne l'est pas actuellement sur cette machine).

Durée : quelques secondes, un GPU CUDA est nécessaire.

## Ce que le test fait

Il rejoue le pipeline complet de localisation sur la **première image de
`Film REF/cleaned/`**, avec des paramètres figés, et compare neuf familles de grandeurs à
un étalon enregistré :

| Contrôle | Tolérance |
|---|---|
| nombre d'objets détectés | exact |
| seuil de détection | 10⁻⁶ en relatif |
| position 3D de chaque objet | 1 nm |
| taille en voxels de chaque objet | exact |
| statistiques de l'hologramme nettoyé | 10⁻⁶ en relatif |
| statistiques de l'hologramme filtré | 10⁻⁶ en relatif |
| statistiques du volume propagé | 10⁻⁶ en relatif |
| statistiques du volume de focus | 10⁻⁶ en relatif |
| les 48 images d'affichage | empreinte exacte |

Pour les quatre dernières, sont comparés `min`, `max`, `mean`, `std`, `sum` **et
l'histogramme** en 40 classes (tolérance : 0,1 % des valeurs changeant de classe).

### Couverture de l'affichage

Les **12 modes d'affichage × 4 superpositions = 48 images** sont produites et comparées par
empreinte SHA-256 du tableau de pixels. Aucune image n'est stockée dans le dépôt.

L'empreinte est **exacte, sans tolérance** : mesuré sur deux exécutions consécutives, les
48 images sont identiques au bit près. Le bruit d'accumulation des barycentres (~10⁻¹⁰ m)
se perd dans l'arrondi au pixel, le rendu n'en dépend pas.

Cette couverture existe pour rendre sûre la séparation du calcul et de l'affichage dans
`core.py` : sans elle, déplacer les 14 méthodes de rendu se ferait à l'aveugle.

Ce que la matrice des empreintes apprend au passage : les superpositions (centroïdes,
segmentation) s'appliquent à tous les modes **sauf les deux vues FFT**, où elles n'auraient
pas de sens. Les projections passent par un chemin de code distinct
(`add_detection_markers_to_image_2d_projection`) des vues plan par plan.

### Appariement des objets

Les objets ne sont pas comparés rang par rang mais **appariés au plus proche voisin** :
deux objets très voisins pourraient permuter dans le tri sous l'effet du bruit
d'accumulation. Le test exige en plus que l'appariement soit bijectif.

## Nature du test

C'est un test de **caractérisation** : il fige le comportement actuel, il ne démontre pas
que le résultat est physiquement juste. Son rôle est de signaler qu'une modification du
code a changé le résultat, pour qu'on décide si le changement est voulu. C'est ce qui
permet de refactoriser sans crainte.

Pour vérifier la justesse physique, il faudrait un hologramme de synthèse avec des billes
à des profondeurs connues. C'est un test complémentaire, à écrire.

## D'où viennent les tolérances

Elles sont calibrées sur **cinq exécutions consécutives du même code** (2026-09-01) :

| Grandeur | Variabilité mesurée |
|---|---|
| nombre d'objets, seuil, tailles en voxels | identiques au bit près |
| statistiques et histogrammes | identiques au bit près |
| positions 3D | **8,7 × 10⁻¹¹ m** d'écart maximal |

Seules les positions bougent, à cause de `jit.atomic_add` dans `device_CCA` : l'ordre
d'accumulation des voxels varie d'un lancement à l'autre et l'addition flottante n'est pas
associative. D'où une tolérance de 1 nm sur les positions — soit un ordre de grandeur de
marge, et à comparer à la taille du voxel : 137,5 nm en XY, 500 nm en Z.

## Le test de l'hologramme moyen: une JUSTESSE, pas une caracterisation

`test_mean_hologram.py` est le seul des trois a verifier que le resultat est *juste*, et
non seulement qu'il n'a pas change. Il fabrique des images de synthese dont la moyenne est
calculable a la main, et exige que le code la retrouve. Il ne depend d'aucun etalon
enregistre, ni d'aucune donnee du depot, ni d'un GPU.

| Controle | Ce qu'il attrape |
|---|---|
| moyenne 8 bits | une erreur de calcul sur le cas nominal |
| moyenne flottante | l'ecrasement en 8 bits d'images flottantes ou 16 bits |
| moyenne geometrique | une confusion entre moyenne logarithmique et geometrique |
| apercu bmp | un apercu tout noir pour une image hors de l'echelle 0-255 |
| accord des implementations | une divergence entre `core` et `pre_treatment` |
| rappel de progression | un rappel manquant ou desordonne |
| recadrage centre | un recadrage depuis un coin au lieu du centre |
| formats de sortie | une perte de precision sur .tif ou .npy, un apercu .bmp noir |
| messages d'erreur | un type de moyenne inconnu, une liste vide ou des tailles melangees passes en silence |

Le controle "moyenne flottante" correspond au defaut corrige le 2026-09-04: un
`.convert('L')` ramenait toute image a 0-255 avant la moyenne. Sur un TIF flottant entre
0 et 1 — la sortie du pre-traitement — l'hologramme moyen obtenu etait **integralement
nul**. Remettre ce `.convert('L')` fait echouer 4 des 6 controles.

## Le troisieme test: la chaine Locate -> Link

`test_link.py` couvre ce que `test_pipeline.py` ne voit pas: la **couture entre les deux
applications**. Locate ecrit un CSV, Link le relit, le relie et le reecrit. C'est la que
se trouvaient la plupart des defauts corriges.

Il fait tourner un vrai lot Locate sur 6 hologrammes, puis fait traverser le CSV produit
par Link. Six controles:

| Controle | Ce qu'il attrape |
|---|---|
| colonnes du CSV | une divergence de format entre les deux applications |
| lignes fantomes | un hologramme sans objet qui ecrirait une ligne a (0,0,0) |
| chargement | des lignes perdues, ou une identite de ligne alteree |
| liaison | un changement du nombre de trajectoires ou de leurs longueurs |
| min length | un filtre sans effet, ou non monotone |
| export | un appariement incomplet, ou de mauvaises colonnes de sortie |

Le controle des lignes fantomes execute **un second lot avec un seuil absurde** (500 sigma),
qui ne detecte rien. Sans ce cas, le defaut ne se manifesterait jamais: dans un lot normal,
toutes les images ont des objets. C'est une lecon de la mise au point — la premiere version
du test laissait passer la regression.

## Fichiers

```
tests/
  pipeline_reference.py     execution instrumentee du pipeline Locate
  link_reference.py         execution instrumentee de la chaine Locate -> Link
                            (chacun est utilise par SON test ET par la generation de
                             l'etalon, pour qu'ils ne puissent pas diverger)
  test_mean_hologram.py     test de justesse du calcul de l'hologramme moyen
                            (images de synthese, pas d'etalon, pas de GPU)
  test_pipeline.py          test de la detection
  test_link.py              test de la couture entre les deux applications
  make_reference.py         regeneration des DEUX etalons
  reference/
    params.json             parametres FIGES, communs aux deux tests
    expected.json           etalon de la detection
    expected_link.json      etalon de la chaine Locate -> Link
```

### Pourquoi des paramètres figés

`tests/reference/params.json` est volontairement séparé de `last_param.json`. Ce dernier
est réécrit par l'interface à chaque utilisation : un test qui le lirait changerait de
résultat sans qu'on touche au code. Cela s'est produit pendant la mise au point — une
mesure est passée de 110 à 75 objets simplement parce que `remove_mean` avait été coché
dans l'interface entre deux exécutions.

Ces paramètres reprennent ceux de `last_param.json` du 2026-09-01, avec trois
différences délibérées : le répertoire et le type d'image pointent sur `Film REF/cleaned`,
et **`remove_mean` est désactivé**. Les images de `cleaned/` sont déjà nettoyées et
normalisées (valeurs 0,09 à 0,75) alors que `mean_arith.tif` est à l'échelle du capteur
(51 à 193) : les diviser l'une par l'autre reviendrait à nettoyer deux fois.

## Quand le test échoue

Le message nomme le contrôle en défaut et donne l'écart. Deux cas :

1. **Régression involontaire** — corriger le code.
2. **Changement voulu et compris** — régénérer l'étalon :

```bash
python tests/make_reference.py
```

Cette commande demande confirmation avant d'écraser. Ne la lancer qu'après avoir compris
*pourquoi* le résultat change : écraser l'étalon sans raison revient à supprimer le test.

## Efficacité vérifiée

Un test qui ne tombe jamais ne sert à rien. Deux régressions ont été introduites
volontairement pour vérifier qu'il les détecte :

| Perturbation | Détectée par |
|---|---|
| accumulateur `cp.ndarray` non initialisé dans `CCA_CUDA_float` (le bug corrigé le 2026-09-01) | contrôle **objets** : appariement non bijectif |
| critère TENEGRAD multiplié par 1,0001 (+0,01 %) | contrôles **seuil** et **volume_focus** |
| rayon des marqueurs de centroïdes porté de 3 à 4 pixels | contrôle **affichage** : 8 images nommées, les autres contrôles intacts |
| retour des lignes fantômes dans le CSV batch (L-01) | `test_link` : « un lot sans aucune détection a écrit 6 ligne(s) » |
| retour du re-linkage et du `except` muet sur min length (K-02) | `test_link` : « min length = 16 : nb_trajectoires vaut 31 au lieu de 0 » |
| identité des lignes retirée de `link_df` (K-03) | `test_link` : « l'index des trajectoires ne correspond plus aux lignes du CSV » |
| retour du `.convert('L')` dans `core.compute_mean_hologram` | `test_mean_hologram` : 4 contrôles sur 6, dont « l'hologramme moyen est entièrement NUL » |

Le second cas est instructif : un facteur d'échelle uniforme laisse le nombre d'objets,
leurs positions et leurs tailles **inchangés** — seules les statistiques du volume le
révèlent. C'est précisément pourquoi le test ne se contente pas de compter les objets.

## Valeurs de référence actuelles

Image `manipe3.859alpr9.000000_cleaned.tif`, 1024 × 1024, 200 plans, TENEGRAD,
seuil 14 σ :

```
images d'affichage: 48 (12 modes × 4 superpositions)
objets détectés   : 70
seuil de détection: 0.296370521
tailles           : 2 à 4056 voxels
hologramme nettoyé: moyenne 0.422781   écart-type 0.0374947
hologramme filtré : moyenne 0.0223024  écart-type 0.0205532
volume propagé    : moyenne 0.0254895  écart-type 0.016434
volume de focus   : moyenne 0.0492641  écart-type 0.0176505
```
