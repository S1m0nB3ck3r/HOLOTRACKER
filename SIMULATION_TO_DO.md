# SIMULATION_TO_DO.md — une mesure à faire avant que HOLOTRACKER construise quoi que ce soit

Demande adressée au projet **`Simu-Bacteria-Holograms`**, en réponse à
`CLAUDE_HOLOTRACKER_TO_DO.md` du 2026-10-13. Document autoportant : il contient les chiffres
dont il parle et les conventions d'HOLOTRACKER, pour être traité sans accès à la conversation
où il a été écrit.

Rédigé le 2026-10-13 côté HOLOTRACKER.

---

## 1. Ce qui a été retenu de vos mesures, et appliqué

`SUM_OF_VARIANCE` est **désormais le critère de focus par défaut d'HOLOTRACKER**
(`core.__init__`), avec `K = 20`. Votre recommandation est suivie sans réserve : le critère
existait déjà dans le code, il n'était pas le défaut, et il n'est pas plus coûteux.

Vérifié ici, sur la première image réelle de `Film REF` (1024², 200 plans de 0,5 µm,
filtre 15–125, pixel caméra 7 µm, grossissement ×40) :

| critère | `sum_size` | focus | CCL | CCA |
|---|---:|---:|---:|---:|
| TENEGRAD | 15 | 383 ms | 87 ms | 13 ms |
| SUM_OF_VARIANCE | 11 | **253 ms** | 49 ms | 7 ms |
| SUM_OF_VARIANCE | 15 | 371 ms | 29 ms | 7 ms |

Et votre point 4 (« le K dépend du critère ») est confirmé sur données réelles, plus
nettement que par la simulation — nombre d'objets détectés sur cette même image :

| critère | K = 14 | K = 20 | K = 25 | K = 30 | K = 40 |
|---|---:|---:|---:|---:|---:|
| TENEGRAD, `sum_size` 15 | **25** | | | | |
| SUM_OF_VARIANCE, `sum_size` 15 | 50 | **38** | 19 | 13 | 10 |
| SUM_OF_VARIANCE, `sum_size` 11 | | 49 | | 25 | 12 |

À K égal, le changement de critère multiplie par deux le nombre de détections. K = 20 a été
retenu comme défaut : c'est le haut de votre fourchette mesurée (10–20), et il donne un
nombre de détections du même ordre que l'ancien défaut.

Votre table de temps du § 3 est également confirmée : rétro-propagation 1024² × 200 plans
en **121 ms à chaud** contre 111 ms annoncés. Attention toutefois, le premier appel coûte
429 ms, le temps que CuPy compile le noyau et construise les plans FFT : un coût mesuré sur
un seul hologramme est faux d'un facteur 3.

---

## 2. La mesure demandée : la table croisée critère × méthode × densité

### Le trou

Votre § 0.2 propose d'ajouter à l'interface d'HOLOTRACKER un mode de seuillage à trois
branches — global, local, maxima locaux — avec maille Z, maille XY, écrêtage κ, nombre de
passes, σ plancher, boîte XY et boîte Z. Six paramètres nouveaux, deux chemins de calcul
supplémentaires, un `nb_pix` qui change de sens, une superposition « segmentation » qui n'en
a plus, un seuil affiché qui devient une carte.

Or la table du § 2.2 qui justifie ce développement est mesurée **avec TENEGRAD**, le critère
que le même document recommande d'abandonner. Et votre § 0.1 point 2 dit, pour la seule
densité N = 200 :

> Avec SUM_OF_VARIANCE, les trois méthodes se valent (0,90 à 0,92), et les maxima locaux
> sont seuls à 100 % de précision.

Si cela vaut aussi aux autres densités, **il n'y a rien à ajouter dans l'interface** : le
critère fait le travail, et le mode de seuillage ne décide plus de rien. Si l'écart revient
à faible densité, alors on saura quoi construire — et seulement cela.

C'est la seule mesure qui manque pour trancher, et elle ne demande pas de code nouveau :
les deux implémentations de référence existent déjà dans votre `localisation_pipeline/`.

### Ce qui est demandé

Le benchmark de densité de `FOCUS_REPORT/DENSE`, **relancé avec SUM_OF_VARIANCE**, aux
quatre densités déjà employées :

| | N = 10 | N = 50 | N = 100 | N = 200 |
|---|---|---|---|---|
| seuil global | ? | ? | ? | 0,90–0,92 (donné) |
| seuil local + CCL3D | ? | ? | ? | 0,90–0,92 (donné) |
| maxima locaux | ? | ? | ? | 0,90–0,92 (donné) |

Pour chaque case, les trois mêmes chiffres que dans votre table du § 2.2 :

1. **rappel**,
2. **précision**,
3. **rappel des bactéries couchées** (φ > 60°), qui est le vrai sujet.

Avec, pour chaque case, le **K retenu** et comment il a été choisi (F1 maximal, comme
précédemment). Un K par case est attendu, pas un K unique : c'est ce que dit votre point 4.

La colonne TENEGRAD existe déjà ; il n'est pas nécessaire de la refaire, mais la garder côte
à côte dans la même table rendrait la comparaison lisible d'un coup d'œil.

### Conditions, pour que ce soit comparable

Identiques à celles du 2026-10-13 : bactéries 1,5–2 × 0,8 µm, n = 1,335 dans l'eau,
λ = 660 nm, ×40, 1024², 200 plans de 0,5 µm, filtre 15–250, bruit capteur 1,5 %, nettoyage
par division, 3 hologrammes par densité, tolérances 3 µm en XY et 8 µm en Z.

**Pixel caméra : 7 µm** (0,175 µm côté objet), ce qui est déjà votre géométrie. C'est la
valeur retenue côté HOLOTRACKER — la question 5,5 / 7 µm est tranchée, le défaut du code est
passé à 7 µm. Rien à changer de votre côté sur ce point.

### Ce qu'on en fera, pour que l'effort ne soit pas perdu

- **si les trois méthodes se valent à toutes les densités** : rien n'est ajouté à
  l'interface, le seuil global reste seul, et les maxima locaux sont écrits comme **script
  hors interface** à côté de `code_locate/script_localisation_pipeline.py`, pour qui veut
  les essayer. Fin du chantier ;
- **si l'écart revient à faible densité** (le N = 10 de votre table TENEGRAD montre 0,80
  contre 0,97, soit 17 points) : on construit **la seule méthode qui gagne**, pas un menu à
  trois branches ;
- **si les maxima locaux gardent leur avantage de précision partout** : ils deviennent un
  second mode, et nous reviendrons vous demander les tailles de boîte (voir § 3).

---

## 3. Trois demandes secondaires

### 3.1 Les mailles et les boîtes en µm, pas en pixels

Vos mailles (64 px en XY, 25 plans en Z) et votre boîte de maxima (23 px, 25 plans) sont
exprimées dans l'échantillonnage de vos simulations. Elles ne se transportent pas :
HOLOTRACKER tourne avec des pas en Z de 0,2 à 2 µm selon l'usage, et la taille de pixel est
un paramètre de l'interface.

Donnez-les **en micromètres** (la boîte de maxima l'est déjà : 3,85 × 12 µm), avec la
conversion que vous avez utilisée. HOLOTRACKER les reconvertira en pixels et en plans à
partir de ses propres paramètres.

### 3.2 La boîte de maxima à la taille de la réponse réaliste

Votre § 1.5 mesure la réponse à 2,1 µm en XY et 5,3 µm en Z avec des bactéries réalistes,
mais la boîte du filtre max est restée à 3,85 × 12 µm, dimensionnée en 2026-09 sur une
bactérie de 3 × 1 µm — soit deux fois trop grande dans les deux directions. Votre § 0
l'inscrit déjà en « à tester ». C'est utile : une boîte trop grande fusionne deux objets
proches en Z, ce qui est précisément le mode d'échec que les maxima locaux sont censés
éviter.

Si vous relancez le benchmark du § 2, le faire avec les deux tailles de boîte double le
nombre de cas ; à vous de voir. La taille réaliste seule nous intéresse plus que l'ancienne.

### 3.3 Le pas en Z de 2 µm : mesurer ce qu'il coûte en précision

Votre § 3 propose de passer le pas de 0,5 à 2 µm — un facteur 4 sur toute la chaîne aval,
que nous avons confirmé (propagation 121 → 32 ms, focus 320 → 54 ms pour 200 → 50 plans).
Vous écrivez « sans perdre d'information de résolution, seulement un peu de précision de
centroïde ».

C'est cette précision qui nous intéresse, parce que c'est elle que consomme **HoloTracker
Link** pour chaîner les trajectoires : une erreur de centroïde en Z supérieure au
déplacement entre deux images casse l'appariement. Vous avez la vérité terrain, nous non.

Demande : **l'erreur de localisation (biais et écart-type, en XY et en Z) en fonction du pas
en Z** — 0,2 / 0,5 / 1 / 2 µm, à densité moyenne, avec SUM_OF_VARIANCE. Un tableau de
quatre lignes suffit.

---

## 4. Ce qui n'est pas demandé, et pourquoi

- **La chaîne CLEAN (votre § 2.6).** 7,9 s par image contre 0,5 s, un autre score de
  détection, un autre modèle de soustraction, et son propre tracking. C'est un pipeline
  concurrent, pas une option du nôtre : il n'entrera pas dans HOLOTRACKER. Vos neuf points
  de critique sont pertinents et méritent d'être traités — chez vous. Si la chaîne devient
  meilleure, elle fera l'objet d'un autre logiciel, pas d'un mode de celui-ci.
- **Le remplacement de `nb_pix` par la proéminence (votre § 2.3).** On le fera, mais en
  **ajoutant** une colonne plutôt qu'en remplaçant : `nb_pix` alimente le filtre min/max
  voxel, le tableau de l'interface et tous les CSV déjà produits. À noter pour vous :
  `code_link` sélectionne ses colonnes **par nom**, donc une colonne de plus ne casse rien
  en aval. Aucune mesure n'est attendue de votre part là-dessus.
- **La compensation de l'anisotropie axiale (votre § 2.4).** Intéressante, et la moins
  spécifiée. À reprendre quand le reste sera tranché.
- **HoloPy.** Le chantier de validation par hologrammes exacts
  (`HOLOPY_test/HOLOPY_TO_DO.md`) est abandonné côté HOLOTRACKER.

---

## 5. Conventions d'HOLOTRACKER, pour mémoire

Utiles si vous produisez des fichiers que ce projet doit relire :

- volumes ordonnés **(Z, Y, X)**, distances en **mètres** partout, y compris dans le script
  hors interface ;
- `baryX` est l'indice de **colonne** × dx, `baryY` l'indice de **ligne** × dy
  (`device_CCA` dans `CCL3D.py`). Si votre simulateur transpose x et y, c'est ici que ça se
  voit ;
- CSV de sortie : `HOLOGRAM NUMBER, OBJECT NUMBER, X POSITION (m), Y POSITION (m),
  Z POSITION (m), NUMBER OF VOXEL` ;
- le seuil global est `moyenne + K·écart-type` sur le **volume entier** (`calc_threshold`),
  calculé sur le premier hologramme du lot ou sur chacun, au choix de l'utilisateur. Le
  choix « premier hologramme » est **délibéré** : il rend les comptages comparables d'une
  image à l'autre ;
- une limite `MAX_OBJECT_DETECTION = 500` existe encore et vide silencieusement un
  hologramme au-delà, en mode lot. Son sort n'est pas tranché. Si vous proposez des maxima
  locaux à K = 4–6, dites combien de candidats cela produit : c'est ce nombre qui décidera.

---

## 6. Résumé en trois lignes

1. **La mesure qui débloque tout** : la table du § 2, SUM_OF_VARIANCE × trois méthodes ×
   quatre densités, avec rappel, précision et rappel des couchées. Rien d'autre n'est
   nécessaire pour décider de l'interface d'HOLOTRACKER.
2. Puis, si et seulement si un écart subsiste : les tailles en µm (§ 3.1) et la boîte
   réaliste (§ 3.2).
3. Et, indépendamment : l'erreur de localisation en fonction du pas en Z (§ 3.3), qui
   commande le facteur 4 de performance.
