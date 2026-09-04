# -*- coding: utf-8 -*-
"""
Filename: test_pipeline.py

Description:
Test de non-régression du pipeline de localisation. Rejoue le pipeline complet sur la
première image de Film REF/cleaned, avec des paramètres figés, et compare au fichier
étalon tests/reference/expected.json.

C'est un test de CARACTÉRISATION: il fige le comportement actuel, il ne démontre pas que
le résultat est physiquement juste. Son rôle est de signaler qu'une modification du code
a changé le résultat, pour qu'on décide si le changement est voulu.

Ce qui est vérifié:
  1. nombre d'objets détectés (exact)
  2. seuil de détection
  3. position 3D et taille en voxels de chaque objet, appariés au plus proche
  4. statistiques de l'hologramme nettoyé, de l'hologramme filtré, du volume propagé
     et du volume de focus
  5. histogrammes de ces quatre grandeurs
  6. les 48 images d'affichage (12 modes x 4 superpositions), par empreinte exacte
  7. la valeur du pixel sous le curseur, pour chaque mode et quelques points
  8. les vignettes XY/XZ/YZ extraites autour du plus gros objet

Usage:
    python tests/test_pipeline.py          (sans dépendance)
    pytest tests/test_pipeline.py          (si pytest est installé)

Si un écart est voulu (changement d'algorithme assumé), régénérer l'étalon:
    python tests/make_reference.py

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pipeline_reference import run_pipeline, load_params, EXPECTED_FILE

# ---------------------------------------------------------------------------
# Tolérances, calibrées sur 5 exécutions consécutives du même code (2026-09-01):
#   - nombre d'objets, seuil, nb de voxels, statistiques et histogrammes: identiques
#     au bit près;
#   - positions: écart maximal 8,7e-11 m, dû à jit.atomic_add dans device_CCA, dont
#     l'ordre d'accumulation varie d'un lancement à l'autre (l'addition flottante
#     n'est pas associative).
# Les seuils ci-dessous laissent environ un ordre de grandeur de marge.
# ---------------------------------------------------------------------------
TOL_POSITION_M = 1e-9      # 1 nm, à comparer au voxel: 137,5 nm en XY, 500 nm en Z
TOL_RELATIVE = 1e-6        # seuil et statistiques
TOL_HIST_FRACTION = 1e-3   # écart total entre histogrammes, en fraction des valeurs

CHAMPS = ("holo_cleaned", "holo_filtered", "volume_propag", "volume_focus")

# Les images d'affichage sont comparées par empreinte EXACTE. Mesuré sur deux exécutions
# consécutives: les 48 images sont identiques au bit près. Une tolérance serait donc du
# laxisme, pas de la prudence — le rendu ne dépend pas du bruit d'accumulation, qui se
# perd dans l'arrondi au pixel.


def _load_expected():
    if not os.path.exists(EXPECTED_FILE):
        raise FileNotFoundError(
            "Étalon absent: {}\nLe générer avec: python tests/make_reference.py".format(
                EXPECTED_FILE))
    with open(EXPECTED_FILE, encoding="utf-8") as f:
        return json.load(f)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def collect():
    """Exécute le pipeline et renvoie (obtenu, attendu).

    Les bornes d'histogramme de l'étalon sont réimposées, sans quoi les classes se
    décaleraient et les effectifs ne seraient plus comparables.
    """
    expected = _load_expected()
    ranges = {champ: expected[champ]["hist_range"] for champ in CHAMPS}
    actual = run_pipeline(load_params(), ranges=ranges)
    return actual, expected


# ---------------------------------------------------------------------------
# Vérifications
# ---------------------------------------------------------------------------

def check_image(actual, expected, log):
    assert actual["image"] == expected["image"], (
        "image différente: {} au lieu de {}".format(actual["image"], expected["image"]))
    log("image analysée      : {}".format(actual["image"]))


def check_object_count(actual, expected, log):
    got, ref = actual["n_objects"], expected["n_objects"]
    assert got == ref, "nombre d'objets: {} au lieu de {}".format(got, ref)
    log("objets détectés     : {}".format(got))


def check_threshold(actual, expected, log):
    got, ref = actual["threshold"], expected["threshold"]
    ecart = _rel(got, ref)
    assert ecart <= TOL_RELATIVE, (
        "seuil de détection: {:.9f} au lieu de {:.9f} (écart relatif {:.2e})".format(
            got, ref, ecart))
    log("seuil de détection  : {:.9f}  (écart {:.1e})".format(got, ecart))


def check_objects(actual, expected, log):
    """Apparie chaque objet de l'étalon avec le plus proche des objets obtenus.

    L'appariement se fait par distance et non par rang dans la liste: deux objets très
    voisins pourraient permuter dans le tri sous l'effet du bruit d'accumulation.
    """
    ref = np.array([o[:3] for o in expected["objects"]], dtype=np.float64)
    got = np.array([o[:3] for o in actual["objects"]], dtype=np.float64)
    ref_vox = np.array([o[3] for o in expected["objects"]])
    got_vox = np.array([o[3] for o in actual["objects"]])

    d = np.linalg.norm(ref[:, None, :] - got[None, :, :], axis=2)
    plus_proche = d.argmin(axis=1)
    distances = d[np.arange(len(ref)), plus_proche]

    assert len(set(plus_proche.tolist())) == len(ref), (
        "l'appariement n'est pas bijectif: deux objets de l'étalon désignent le même "
        "objet détecté, les positions ont trop bougé")

    pire = int(distances.argmax())
    assert distances[pire] <= TOL_POSITION_M, (
        "objet le plus déplacé: {:.3f} nm (tolérance {:.1f} nm)\n"
        "  attendu {}\n  obtenu  {}".format(
            distances[pire] * 1e9, TOL_POSITION_M * 1e9, ref[pire], got[plus_proche[pire]]))

    mauvais = [(i, int(ref_vox[i]), int(got_vox[plus_proche[i]]))
               for i in range(len(ref)) if ref_vox[i] != got_vox[plus_proche[i]]]
    assert not mauvais, "taille en voxels différente pour {} objet(s): {}".format(
        len(mauvais), mauvais[:5])

    log("positions 3D        : écart max {:.4f} nm (tolérance {:.0f} nm)".format(
        distances.max() * 1e9, TOL_POSITION_M * 1e9))
    log("tailles en voxels   : identiques ({} à {} voxels)".format(
        int(ref_vox.min()), int(ref_vox.max())))


def check_display(actual, expected, log):
    """Compare les 48 images d'affichage à leurs empreintes de référence."""
    att = expected.get("display")
    obt = actual.get("display")
    assert att, ("l'étalon ne contient pas de section 'display': le régénérer avec "
                 "python tests/make_reference.py")

    manquantes = sorted(set(att) - set(obt))
    assert not manquantes, "combinaisons d'affichage absentes du résultat: {}".format(manquantes[:5])

    vides = sorted(k for k, v in obt.items() if v is None and att.get(k) is not None)
    assert not vides, "aucune image produite pour: {}".format(vides[:5])

    formes = sorted(k for k in att if att[k] and obt[k]
                    and (att[k]["shape"] != obt[k]["shape"] or att[k]["dtype"] != obt[k]["dtype"]))
    assert not formes, "forme ou type d'image différents pour: {}".format(
        [(k, att[k]["shape"], obt[k]["shape"]) for k in formes[:3]])

    differentes = sorted(k for k in att if att[k] and obt[k]
                         and att[k]["sha256"] != obt[k]["sha256"])
    assert not differentes, "{} image(s) sur {} ont changé:\n    {}".format(
        len(differentes), len(att), "\n    ".join(differentes[:8]))

    log("images d'affichage  : {} identiques ({} modes x {} superpositions)".format(
        len(att), len(set(k.split('|')[0] for k in att)), len(set(k.split('|')[1] for k in att))))


def check_pixel_values(actual, expected, log):
    """Compare les valeurs relevées par get_pixel_value, second chemin de rendu."""
    att = expected.get("pixel_values")
    obt = actual.get("pixel_values")
    assert att, ("l'étalon ne contient pas de section 'pixel_values': le régénérer avec "
                 "python tests/make_reference.py")

    manquants = sorted(set(att) - set(obt))
    assert not manquants, "relevés absents du résultat: {}".format(manquants[:5])

    ecarts = []
    for cle, ref in att.items():
        got = obt[cle]
        if ref is None or got is None:
            if ref != got:
                ecarts.append((cle, ref, got))
        elif _rel(got, ref) > TOL_RELATIVE:
            ecarts.append((cle, ref, got))
    assert not ecarts, "{} relevé(s) différent(s): {}".format(len(ecarts), ecarts[:5])

    non_nuls = sum(1 for v in att.values() if v)
    log("valeurs de pixel    : {} relevés identiques ({} non nuls)".format(len(att), non_nuls))


def check_object_slices(actual, expected, log):
    """Compare les vignettes extraites autour du plus gros objet."""
    att = expected.get("object_slices")
    obt = actual.get("object_slices")
    assert att, ("l'étalon ne contient pas de section 'object_slices': le régénérer avec "
                 "python tests/make_reference.py")
    assert obt is not None, "aucune vignette produite alors que l'étalon en contient"

    assert att["nb_voxels"] == obt["nb_voxels"], (
        "l'objet le plus gros a changé de taille: {} au lieu de {} voxels".format(
            obt["nb_voxels"], att["nb_voxels"]))

    for nom in ("xy_slice", "xz_slice", "yz_slice"):
        a, b = att[nom], obt[nom]
        assert a["shape"] == b["shape"], "{}: forme {} au lieu de {}".format(
            nom, b["shape"], a["shape"])
        assert a["sha256"] == b["sha256"], "{}: contenu différent".format(nom)

    log("vignettes d'objet   : XY, XZ, YZ identiques (objet de {} voxels)".format(
        att["nb_voxels"]))


def check_field(champ, actual, expected, log):
    a, e = actual[champ], expected[champ]
    pires = {k: _rel(a[k], e[k]) for k in ("min", "max", "mean", "std", "sum")}
    pire_nom = max(pires, key=pires.get)
    assert pires[pire_nom] <= TOL_RELATIVE, (
        "{}: statistique '{}' = {:.6g} au lieu de {:.6g} (écart relatif {:.2e})".format(
            champ, pire_nom, a[pire_nom], e[pire_nom], pires[pire_nom]))

    ha = np.array(a["hist"], dtype=np.int64)
    he = np.array(e["hist"], dtype=np.int64)
    total = int(he.sum())
    ecart = float(np.abs(ha - he).sum()) / max(total, 1)
    assert ecart <= TOL_HIST_FRACTION, (
        "{}: histogramme différent, {:.3f} % des valeurs ont changé de classe "
        "(tolérance {:.1f} %)".format(champ, ecart * 100, TOL_HIST_FRACTION * 100))

    log("{:20s}: moy={:.6g} ecart-type={:.6g} | stats {:.1e} | histogramme {:.4f} %".format(
        champ, a["mean"], a["std"], pires[pire_nom], ecart * 100))


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------

_CACHE = {}


def _pair():
    """Le pipeline coûte plusieurs secondes: on ne l'exécute qu'une fois par session."""
    if "v" not in _CACHE:
        _CACHE["v"] = collect()
    return _CACHE["v"]


def _muet(_message):
    pass


def test_image():
    check_image(*_pair(), log=_muet)


def test_object_count():
    check_object_count(*_pair(), log=_muet)


def test_threshold():
    check_threshold(*_pair(), log=_muet)


def test_objects():
    check_objects(*_pair(), log=_muet)


def test_display():
    check_display(*_pair(), log=_muet)


def test_pixel_values():
    check_pixel_values(*_pair(), log=_muet)


def test_object_slices():
    check_object_slices(*_pair(), log=_muet)


def test_holo_cleaned():
    check_field("holo_cleaned", *_pair(), log=_muet)


def test_holo_filtered():
    check_field("holo_filtered", *_pair(), log=_muet)


def test_volume_propag():
    check_field("volume_propag", *_pair(), log=_muet)


def test_volume_focus():
    check_field("volume_focus", *_pair(), log=_muet)


def main():
    print("=" * 74)
    print("Test de non-régression du pipeline de localisation")
    print("=" * 74)
    actual, expected = collect()
    print("étalon généré le    : {}".format(
        expected.get("_meta", {}).get("genere_le", "inconnu")))

    lignes = []
    controles = [
        ("image", check_image),
        ("nombre d'objets", check_object_count),
        ("seuil", check_threshold),
        ("objets", check_objects),
        ("affichage", check_display),
        ("valeurs de pixel", check_pixel_values),
        ("vignettes d'objet", check_object_slices),
    ] + [(champ, (lambda c: lambda a, e, log: check_field(c, a, e, log))(champ))
         for champ in CHAMPS]

    echecs = []
    for nom, fn in controles:
        try:
            fn(actual, expected, lignes.append)
        except AssertionError as err:
            echecs.append((nom, str(err)))

    print("-" * 74)
    for ligne in lignes:
        print("  " + ligne)
    print("-" * 74)
    if echecs:
        print("ÉCHEC — {} contrôle(s) en défaut sur {}:\n".format(len(echecs), len(controles)))
        for nom, err in echecs:
            print("  [{}]\n    {}\n".format(nom, err.replace("\n", "\n    ")))
        print("Si ce changement est voulu et compris, régénérer l'étalon:")
        print("    python tests/make_reference.py")
        return 1
    print("SUCCÈS — {} contrôles passés.".format(len(controles)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
