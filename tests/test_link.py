# -*- coding: utf-8 -*-
"""
Filename: test_link.py

Description:
Test de non-régression de la chaîne complète HoloTracker Locate -> HoloTracker Link.

Là où test_pipeline.py couvre la détection, celui-ci couvre la COUTURE entre les deux
applications: Locate écrit un CSV, Link le relit, le relie et le réécrit. C'est à cette
jonction que se trouvaient la plupart des défauts corrigés:

  - des lignes 'aucun objet détecté' à (0,0,0) que Link prenait pour de vraies détections
    et reliait en une trajectoire fantôme immobile;
  - le filtre `min length` sans effet dès qu'il dépassait le nombre d'images;
  - l'identité des lignes perdue à l'export, les numéros de trajectoire atterrissant sur
    les mauvaises lignes du CSV.

Ce qui est vérifié:
  1. les colonnes écrites par Locate sont exactement celles attendues par Link
  2. aucune ligne fantôme (OBJECT NUMBER = 0) dans le CSV produit
  3. Link charge toutes les lignes, en conservant leur identité
  4. le nombre de trajectoires et les longueurs, pour plusieurs valeurs de min length
  5. min length est monotone, et la longueur minimale obtenue est bien celle demandée
  6. l'appariement à l'export est exact: autant de lignes numérotées que de points liés
  7. les colonnes de sortie: OBJECT NUMBER remplacée par TRAJECTORY NUMBER

Usage:
    python tests/test_link.py          (sans dépendance)
    pytest tests/test_link.py          (si pytest est installé)

Si un écart est voulu, régénérer les étalons:
    python tests/make_reference.py

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from link_reference import (run_chaine, EXPECTED_FILE, COLONNES_LOCATE,
                            MIN_LENGTHS, NB_HOLOGRAMMES)

COLONNES_SORTIE_ATTENDUES = ["HOLOGRAM NUMBER", "X POSITION (m)", "Y POSITION (m)",
                             "Z POSITION (m)", "NUMBER OF VOXEL", "TRAJECTORY NUMBER"]


def _load_expected():
    if not os.path.exists(EXPECTED_FILE):
        raise FileNotFoundError(
            "Étalon absent: {}\nLe générer avec: python tests/make_reference.py".format(
                EXPECTED_FILE))
    with open(EXPECTED_FILE, encoding="utf-8") as f:
        return json.load(f)


def collect():
    return run_chaine(), _load_expected()


# ---------------------------------------------------------------------------
# Vérifications
# ---------------------------------------------------------------------------

def check_colonnes_csv(actual, expected, log):
    """Les deux applications doivent s'accorder sur le format du fichier."""
    assert actual["csv_colonnes"] == COLONNES_LOCATE, (
        "Locate n'écrit pas les colonnes attendues par Link.\n"
        "  écrites : {}\n  attendues: {}".format(actual["csv_colonnes"], COLONNES_LOCATE))
    log("colonnes du CSV     : conformes ({} colonnes)".format(len(COLONNES_LOCATE)))


def check_lignes_fantomes(actual, expected, log):
    """Un hologramme sans détection ne doit rien écrire du tout.

    Deux contrôles complémentaires: aucune ligne 'OBJECT NUMBER = 0' dans un lot normal,
    et surtout un lot où RIEN n'est détecté doit produire un fichier réduit à son en-tête.
    Le second est indispensable: dans un lot normal toutes les images ont des objets, le
    défaut ne se manifesterait jamais.
    """
    assert actual["csv_lignes_fantomes"] == 0, (
        "{} ligne(s) 'OBJECT NUMBER = 0' dans le CSV: Link les prendrait pour de vraies "
        "détections et en ferait une trajectoire immobile à l'origine".format(
            actual["csv_lignes_fantomes"]))

    assert actual["sans_detection_nb_lignes"] == 0, (
        "un lot sans aucune détection a écrit {} ligne(s) au lieu d'aucune.\n"
        "  Ces lignes valent (0,0,0) et Link en ferait une trajectoire fantôme immobile "
        "à l'origine, présente sur toutes les images vides.".format(
            actual["sans_detection_nb_lignes"]))

    assert actual["sans_detection_colonnes"] == COLONNES_LOCATE, (
        "le CSV d'un lot sans détection n'a pas le bon en-tête: {}".format(
            actual["sans_detection_colonnes"]))

    log("lignes fantômes     : aucune, y compris sur un lot sans aucune détection")


def check_chargement(actual, expected, log):
    assert actual["csv_nb_lignes"] == expected["csv_nb_lignes"], (
        "le CSV produit contient {} lignes au lieu de {}".format(
            actual["csv_nb_lignes"], expected["csv_nb_lignes"]))
    assert actual["csv_nb_images"] == expected["csv_nb_images"], (
        "{} images au lieu de {}".format(actual["csv_nb_images"], expected["csv_nb_images"]))
    assert actual["charge_nb_lignes"] == actual["csv_nb_lignes"], (
        "Link n'a chargé que {} lignes sur {}".format(
            actual["charge_nb_lignes"], actual["csv_nb_lignes"]))
    assert actual["charge_index_conforme"], (
        "l'index du tableau chargé ne correspond plus aux lignes du CSV")
    log("Locate -> Link      : {} détections sur {} images, toutes chargées".format(
        actual["csv_nb_lignes"], actual["csv_nb_images"]))


def check_liaisons(actual, expected, log):
    """Nombre de trajectoires et longueurs, pour chaque valeur de min length."""
    for cle, att in expected["liaisons"].items():
        obt = actual["liaisons"].get(cle)
        assert obt is not None, "min length = {} absent du résultat".format(cle)
        for champ in ("nb_trajectoires", "nb_points", "longueur_min", "longueur_max"):
            assert obt[champ] == att[champ], (
                "min length = {}: {} vaut {} au lieu de {}".format(
                    cle, champ, obt[champ], att[champ]))
    resume = ", ".join(
        "ml={}:{}traj".format(k, actual["liaisons"][k]["nb_trajectoires"])
        for k in sorted(actual["liaisons"], key=int))
    log("liaison             : {}".format(resume))


def check_min_length(actual, expected, log):
    """min length doit filtrer réellement, et de façon monotone.

    C'est le contrôle qui aurait attrapé le défaut d'origine: le filtre était sans effet
    dès qu'il dépassait le nombre d'images du film, l'exception étant avalée.
    """
    liaisons = actual["liaisons"]
    precedent = None
    for ml in sorted((int(k) for k in liaisons), reverse=False):
        info = liaisons[str(ml)]
        if info["nb_trajectoires"] and ml > 0:
            assert info["longueur_min"] >= ml, (
                "min length = {} mais une trajectoire ne fait que {} points".format(
                    ml, info["longueur_min"]))
        if precedent is not None:
            assert info["nb_trajectoires"] <= precedent, (
                "min length croissant devrait donner moins de trajectoires: "
                "{} après {}".format(info["nb_trajectoires"], precedent))
        precedent = info["nb_trajectoires"]

    trop_grand = str(NB_HOLOGRAMMES + 10)
    assert liaisons[trop_grand]["nb_trajectoires"] == 0, (
        "min length = {} dépasse le nombre d'images ({}): aucune trajectoire ne peut "
        "l'atteindre, or {} sont renvoyées — le filtre est de nouveau sans effet".format(
            trop_grand, NB_HOLOGRAMMES, liaisons[trop_grand]["nb_trajectoires"]))
    log("filtre min length   : monotone, et sans effet de bord au-delà de {} images".format(
        NB_HOLOGRAMMES))


def check_export(actual, expected, log):
    """L'appariement doit être exact, sans recourir au repli par coordonnées."""
    for cle, info in actual["liaisons"].items():
        if info["nb_points"] == 0:
            continue
        assert info["index_conforme"], (
            "min length = {}: l'index des trajectoires ne correspond plus aux lignes "
            "du CSV d'origine".format(cle))
        assert info["nb_lignes_attribuees"] == info["nb_points"], (
            "min length = {}: {} lignes numérotées pour {} points liés — "
            "l'appariement est incomplet".format(
                cle, info["nb_lignes_attribuees"], info["nb_points"]))
        assert info["colonnes_sortie"] == COLONNES_SORTIE_ATTENDUES, (
            "min length = {}: colonnes de sortie inattendues\n  obtenues : {}\n"
            "  attendues: {}".format(cle, info["colonnes_sortie"],
                                     COLONNES_SORTIE_ATTENDUES))
    log("export              : appariement exact, OBJECT NUMBER remplacée par "
        "TRAJECTORY NUMBER")


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------

_CACHE = {}


def _pair():
    """La chaîne complète coûte quelques secondes: une seule exécution par session."""
    if "v" not in _CACHE:
        _CACHE["v"] = collect()
    return _CACHE["v"]


def _muet(_message):
    pass


def test_colonnes_csv():
    check_colonnes_csv(*_pair(), log=_muet)


def test_lignes_fantomes():
    check_lignes_fantomes(*_pair(), log=_muet)


def test_chargement():
    check_chargement(*_pair(), log=_muet)


def test_liaisons():
    check_liaisons(*_pair(), log=_muet)


def test_min_length():
    check_min_length(*_pair(), log=_muet)


def test_export():
    check_export(*_pair(), log=_muet)


def main():
    print("=" * 74)
    print("Test de non-régression de la chaîne Locate -> Link")
    print("=" * 74)
    actual, expected = collect()
    print("étalon généré le    : {}".format(
        expected.get("_meta", {}).get("genere_le", "inconnu")))

    lignes = []
    controles = [
        ("colonnes du CSV", check_colonnes_csv),
        ("lignes fantômes", check_lignes_fantomes),
        ("chargement", check_chargement),
        ("liaison", check_liaisons),
        ("min length", check_min_length),
        ("export", check_export),
    ]
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
