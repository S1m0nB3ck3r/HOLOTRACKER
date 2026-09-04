# -*- coding: utf-8 -*-
"""
Filename: test_mean_hologram.py

Description:
Test du calcul de l'hologramme moyen, sur des images de synthèse dont on connaît la
moyenne exacte.

Contrairement aux deux autres tests, celui-ci ne compare pas à un étalon enregistré: il
vérifie une **justesse**. La moyenne de N images synthétiques est calculable à la main,
le test exige que le code la retrouve.

Ce qui est vérifié:
  1. la moyenne arithmétique sur des images 8 bits vaut la moyenne exacte
  2. la moyenne arithmétique sur des images TIF FLOTTANTES vaut la moyenne exacte
     -> c'est le défaut corrigé le 2026-09-04: un .convert('L') ramenait toute image à
        0-255 avant la moyenne, et un TIF flottant entre 0 et 1 donnait un hologramme
        moyen intégralement NUL
  3. la moyenne logarithmique est bien la moyenne géométrique
  4. l'aperçu .bmp n'est pas noir quand l'image n'est pas à l'échelle 0-255
  5. les deux implémentations du projet (core et pre_treatment) donnent le même résultat
     sur les mêmes données

Ne nécessite pas de GPU.

Usage:
    python tests/test_mean_hologram.py
    pytest tests/test_mean_hologram.py

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import os
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code_locate"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))

import numpy as np
from PIL import Image

from core import HoloTrackerCore
import traitement_holo
import pre_treatment

TAILLE = (16, 24)          # (lignes, colonnes), volontairement non carré
NB_IMAGES = 5
TOLERANCE = 1e-5           # les images sont enregistrées en float32


def _images_synthetiques(rng, echelle):
    """N images dont on connaîtra la moyenne exacte, à l'échelle demandée."""
    return [(rng.random(TAILLE) * echelle).astype(np.float64) for _ in range(NB_IMAGES)]


def _ecrire(images, repertoire, mode):
    """Écrit les images en TIF flottant ('F') ou en BMP 8 bits ('L')."""
    for i, img in enumerate(images):
        chemin = os.path.join(repertoire, "img_{:03d}.{}".format(i, "tif" if mode == "F" else "bmp"))
        if mode == "F":
            Image.fromarray(img.astype(np.float32), mode="F").save(chemin)
        else:
            Image.fromarray(img.astype(np.uint8), mode="L").save(chemin)


def _moyenne_core(repertoire, image_type, mean_type):
    """Appelle le calcul du bouton 'Mean hologram computation' et relit son résultat."""
    core = HoloTrackerCore()
    chemin_tif = core.compute_mean_hologram(repertoire, image_type, mean_type=mean_type)
    return np.asarray(Image.open(chemin_tif), dtype=np.float64), chemin_tif


# ---------------------------------------------------------------------------

def check_moyenne_8bits(log):
    """Cas nominal: images 8 bits, moyenne arithmétique."""
    rng = np.random.default_rng(0)
    images = [np.floor(i).astype(np.float64) for i in _images_synthetiques(rng, 200.0)]
    rep = tempfile.mkdtemp(prefix="mean_8bits_")
    try:
        _ecrire(images, rep, "L")
        obtenue, _ = _moyenne_core(rep, "BMP", "arithmetic")
        attendue = np.mean(images, axis=0)
        ecart = float(np.max(np.abs(obtenue - attendue)))
        assert ecart <= TOLERANCE, (
            "moyenne 8 bits fausse: écart max {:.3g} (tolérance {:.0e})".format(ecart, TOLERANCE))
        log("images 8 bits       : moyenne exacte (écart {:.1e})".format(ecart))
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_moyenne_flottante(log):
    """Le défaut corrigé: un TIF flottant entre 0 et 1 donnait une moyenne nulle."""
    rng = np.random.default_rng(1)
    images = _images_synthetiques(rng, 1.0)          # valeurs entre 0 et 1
    rep = tempfile.mkdtemp(prefix="mean_float_")
    try:
        _ecrire(images, rep, "F")
        obtenue, _ = _moyenne_core(rep, "TIF", "arithmetic")

        assert obtenue.max() > 0, (
            "l'hologramme moyen est entièrement NUL: les images flottantes ont été "
            "écrasées en 8 bits avant la moyenne")

        attendue = np.mean([i.astype(np.float32).astype(np.float64) for i in images], axis=0)
        ecart = float(np.max(np.abs(obtenue - attendue)))
        assert ecart <= TOLERANCE, (
            "moyenne flottante fausse: écart max {:.3g} (tolérance {:.0e})".format(
                ecart, TOLERANCE))
        log("images flottantes   : moyenne exacte, dynamique conservée "
            "(max {:.4f}, écart {:.1e})".format(float(obtenue.max()), ecart))
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_moyenne_geometrique(log):
    """La moyenne 'logarithmique' doit être la moyenne géométrique."""
    rng = np.random.default_rng(2)
    images = _images_synthetiques(rng, 100.0)
    images = [np.clip(i, 1.0, None) for i in images]   # pas de zéro: log défini
    rep = tempfile.mkdtemp(prefix="mean_geo_")
    try:
        _ecrire(images, rep, "F")
        obtenue, _ = _moyenne_core(rep, "TIF", "logarithmic")
        pile = np.stack([i.astype(np.float32).astype(np.float64) for i in images])
        attendue = np.exp(np.mean(np.log(pile), axis=0))     # moyenne géométrique
        ecart = float(np.max(np.abs(obtenue - attendue) / attendue))
        assert ecart <= 1e-4, (
            "la moyenne logarithmique ne vaut pas la moyenne géométrique: "
            "écart relatif max {:.3g}".format(ecart))
        # et elle doit être inférieure à la moyenne arithmétique (inégalité des moyennes)
        arithmetique = pile.mean(axis=0)
        assert (obtenue <= arithmetique + 1e-6).all(), (
            "la moyenne géométrique devrait être inférieure ou égale à l'arithmétique")
        log("moyenne logarithmique: c'est bien la moyenne géométrique "
            "(écart relatif {:.1e})".format(ecart))
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_apercu_bmp(log):
    """L'aperçu .bmp doit rester lisible pour une image hors de l'échelle 0-255."""
    rng = np.random.default_rng(3)
    images = _images_synthetiques(rng, 1.0)
    rep = tempfile.mkdtemp(prefix="mean_bmp_")
    try:
        _ecrire(images, rep, "F")
        _, chemin_tif = _moyenne_core(rep, "TIF", "arithmetic")
        chemin_bmp = chemin_tif.replace(".tif", ".bmp")
        assert os.path.isfile(chemin_bmp), "aucun aperçu .bmp produit"
        apercu = np.asarray(Image.open(chemin_bmp))
        assert apercu.max() > 0, (
            "l'aperçu .bmp est entièrement noir: il faut normaliser sur la dynamique "
            "réelle, pas seulement tronquer à [0, 255]")
        log("aperçu .bmp         : lisible (niveaux {} à {})".format(
            int(apercu.min()), int(apercu.max())))
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_accord_entre_implementations(log):
    """core.compute_mean_hologram et pre_treatment.compute_mean_image doivent s'accorder."""
    rng = np.random.default_rng(4)
    images = _images_synthetiques(rng, 1.0)
    rep = tempfile.mkdtemp(prefix="mean_accord_")
    try:
        _ecrire(images, rep, "F")
        par_core, _ = _moyenne_core(rep, "TIF", "arithmetic")
        chemins = pre_treatment.list_images(rep, "tif")
        par_pre_treatment, _, _ = pre_treatment.compute_mean_image(chemins, "arithmetic")
        ecart = float(np.max(np.abs(par_core - par_pre_treatment.astype(np.float64))))
        assert ecart <= TOLERANCE, (
            "les deux implémentations du projet ne donnent pas la même moyenne: "
            "écart max {:.3g}".format(ecart))
        log("accord des 2 codes  : core et pre_treatment concordent (écart {:.1e})".format(ecart))
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_recadrage(log):
    """L'option 'taille' doit recadrer au CENTRE, pas depuis un coin."""
    grande = np.arange(20 * 30, dtype=np.float64).reshape(20, 30)
    rep = tempfile.mkdtemp(prefix="mean_crop_")
    try:
        chemin = os.path.join(rep, "g.tif")
        Image.fromarray(grande.astype(np.float32), mode="F").save(chemin)
        moyenne, _, _ = traitement_holo.calc_holo_moyen([chemin], taille=(10, 8))
        attendu = grande[(20 - 8) // 2:(20 - 8) // 2 + 8, (30 - 10) // 2:(30 - 10) // 2 + 10]
        assert moyenne.shape == (8, 10), "forme apres recadrage: {}".format(moyenne.shape)
        assert np.allclose(moyenne, attendu), "le recadrage n'est pas centre"
        log("recadrage centre    : (20, 30) -> (8, 10), au bon endroit")
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_formats_sortie(log):
    """Le format du fichier ecrit doit venir de son extension, sans perte pour tif/npy."""
    rng = np.random.default_rng(6)
    images = _images_synthetiques(rng, 1.0)
    rep = tempfile.mkdtemp(prefix="mean_formats_")
    try:
        _ecrire(images, rep, "F")
        chemins = traitement_holo.lister_images(rep, "tif")
        sorties = [os.path.join(rep, "m." + e) for e in ("tif", "npy", "bmp")]
        moyenne, _, _ = traitement_holo.calc_holo_moyen(chemins, fichiers_sortie=sorties)

        for chemin in sorties:
            assert os.path.isfile(chemin), "fichier non ecrit: {}".format(chemin)

        relu_tif = np.asarray(Image.open(sorties[0]), dtype=np.float64)
        assert np.allclose(relu_tif, moyenne), "le .tif ne restitue pas la moyenne"
        relu_npy = np.load(sorties[1])
        assert np.allclose(relu_npy, moyenne), "le .npy ne restitue pas la moyenne"
        apercu = np.asarray(Image.open(sorties[2]))
        assert apercu.dtype == np.uint8 and apercu.max() == 255, (
            "l'apercu .bmp devrait etre normalise sur 0-255, obtenu max={}".format(apercu.max()))
        log("formats de sortie   : .tif et .npy sans perte, .bmp normalise")
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_erreurs_explicites(log):
    """Une mauvaise option ou des tailles incoherentes doivent produire un message clair."""
    rng = np.random.default_rng(7)
    rep = tempfile.mkdtemp(prefix="mean_erreurs_")
    try:
        _ecrire(_images_synthetiques(rng, 1.0), rep, "F")
        chemins = traitement_holo.lister_images(rep, "tif")

        try:
            traitement_holo.calc_holo_moyen(chemins, type_moyenne="mediane")
            raise AssertionError("un type_moyenne inconnu devrait lever")
        except ValueError as err:
            assert "mediane" in str(err), "message peu clair: {}".format(err)

        try:
            traitement_holo.calc_holo_moyen([])
            raise AssertionError("une liste vide devrait lever")
        except ValueError:
            pass

        autre = os.path.join(rep, "autre.tif")
        Image.fromarray(np.zeros((5, 5), dtype=np.float32), mode="F").save(autre)
        try:
            traitement_holo.calc_holo_moyen(chemins + [autre])
            raise AssertionError("des tailles incoherentes devraient lever")
        except ValueError as err:
            assert "taille" in str(err), "message peu clair: {}".format(err)
        log("messages d'erreur   : explicites (type inconnu, liste vide, tailles melangees)")
    finally:
        shutil.rmtree(rep, ignore_errors=True)


def check_rappel_progression(log):
    """Le rappel de progression doit être appelé une fois par image, dans l'ordre."""
    rng = np.random.default_rng(5)
    images = _images_synthetiques(rng, 1.0)
    rep = tempfile.mkdtemp(prefix="mean_progress_")
    try:
        _ecrire(images, rep, "F")
        chemins = pre_treatment.list_images(rep, "tif")
        appels = []
        pre_treatment.compute_mean_image(chemins, "arithmetic",
                                         progress_callback=lambda i, n: appels.append((i, n)))
        attendu = [(i + 1, NB_IMAGES) for i in range(NB_IMAGES)]
        assert appels == attendu, "progression inattendue: {}".format(appels)
        log("rappel progression  : {} appels, dans l'ordre".format(len(appels)))
    finally:
        shutil.rmtree(rep, ignore_errors=True)


# ---------------------------------------------------------------------------

def _muet(_m):
    pass


def test_moyenne_8bits():
    check_moyenne_8bits(_muet)


def test_moyenne_flottante():
    check_moyenne_flottante(_muet)


def test_moyenne_geometrique():
    check_moyenne_geometrique(_muet)


def test_apercu_bmp():
    check_apercu_bmp(_muet)


def test_accord_entre_implementations():
    check_accord_entre_implementations(_muet)


def test_recadrage():
    check_recadrage(_muet)


def test_formats_sortie():
    check_formats_sortie(_muet)


def test_erreurs_explicites():
    check_erreurs_explicites(_muet)


def test_rappel_progression():
    check_rappel_progression(_muet)


def main():
    print("=" * 74)
    print("Test du calcul de l'hologramme moyen (justesse, sans GPU)")
    print("=" * 74)
    controles = [
        ("moyenne 8 bits", check_moyenne_8bits),
        ("moyenne flottante", check_moyenne_flottante),
        ("moyenne géométrique", check_moyenne_geometrique),
        ("aperçu bmp", check_apercu_bmp),
        ("accord des implémentations", check_accord_entre_implementations),
        ("recadrage centré", check_recadrage),
        ("formats de sortie", check_formats_sortie),
        ("messages d'erreur", check_erreurs_explicites),
        ("rappel de progression", check_rappel_progression),
    ]
    lignes = []
    echecs = []
    for nom, fn in controles:
        try:
            fn(lignes.append)
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
        return 1
    print("SUCCÈS — {} contrôles passés.".format(len(controles)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
