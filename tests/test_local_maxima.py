# -*- coding: utf-8 -*-
"""
Filename: test_local_maxima.py

Description:
Test du seuil local par mailles et de la détection par maxima locaux, sur des volumes de
focus de synthèse dont on connaît les objets.

Comme test_mean_hologram.py, celui-ci vérifie une **justesse** et non une non-régression:
trois bosses gaussiennes posées à des positions connues doivent donner trois objets à ces
positions. Rien n'est comparé à un étalon enregistré.

Ce qui est vérifié:
  1. trois bosses isolées donnent exactement trois objets, aux bonnes positions
  2. le centroïde est plus précis que le voxel: l'erreur reste sous un demi-voxel alors
     que les bosses sont posées entre les voxels
  3. deux bosses plus proches qu'une demi-boîte ne donnent qu'un objet
     -> c'est la limite de séparation de la méthode, fixée par la taille de la boîte
  4. l'écrêtage fait son travail: le seuil d'une maille contenant un objet très brillant
     reste proche de celui d'une maille vide. Sans écrêtage, l'objet gonfle le sigma de sa
     propre maille et relève le seuil censé le détecter
  5. le sigma plancher empêche une maille anormalement calme de donner un seuil très bas
  6. un volume dont la taille n'est pas un multiple des mailles est accepté, et le reste
     est aligné sur la dernière maille complète
  7. un seuil scalaire (seuil global) est accepté à la place du volume de seuil local
  8. les distances rendues sont en mètres, avec le décalage z_offset du premier plan

Nécessite un GPU (CuPy), comme le pipeline qu'il teste.

Usage:
    python tests/test_local_maxima.py
    pytest tests/test_local_maxima.py

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "code_locate"))

import cupy as cp
from CCL3D import (calc_threshold_local, mailles_fond_sigma, maxima_centroides,
                   taille_boite_voxels)

# Géométrie: celle du film de référence, en mètres
DX = DY = 7e-6 / 40.0
DZ = 0.5e-6
BOITE = taille_boite_voxels(2.1e-6, 5.3e-6, DX, DZ)   # (11, 13, 13) voxels

NZ, NY, NX = 50, 256, 256
MAILLE_XY, MAILLE_Z = 64, 25


def volume_de_bosses(bosses, bruit=0.01, graine=12345):
    """Volume de focus synthétique: fond bruité + une gaussienne par bosse.

    bosses: liste de (z, y, x, amplitude) en voxels, coordonnées pouvant être fractionnaires.
    Les largeurs reproduisent la réponse mesurée du critère de focus: 2,1 µm en XY et
    5,3 µm en Z, soit l'équivalent de la boîte des maxima.
    """
    rng = np.random.default_rng(graine)
    vol = rng.normal(1.0, bruit, size=(NZ, NY, NX)).astype(np.float32)
    zz, yy, xx = np.ogrid[0:NZ, 0:NY, 0:NX]
    sigma_xy = (2.1e-6 / DX) / 2.355      # largeur à mi-hauteur -> écart-type
    sigma_z = (5.3e-6 / DZ) / 2.355
    for z0, y0, x0, amp in bosses:
        vol += (amp * np.exp(-((zz - z0) ** 2) / (2 * sigma_z ** 2)
                             - ((yy - y0) ** 2) / (2 * sigma_xy ** 2)
                             - ((xx - x0) ** 2) / (2 * sigma_xy ** 2))).astype(np.float32)
    return cp.asarray(vol)


def detecte(d_vol, K=15.0, z_offset=0.0, **kwargs):
    """Chaine complete: mailles puis maxima. Aucun volume de seuil n'est construit."""
    infos = mailles_fond_sigma(d_vol, K, MAILLE_XY, MAILLE_Z, **kwargs)
    features, descripteurs = maxima_centroides(d_vol, infos, BOITE, DX, DY, DZ,
                                               z_offset=z_offset)
    return features, descripteurs, infos


def check_trois_bosses(note):
    """1, 2 et 8: trois bosses isolées, positions en mètres, précision sous le voxel."""
    bosses = [(12.5, 60.3, 70.7, 0.5), (25.0, 150.5, 180.2, 0.3), (37.4, 200.1, 60.9, 0.4)]
    d_vol = volume_de_bosses(bosses)
    z_offset = 20e-6
    features, descripteurs, _ = detecte(d_vol, z_offset=z_offset)

    assert len(features) == 3, "3 bosses isolées donnent {} objet(s)".format(len(features))

    # Apparier chaque objet trouvé à la bosse la plus proche
    erreurs = []
    for z0, y0, x0, _ in bosses:
        vrai = np.array([x0 * DX, y0 * DY, z_offset + z0 * DZ])
        d = np.linalg.norm(features[:, 1:4] - vrai[None, :], axis=1)
        erreurs.append(features[int(np.argmin(d)), 1:4] - vrai)
    erreurs = np.array(erreurs)

    err_xy = np.abs(erreurs[:, :2]).max()
    err_z = np.abs(erreurs[:, 2]).max()
    assert err_xy < DX / 2, "erreur XY de {:.3g} m, soit plus d'un demi-voxel".format(err_xy)
    assert err_z < DZ / 2, "erreur Z de {:.3g} m, soit plus d'un demi-plan".format(err_z)
    note("3 bosses -> 3 objets ; erreur max {:.1f} nm en XY ({:.2f} voxel), "
         "{:.1f} nm en Z ({:.2f} plan)".format(err_xy * 1e9, err_xy / DX,
                                               err_z * 1e9, err_z / DZ))

    assert (descripteurs['rsb'] > 15).all(), \
        "RSB de {} pour des bosses bien au-dessus du bruit".format(descripteurs['rsb'])
    note("RSB des trois objets : {}".format(np.round(descripteurs['rsb'], 0).tolist()))


def check_separation(note):
    """3: deux bosses plus proches qu'une demi-boîte ne donnent qu'un objet."""
    demi_xy = BOITE[2] // 2
    proches = [(25.0, 128.0, 128.0, 0.5), (25.0, 128.0, 128.0 + demi_xy - 2, 0.5)]
    loin = [(25.0, 128.0, 60.0, 0.5), (25.0, 128.0, 60.0 + 3 * demi_xy, 0.5)]

    n_proches = len(detecte(volume_de_bosses(proches))[0])
    n_loin = len(detecte(volume_de_bosses(loin))[0])

    assert n_proches == 1, "deux bosses à {} px donnent {} objets, 1 attendu".format(
        demi_xy - 2, n_proches)
    assert n_loin == 2, "deux bosses à {} px donnent {} objets, 2 attendus".format(
        3 * demi_xy, n_loin)
    note("séparation : {} px -> 1 objet (bosses fusionnées), {} px -> 2 objets".format(
        demi_xy - 2, 3 * demi_xy))


def gonflement_du_seuil(amplitude, nb_passes):
    """Écart relatif entre le seuil de la maille contenant l'objet et celui des mailles vides."""
    d_vol = volume_de_bosses([(12.0, 32.0, 32.0, amplitude)])
    infos = mailles_fond_sigma(d_vol, 15.0, MAILLE_XY, MAILLE_Z, nb_passes=nb_passes)
    seuils = cp.asnumpy(infos['seuil_mailles'])
    vides = np.median(np.delete(seuils.ravel(), 0))
    del d_vol
    cp.get_default_memory_pool().free_all_blocks()
    return abs(seuils[0, 0, 0] - vides) / vides


def check_ecretage(note):
    """4: l'écrêtage empêche un objet de relever le seuil de sa propre maille.

    Le fond du volume de synthèse a un écart-type de 0,01, donc une bosse d'amplitude 0,5
    vaut 50 sigma: c'est le contraste mesuré en simulation pour une bactérie couchée
    (62 sigma avec SUM_OF_VARIANCE). Dans ce régime, 3 passes suffisent.

    La convergence est d'autant plus lente que l'objet est brillant. Gonflement du seuil de
    sa maille, mesuré ici:

        contraste    sans écrêtage   1 passe   3 passes   6 passes
          20 sigma        11 %          3 %       1 %        1 %
          50 sigma        40 %         11 %       2 %        1 %
         200 sigma       193 %         65 %       8 %        1 %
        5000 sigma      5131 %       1911 %     355 %       30 %

    Conséquence à connaître: un objet anormalement brillant (poussière, agrégat) garde un
    seuil local élevé autour de lui et peut masquer ses voisines. Les 3 passes par défaut
    conviennent aux bactéries, pas à n'importe quel objet.
    """
    ecart = gonflement_du_seuil(0.5, 3)
    assert ecart < 0.05, ("a 50 sigma, la maille contenant l'objet a un seuil {:.0f} % plus "
                          "haut que les mailles vides: l'écrêtage ne retire pas "
                          "l'objet".format(ecart * 100))

    # Chaque passe doit réduire le gonflement, sinon l'itération ne sert à rien.
    sans, une = gonflement_du_seuil(0.5, 0), gonflement_du_seuil(0.5, 1)
    assert sans > une > ecart, \
        "l'écrêtage ne converge pas: {:.2f} sans passe, {:.2f} à 1 passe, {:.2f} à 3".format(
            sans, une, ecart)
    note("écrêtage à 50 sigma : gonflement du seuil {:.0f} % sans écrêtage, {:.0f} % à "
         "1 passe, {:.0f} % à 3 passes".format(sans * 100, une * 100, ecart * 100))

    # Un objet très brillant n'est pas entièrement retiré en 3 passes: la limite est mesurée
    # ici pour qu'elle reste visible si quelqu'un change le nombre de passes.
    extreme = gonflement_du_seuil(50.0, 3)
    assert extreme > 1.0, ("un objet a 5000 sigma devrait encore gonfler le seuil de sa "
                           "maille apres 3 passes, or le gonflement est de "
                           "{:.0f} %".format(extreme * 100))
    note("limite : a 5000 sigma (poussiere), le gonflement reste de {:.0f} % apres "
         "3 passes".format(extreme * 100))


def check_sigma_plancher(note):
    """5: une maille anormalement calme ne donne pas un seuil très bas."""
    rng = np.random.default_rng(7)
    vol = rng.normal(1.0, 0.01, size=(NZ, NY, NX)).astype(np.float32)
    vol[:MAILLE_Z, :MAILLE_XY, :MAILLE_XY] = 1.0        # une maille sans aucun bruit
    d_vol = cp.asarray(vol)

    infos = mailles_fond_sigma(d_vol, 15.0, MAILLE_XY, MAILLE_Z)
    sigma = cp.asnumpy(infos['sigma'])
    seuil = cp.asnumpy(infos['seuil_mailles'])
    assert sigma[0, 0, 0] < infos['sigma_plancher'], \
        "la maille sans bruit devrait avoir un sigma sous le plancher"
    attendu = 1.0 + 15.0 * infos['sigma_plancher']
    assert abs(seuil[0, 0, 0] - attendu) < 1e-6, \
        "seuil {:.6g} au lieu de {:.6g}: le plancher n'est pas applique".format(
            seuil[0, 0, 0], attendu)
    note("sigma plancher : maille sans bruit, sigma {:.2g} -> plancher {:.2g} appliqué".format(
        sigma[0, 0, 0], infos['sigma_plancher']))


def check_volume_non_multiple(note):
    """6: un volume qui n'est pas un multiple des mailles est accepté."""
    rng = np.random.default_rng(3)
    nz, ny, nx = 57, 200, 150        # ni 57/25, ni 200/64, ni 150/64 ne sont entiers
    d_vol = cp.asarray(rng.normal(1.0, 0.01, size=(nz, ny, nx)).astype(np.float32))
    d_seuil, infos = calc_threshold_local(d_vol, 15.0, MAILLE_XY, MAILLE_Z)
    assert d_seuil.shape == (nz, ny, nx), \
        "seuil de taille {} pour un volume {}".format(d_seuil.shape, (nz, ny, nx))
    # Les comparaisons se font sur le CPU: une réduction booléenne CuPy (.all(), .any(),
    # count_nonzero) passe par CUB, qui exige nvcc. Voir la note de CLAUDE.md.
    seuil = cp.asnumpy(d_seuil)
    assert np.isfinite(seuil).all(), "le seuil contient des valeurs non finies"
    # Le reste est aligné sur la dernière maille complète
    gz, gy, gx = nz // MAILLE_Z, ny // MAILLE_XY, nx // MAILLE_XY
    assert (seuil[gz * MAILLE_Z:] == seuil[gz * MAILLE_Z - 1]).all(), \
        "les plans au-delà de la dernière maille complète ne sont pas alignés sur elle"
    note("volume {}x{}x{} non multiple des mailles : seuil complet, bords alignés".format(
        nz, ny, nx))


def check_bosse_au_bord(note):
    """7: une bosse collée au bord est trouvée, et son centroïde reste au bord.

    Le noyau écrête les indices de la boîte aux limites du volume. Si l'indice d'un fil est
    traité comme non signé, `i - 6` pour i = 0 repasse par le haut et l'écrêtage renvoie le
    bord OPPOSÉ: la bosse est alors comparée aux voxels d'en face, et son centroïde part à
    l'autre bout du volume. C'est arrivé, d'où ce contrôle.
    """
    bosses = [(25.0, 2.0, 3.0, 0.5)]      # à 2 et 3 voxels des bords Y et X
    d_vol = volume_de_bosses(bosses)
    features, _, _ = detecte(d_vol)

    assert len(features) == 1, \
        "une bosse au bord donne {} objet(s), 1 attendu".format(len(features))
    x, y, z = features[0, 1] / DX, features[0, 2] / DY, features[0, 3] / DZ
    assert abs(x - 3.0) < 2 and abs(y - 2.0) < 2 and abs(z - 25.0) < 2, \
        ("centroïde en (x={:.1f}, y={:.1f}, z={:.1f}) voxels au lieu de (3, 2, 25): "
         "la boîte déborde du mauvais côté".format(x, y, z))
    note("bosse à 2 voxels du bord : trouvée en (x={:.1f}, y={:.1f}, z={:.1f}) voxels".format(
        x, y, z))


def check_seuil_scalaire(note):
    """7: un seuil global (flottant) est accepté à la place du volume de seuil local."""
    bosses = [(12.5, 60.3, 70.7, 0.5), (25.0, 150.5, 180.2, 0.5)]
    d_vol = volume_de_bosses(bosses)
    seuil = float(d_vol.mean().item() + 15.0 * d_vol.std().item())
    features, descripteurs = maxima_centroides(d_vol, seuil, BOITE, DX, DY, DZ)
    assert len(features) == 2, \
        "{} objets avec un seuil scalaire, 2 attendus".format(len(features))
    assert (descripteurs['rsb'] > 1).all(), "RSB incohérent avec un seuil scalaire"
    note("seuil scalaire (mode global) accepté : 2 bosses -> 2 objets")


def main():
    controles = [
        ("trois bosses isolées", check_trois_bosses),
        ("limite de séparation", check_separation),
        ("écrêtage des objets", check_ecretage),
        ("sigma plancher", check_sigma_plancher),
        ("volume non multiple des mailles", check_volume_non_multiple),
        ("bosse au bord du volume", check_bosse_au_bord),
        ("seuil scalaire", check_seuil_scalaire),
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
