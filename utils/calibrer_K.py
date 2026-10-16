# -*- coding: utf-8 -*-

"""
Filename: calibrer_K.py

Description:
Cherche le meilleur K de chacune des trois methodes de detection, sur les hologrammes
simules de 'holograms simules/' dont la verite terrain est connue.

HOLOTRACKER applique **un seul K a tout un film**, quelle que soit la densite. Le K retenu
ici est donc celui qui maximise le F1 **moyen sur toutes les densites**, et non le meilleur
K de chaque densite prise a part - ce dernier est affiche a titre indicatif, pour voir a
quel point l'optimum se deplace.

Le fond et l'ecart-type des mailles ne dependent pas de K: ils sont calcules une fois par
hologramme, et seul le seuil est reconstruit pour chaque K. C'est ce qui rend le balayage
abordable.

Usage:
    python utils/calibrer_K.py
    python utils/calibrer_K.py "holograms simules"

Les resultats sont ecrits dans <repertoire>/calibration_K.csv, une ligne par
(densite, methode, K).

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import glob
import json
import os
import sys

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "code_locate"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy as cp
import pandas as pd

import focus
import propagation as propag
from CCL3D import (CCA_CUDA_float, CCL3D, calc_threshold, calc_threshold_local,
                   mailles_fond_median, mailles_fond_sigma,
                   maxima_centroides, taille_boite_voxels, type_threshold)

import benchmark_detection as banc

# K balayes. Les trois premieres methodes comptent des ecarts-types; la quatrieme
# multiplie le fond median, donc son echelle est tout autre et sa grille aussi.
GRILLE_K = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
GRILLE_K_MEDIANE = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0)


def seuil_mailles_pour(infos, K):
    """Reconstruit le seuil des mailles pour un autre K, sans refaire l'ecretage."""
    return infos['fond'] + K * cp.maximum(infos['sigma'], cp.float32(infos['sigma_plancher']))


def f1(rappel, precision):
    if not np.isfinite(rappel) or not np.isfinite(precision) or rappel + precision == 0:
        return 0.0
    return 2 * rappel * precision / (rappel + precision)


def main(racine_holo):
    bufs = [cp.zeros((banc.NB_PIX_Y, banc.NB_PIX_X), dtype=t) for t in
            (cp.complex64, cp.complex64, cp.complex64, cp.float32, cp.complex64, cp.complex64)]
    d_volume = cp.zeros((banc.NB_PLANE, banc.NB_PIX_Y, banc.NB_PIX_X), dtype=cp.float32)
    d_binaire = cp.zeros((banc.NB_PLANE, banc.NB_PIX_Y, banc.NB_PIX_X), dtype=cp.bool_)
    boite = taille_boite_voxels(banc.BOITE_XY, banc.BOITE_Z, banc.DX, banc.DZ)

    lignes = []
    for dossier_densite in sorted(glob.glob(os.path.join(racine_holo, "N*"))):
        sous = [d for d in glob.glob(os.path.join(dossier_densite, "*")) if os.path.isdir(d)]
        if not sous:
            continue
        dossier = sous[0]
        config = json.load(open(os.path.join(dossier, "config_simu.json")))
        distance = float(config["distance_volume_camera"])
        holos = sorted(glob.glob(os.path.join(dossier, "simulated_hologram", "*.npy")))

        for chemin in holos:
            base = os.path.basename(chemin).replace("holo_", "", 1)[:-4]
            ref = os.path.join(dossier, "holo_ref", "holo_ref_{}.npy".format(base))
            vrai, phi = banc.lire_verite(dossier, base, distance)

            h_net = np.load(chemin).astype(np.float32) / np.load(ref).astype(np.float32)
            propag.volume_propag_angular_spectrum_to_module(
                cp.asarray(h_net), *bufs, d_volume,
                banc.WAVELENGTH / banc.INDEX_MEDIUM, banc.MAGNIFICATION, banc.PIX_SIZE,
                banc.NB_PIX_X, banc.NB_PIX_Y, banc.DISTANCE_INI, banc.DZ, banc.NB_PLANE,
                *banc.FILTRE)
            focus.focus(d_volume, d_volume, banc.SUM_SIZE, banc.TYPE_FOCUS)

            # fond et sigma: une seule fois, ils ne dependent pas de K
            infos = mailles_fond_sigma(d_volume, 1.0, banc.MAILLE_XY, banc.MAILLE_Z,
                                       banc.KAPPA, banc.NB_PASSES, banc.PERCENTILE,
                                       banc.PAS_ECHANTILLON)
            # le fond median non plus
            fond_med = mailles_fond_median(d_volume, 0.0, banc.MAILLE_XY,
                                           banc.MAILLE_Z)['fond']

            for K, K_med in zip(GRILLE_K, GRILLE_K_MEDIANE):
                # --- seuil global + CCL
                seuil = calc_threshold(d_volume, K)
                lab, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, seuil,
                                banc.CONNECTIVITE)
                f_glob = CCA_CUDA_float(lab, d_volume, nb, 1, banc.NB_PIX_X, banc.NB_PIX_Y,
                                        banc.NB_PLANE, banc.DX, banc.DY, banc.DZ,
                                        z_offset=banc.DISTANCE_INI)
                del lab
                cp.get_default_memory_pool().free_all_blocks()

                # --- seuil local + CCL
                infos_K = dict(infos)
                infos_K['seuil_mailles'] = seuil_mailles_pour(infos, K)
                d_seuil, _ = calc_threshold_local(
                    d_volume, K, banc.MAILLE_XY, banc.MAILLE_Z, banc.KAPPA, banc.NB_PASSES,
                    banc.PERCENTILE, banc.PAS_ECHANTILLON, infos=infos_K)
                lab, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, d_seuil,
                                banc.CONNECTIVITE)
                del d_seuil
                cp.get_default_memory_pool().free_all_blocks()
                f_loc = CCA_CUDA_float(lab, d_volume, nb, 1, banc.NB_PIX_X, banc.NB_PIX_Y,
                                       banc.NB_PLANE, banc.DX, banc.DY, banc.DZ,
                                       z_offset=banc.DISTANCE_INI)
                del lab
                cp.get_default_memory_pool().free_all_blocks()

                # --- seuil local par mediane + CCL
                infos_med = {'fond': fond_med, 'sigma': fond_med * K_med,
                             'sigma_plancher': 0.0,
                             'seuil_mailles': fond_med * np.float32(1.0 + K_med),
                             'mailles': (banc.MAILLE_Z, banc.MAILLE_XY, banc.MAILLE_XY)}
                d_seuil, _ = calc_threshold_local(
                    d_volume, K_med, banc.MAILLE_XY, banc.MAILLE_Z, infos=infos_med)
                lab, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, d_seuil,
                                banc.CONNECTIVITE)
                del d_seuil
                cp.get_default_memory_pool().free_all_blocks()
                f_med = CCA_CUDA_float(lab, d_volume, nb, 1, banc.NB_PIX_X, banc.NB_PIX_Y,
                                       banc.NB_PLANE, banc.DX, banc.DY, banc.DZ,
                                       z_offset=banc.DISTANCE_INI)
                del lab
                cp.get_default_memory_pool().free_all_blocks()

                # --- seuil local + maxima
                f_max, _ = maxima_centroides(d_volume, infos_K, boite, banc.DX, banc.DY,
                                             banc.DZ, z_offset=banc.DISTANCE_INI,
                                             capacite=900000)
                cp.get_default_memory_pool().free_all_blocks()

                for nom, f in zip(banc.METHODES, (f_glob, f_loc, f_med, f_max)):
                    s = banc.scores(f[:, 1:4] if len(f) else np.zeros((0, 3)), vrai, phi)
                    lignes.append({
                        "densite": len(vrai), "hologramme": base, "methode": nom,
                        "K": K_med if nom.startswith("local mediane") else K,
                        "rappel": s["rappel"], "precision": s["precision"],
                        "rappel_couchees": s["rappel_couchees"],
                        "detections": s["nb_detections"],
                        "f1": f1(s["rappel"], s["precision"])})
            print("  {} {} : {} K balayes".format(
                os.path.basename(dossier_densite), base, len(GRILLE_K)))

    df = pd.DataFrame(lignes)
    sortie = os.path.join(racine_holo, "calibration_K.csv")
    df.to_csv(sortie, index=False)

    moyennes = df.groupby(["methode", "K", "densite"]).mean(numeric_only=True).reset_index()

    print("\nF1 par densite et par K\n")
    for methode in banc.METHODES:
        sous = moyennes[moyennes.methode == methode]
        densites = sorted(sous.densite.unique())
        print("{}".format(methode))
        print("{:>6}".format("K") + "".join("{:>9}".format("N={}".format(n)) for n in densites)
              + "{:>10}{:>10}".format("F1 moyen", "F1 min"))
        meilleur, meilleur_score = None, -1
        grille = GRILLE_K_MEDIANE if methode.startswith("local mediane") else GRILLE_K
        for K in grille:
            l = sous[sous.K == K].set_index("densite")["f1"]
            vals = [l.get(n, np.nan) for n in densites]
            moy, mini = np.nanmean(vals), np.nanmin(vals)
            if moy > meilleur_score:
                meilleur, meilleur_score = K, moy
            print("{:>6.1f}".format(K) + "".join("{:>9.2f}".format(v) for v in vals)
                  + "{:>10.3f}{:>10.2f}".format(moy, mini))
        print("  -> meilleur K unique : {:.1f}  (F1 moyen {:.3f})\n".format(
            meilleur, meilleur_score))

    print("\nOptimum densite par densite (pour voir le deplacement)\n")
    print("{:<18}".format("methode") + "".join(
        "{:>12}".format("N={}".format(n)) for n in sorted(moyennes.densite.unique())))
    for methode in banc.METHODES:
        sous = moyennes[moyennes.methode == methode]
        ligne = "{:<18}".format(methode)
        for n in sorted(sous.densite.unique()):
            l = sous[sous.densite == n]
            k = l.loc[l.f1.idxmax(), "K"]
            ligne += "{:>12}".format("K={:.1f} ({:.2f})".format(k, l.f1.max()))
        print(ligne)

    print("\nresultats detailles : {}".format(sortie))
    return 0


if __name__ == "__main__":
    chemin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RACINE, "holograms simules")
    sys.exit(main(chemin))
