# -*- coding: utf-8 -*-

"""
Filename: benchmark_detection.py

Description:
Rappel et precision des deux methodes de detection, sur les hologrammes simules de
'holograms simules/', dont la verite terrain est connue.

    seuil global + CCL3D   -> ce que fait script_localisation_pipeline.py
    seuil local + maxima   -> ce que fait script_localisation_pipeline_local_maxima.py

Les deux partagent le meme volume de focus: il est calcule une fois par hologramme, seule
l'etape de detection change. Les parametres ci-dessous recopient ceux des deux scripts; les
modifier ici ne change rien aux scripts, et reciproquement.

Trois conventions de la simulation, trouvees en comparant les positions et verifiees a
chaque execution (voir l'ecart median affiche):

  - le simulateur ecrit l'objet en tableau[x_voxel, y_voxel], donc son x_position_m
    correspond a notre Y (indice de ligne) et son y_position_m a notre X ;
  - son z_position_m est compte depuis le premier plan du volume simule, alors que nos
    scripts reconstruisent depuis la camera: il faut ajouter distance_volume_camera ;
  - l'hologramme se nettoie en le divisant par son holo_ref, qui est l'illumination sans
    objet.

Un objet est considere retrouve si une detection est a moins de 3 um en XY et 5 um en Z,
comme dans les mesures du projet Simu-Bacteria-Holograms.

Usage:
    python utils/benchmark_detection.py
    python utils/benchmark_detection.py "holograms simules"

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

import cupy as cp
import pandas as pd

import focus
import propagation as propag
from focus import Focus_type
from CCL3D import (CCA_CUDA_float, CCL3D, calc_threshold, calc_threshold_local,
                   mailles_fond_median, mailles_fond_sigma,
                   maxima_centroides, taille_boite_voxels, type_threshold)

# --- parametres, recopies des deux scripts hors interface -------------------------------
NB_PIX_X = NB_PIX_Y = 1024
NB_PLANE = 240
PIX_SIZE = 7e-6
MAGNIFICATION = 40.0
WAVELENGTH = 660e-9
INDEX_MEDIUM = 1.33
DX = DY = PIX_SIZE / MAGNIFICATION
DZ = 0.5e-6
DISTANCE_INI = 0.0
FILTRE = (15, 125)
TYPE_FOCUS = Focus_type.SUM_OF_VARIANCE
SUM_SIZE = 15

K_GLOBAL = 11.0                 # script_localisation_pipeline.py
CONNECTIVITE = 26
MIN_VOXEL = 0                   # 0 = pas de filtre sur la taille

K_LOCAL_CCL = 20.0              # script_localisation_pipeline_local_CCL.py
K_MEDIANE = 6.0                 # seuil = fond median x (1 + K); autre echelle que les K ci-dessus
K_MAXIMA = 15.0                 # script_localisation_pipeline_local_maxima.py
MAILLE_XY, MAILLE_Z = 64, 25
KAPPA, NB_PASSES, PERCENTILE = 3.0, 3, 10
PAS_ECHANTILLON = 1
BOITE_XY, BOITE_Z = 2.1e-6, 5.3e-6

# Les trois methodes comparees, dans l'ordre d'affichage. Elles partagent le meme volume de
# focus: il est calcule une fois par hologramme, seule l'etape de detection change.
METHODES = ("global + CCL", "local + CCL", "local mediane + CCL", "local + maxima")

# --- tolerances d'appariement -----------------------------------------------------------
TOL_XY, TOL_Z = 3e-6, 5e-6
ANGLE_COUCHEE = 60.0            # phi au-dela duquel la bacterie est dite couchee


def lire_verite(dossier, base, distance_volume_camera):
    """Positions vraies, ramenees dans la convention d'HOLOTRACKER (metres)."""
    v = pd.read_csv(os.path.join(dossier, "object_positions", "bacteria_{}.csv".format(base)))
    positions = np.stack([v["y_position_m"].to_numpy(),            # leur y est notre X
                          v["x_position_m"].to_numpy(),            # leur x est notre Y
                          v["z_position_m"].to_numpy() + distance_volume_camera], axis=1)
    return positions, v["phi_angle"].to_numpy()


def apparie(detections, vrai):
    """Appariement glouton au plus proche, sous les tolerances.

    Rend (trouve, nb_detections_appariees, ecarts) ou trouve[i] dit si l'objet vrai i a ete
    retrouve. Un objet vrai et une detection ne peuvent servir qu'une fois.
    """
    trouve = np.zeros(len(vrai), dtype=bool)
    if len(detections) == 0:
        return trouve, 0, np.zeros((0, 3))

    dxy = np.hypot(vrai[:, None, 0] - detections[None, :, 0],
                   vrai[:, None, 1] - detections[None, :, 1])
    dz = np.abs(vrai[:, None, 2] - detections[None, :, 2])
    cout = np.where((dxy <= TOL_XY) & (dz <= TOL_Z), dxy + dz, np.inf)

    pris = np.zeros(len(detections), dtype=bool)
    ecarts = []
    for _ in range(min(len(vrai), len(detections))):
        i, j = np.unravel_index(np.argmin(cout), cout.shape)
        if not np.isfinite(cout[i, j]):
            break
        trouve[i], pris[j] = True, True
        ecarts.append(detections[j] - vrai[i])
        cout[i, :] = np.inf
        cout[:, j] = np.inf
    return trouve, int(pris.sum()), np.array(ecarts) if ecarts else np.zeros((0, 3))


def scores(detections, vrai, phi):
    trouve, n_appariees, ecarts = apparie(detections, vrai)
    couchees = phi > ANGLE_COUCHEE
    return {
        "rappel": trouve.mean() if len(vrai) else np.nan,
        "precision": n_appariees / len(detections) if len(detections) else np.nan,
        "rappel_couchees": trouve[couchees].mean() if couchees.any() else np.nan,
        "nb_detections": len(detections),
        "ecarts": ecarts,
    }


def main(racine_holo):
    if not os.path.isdir(racine_holo):
        print("repertoire introuvable: {}".format(racine_holo))
        return 1

    bufs = [cp.zeros((NB_PIX_Y, NB_PIX_X), dtype=t) for t in
            (cp.complex64, cp.complex64, cp.complex64, cp.float32, cp.complex64, cp.complex64)]
    d_volume = cp.zeros((NB_PLANE, NB_PIX_Y, NB_PIX_X), dtype=cp.float32)
    d_binaire = cp.zeros((NB_PLANE, NB_PIX_Y, NB_PIX_X), dtype=cp.bool_)
    boite = taille_boite_voxels(BOITE_XY, BOITE_Z, DX, DZ)

    print("Banc d'essai detection, {} plans de {:.1f} um depuis {:.1f} um, filtre {}-{},"
          " {} sum_size {}".format(NB_PLANE, DZ * 1e6, DISTANCE_INI * 1e6, FILTRE[0],
                                   FILTRE[1], TYPE_FOCUS.name, SUM_SIZE))
    print("boite des maxima: {} plans x {} x {} px\n".format(*boite))
    print("{:>5} {:>6} {:<22} {:>7} {:>10} {:>9} {:>9}".format(
        "N", "holos", "methode", "rappel", "precision", "couchees", "detect."))

    resultats = []
    for dossier_densite in sorted(glob.glob(os.path.join(racine_holo, "N*"))):
        sous = [d for d in glob.glob(os.path.join(dossier_densite, "*")) if os.path.isdir(d)]
        if not sous:
            continue
        dossier = sous[0]
        config = json.load(open(os.path.join(dossier, "config_simu.json")))
        distance_volume_camera = float(config["distance_volume_camera"])
        holos = sorted(glob.glob(os.path.join(dossier, "simulated_hologram", "*.npy")))

        acc = {nom: [] for nom in METHODES}
        ecarts = {nom: [] for nom in METHODES}
        for chemin in holos:
            base = os.path.basename(chemin).replace("holo_", "", 1)[:-4]
            ref = os.path.join(dossier, "holo_ref", "holo_ref_{}.npy".format(base))
            vrai, phi = lire_verite(dossier, base, distance_volume_camera)

            h_net = np.load(chemin).astype(np.float32) / np.load(ref).astype(np.float32)
            propag.volume_propag_angular_spectrum_to_module(
                cp.asarray(h_net), *bufs, d_volume,
                WAVELENGTH / INDEX_MEDIUM, MAGNIFICATION, PIX_SIZE, NB_PIX_X, NB_PIX_Y,
                DISTANCE_INI, DZ, NB_PLANE, *FILTRE)
            focus.focus(d_volume, d_volume, SUM_SIZE, TYPE_FOCUS)

            # seuil global + composantes connexes
            seuil = calc_threshold(d_volume, K_GLOBAL)
            d_labels, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, seuil,
                                 CONNECTIVITE)
            f_ccl = CCA_CUDA_float(d_labels, d_volume, nb, 1, NB_PIX_X, NB_PIX_Y, NB_PLANE,
                                   DX, DY, DZ, z_offset=DISTANCE_INI)
            del d_labels
            cp.get_default_memory_pool().free_all_blocks()
            if MIN_VOXEL > 0 and len(f_ccl):
                f_ccl = f_ccl[f_ccl[:, 4] >= MIN_VOXEL]

            # seuil local + composantes connexes
            d_seuil, _ = calc_threshold_local(d_volume, K_LOCAL_CCL, MAILLE_XY, MAILLE_Z,
                                              KAPPA, NB_PASSES, PERCENTILE, PAS_ECHANTILLON)
            d_labels, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, d_seuil,
                                 CONNECTIVITE)
            del d_seuil
            cp.get_default_memory_pool().free_all_blocks()
            f_local = CCA_CUDA_float(d_labels, d_volume, nb, 1, NB_PIX_X, NB_PIX_Y, NB_PLANE,
                                     DX, DY, DZ, z_offset=DISTANCE_INI)
            del d_labels
            cp.get_default_memory_pool().free_all_blocks()
            if MIN_VOXEL > 0 and len(f_local):
                f_local = f_local[f_local[:, 4] >= MIN_VOXEL]

            # seuil local par mediane + composantes connexes
            infos_med = mailles_fond_median(d_volume, K_MEDIANE, MAILLE_XY, MAILLE_Z)
            d_seuil, _ = calc_threshold_local(d_volume, K_MEDIANE, MAILLE_XY, MAILLE_Z,
                                              infos=infos_med)
            d_labels, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, d_seuil,
                                 CONNECTIVITE)
            del d_seuil
            cp.get_default_memory_pool().free_all_blocks()
            f_med = CCA_CUDA_float(d_labels, d_volume, nb, 1, NB_PIX_X, NB_PIX_Y, NB_PLANE,
                                   DX, DY, DZ, z_offset=DISTANCE_INI)
            del d_labels
            cp.get_default_memory_pool().free_all_blocks()

            # seuil local + maxima
            infos = mailles_fond_sigma(d_volume, K_MAXIMA, MAILLE_XY, MAILLE_Z,
                                       KAPPA, NB_PASSES, PERCENTILE, PAS_ECHANTILLON)
            f_max, _ = maxima_centroides(d_volume, infos, boite, DX, DY, DZ,
                                         z_offset=DISTANCE_INI, capacite=400000)
            cp.get_default_memory_pool().free_all_blocks()

            for nom, f in zip(METHODES, (f_ccl, f_local, f_med, f_max)):
                s = scores(f[:, 1:4] if len(f) else np.zeros((0, 3)), vrai, phi)
                acc[nom].append((s["rappel"], s["precision"], s["rappel_couchees"],
                                 s["nb_detections"]))
                ecarts[nom].append(s["ecarts"])

        n_vrai = len(vrai)
        for nom in METHODES:
            a = np.array(acc[nom], dtype=float)
            moy = np.nanmean(a, axis=0)
            print("{:>5} {:>6} {:<22} {:>7.2f} {:>10.2f} {:>9.2f} {:>9.0f}".format(
                n_vrai, len(holos), nom, moy[0], moy[1], moy[2], moy[3]))
            e = np.vstack([x for x in ecarts[nom] if len(x)])
            resultats.append((n_vrai, nom, moy, e))
        print()

    print("Erreur de position des objets retrouves (biais +/- ecart-type, en um)")
    print("{:>5} {:<22} {:>16} {:>16} {:>16}".format("N", "methode", "X", "Y", "Z"))
    for n_vrai, nom, _, e in resultats:
        print("{:>5} {:<22} {:>+8.3f} +/-{:5.3f} {:>+8.3f} +/-{:5.3f} {:>+8.3f} +/-{:5.3f}".format(
            n_vrai, nom, e[:, 0].mean() * 1e6, e[:, 0].std() * 1e6,
            e[:, 1].mean() * 1e6, e[:, 1].std() * 1e6,
            e[:, 2].mean() * 1e6, e[:, 2].std() * 1e6))
    print("\nUn biais superieur a la tolerance signalerait une convention mal appliquee.")
    return 0


if __name__ == "__main__":
    chemin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RACINE, "holograms simules")
    sys.exit(main(chemin))
