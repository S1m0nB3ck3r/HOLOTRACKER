# -*- coding: utf-8 -*-

"""
Filename: comparer_methodes_film.py

Description:
Compare les quatre methodes de detection sur un FILM REEL, ou il n'y a pas de verite
terrain.

Faute de verite, on juge par la **persistance temporelle**: une vraie bacterie se retrouve
a l'image suivante, un sommet de bruit non. Les detections de chaque methode sont donc
chainees par trackpy (code_link), et on regarde quelle fraction d'entre elles appartient a
une trajectoire d'au moins trois images. C'est un indicateur de qualite qui ne demande
aucune reference exterieure, et il juge sur les donnees de l'utilisateur plutot que sur une
simulation.

A lire avec prudence: une methode qui detecte peu et uniquement des objets brillants aura
une excellente persistance. Le nombre de detections et la persistance se lisent ensemble,
comme le rappel et la precision.

Les K de chaque methode sont ceux calibres sur hologrammes simules (voir calibrer_K.py).

Usage:
    python utils/comparer_methodes_film.py [nb_images]

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import os
import sys
import time

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "code_locate"))
sys.path.insert(0, os.path.join(RACINE, "code_link"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy as cp
import pandas as pd

import focus
import propagation as propag
import processor
from CCL3D import (CCA_CUDA_float, CCL3D, calc_threshold, calc_threshold_local,
                   mailles_fond_median, mailles_fond_sigma, maxima_centroides,
                   taille_boite_voxels, type_threshold)

import benchmark_detection as banc

# K calibres sur hologrammes simules, chacun a son echelle
K = {"global + CCL": 15.0, "local + CCL": 20.0, "local mediane + CCL": 8.0,
     "local + maxima": 20.0}

FILM = os.path.join(RACINE, "Film REF")
RAYON_CHAINAGE = 5e-6           # deplacement maximal entre deux images, en metres
MEMOIRE = 1                     # images d'absence tolerees par trackpy
LONGUEUR_MINIMALE = 3           # une trajectoire credible


def detections(d_volume, d_binaire, boite, i_image):
    """Les quatre methodes sur le meme volume de focus deja calcule."""
    resultats = {}

    seuil = calc_threshold(d_volume, K["global + CCL"])
    lab, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, seuil, banc.CONNECTIVITE)
    resultats["global + CCL"] = CCA_CUDA_float(
        lab, d_volume, nb, i_image, banc.NB_PIX_X, banc.NB_PIX_Y, banc.NB_PLANE,
        banc.DX, banc.DY, banc.DZ, z_offset=banc.DISTANCE_INI)
    del lab
    cp.get_default_memory_pool().free_all_blocks()

    infos = mailles_fond_sigma(d_volume, K["local + CCL"], banc.MAILLE_XY, banc.MAILLE_Z,
                               banc.KAPPA, banc.NB_PASSES, banc.PERCENTILE,
                               banc.PAS_ECHANTILLON)
    d_seuil, _ = calc_threshold_local(d_volume, K["local + CCL"], banc.MAILLE_XY,
                                      banc.MAILLE_Z, infos=infos)
    lab, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, d_seuil,
                    banc.CONNECTIVITE)
    del d_seuil
    cp.get_default_memory_pool().free_all_blocks()
    resultats["local + CCL"] = CCA_CUDA_float(
        lab, d_volume, nb, i_image, banc.NB_PIX_X, banc.NB_PIX_Y, banc.NB_PLANE,
        banc.DX, banc.DY, banc.DZ, z_offset=banc.DISTANCE_INI)
    del lab
    cp.get_default_memory_pool().free_all_blocks()

    infos_med = mailles_fond_median(d_volume, K["local mediane + CCL"], banc.MAILLE_XY,
                                    banc.MAILLE_Z)
    d_seuil, _ = calc_threshold_local(d_volume, K["local mediane + CCL"], banc.MAILLE_XY,
                                      banc.MAILLE_Z, infos=infos_med)
    lab, nb = CCL3D(d_binaire, d_volume, type_threshold.THRESHOLD, d_seuil,
                    banc.CONNECTIVITE)
    del d_seuil
    cp.get_default_memory_pool().free_all_blocks()
    resultats["local mediane + CCL"] = CCA_CUDA_float(
        lab, d_volume, nb, i_image, banc.NB_PIX_X, banc.NB_PIX_Y, banc.NB_PLANE,
        banc.DX, banc.DY, banc.DZ, z_offset=banc.DISTANCE_INI)
    del lab
    cp.get_default_memory_pool().free_all_blocks()

    infos_max = mailles_fond_sigma(d_volume, K["local + maxima"], banc.MAILLE_XY,
                                   banc.MAILLE_Z, banc.KAPPA, banc.NB_PASSES,
                                   banc.PERCENTILE, banc.PAS_ECHANTILLON)
    f, _ = maxima_centroides(d_volume, infos_max, boite, banc.DX, banc.DY, banc.DZ,
                             i_image=i_image, z_offset=banc.DISTANCE_INI, capacite=900000)
    resultats["local + maxima"] = f
    cp.get_default_memory_pool().free_all_blocks()
    return resultats


def persistance(detections_par_image):
    """Chaine les detections et mesure ce qui tient dans le temps."""
    lignes = []
    for f in detections_par_image:
        for ligne in f:
            lignes.append({"frame": int(ligne[0]), "x": ligne[1], "y": ligne[2],
                           "z": ligne[3]})
    df = pd.DataFrame(lignes)
    if df.empty:
        return {"detections": 0, "trajectoires": 0, "persistance": np.nan,
                "longueur_mediane": np.nan}
    lien = processor.link_df(df, search_range=RAYON_CHAINAGE, memory=MEMOIRE)
    tailles = lien.groupby("particle").size()
    longues = tailles[tailles >= LONGUEUR_MINIMALE]
    return {
        "detections": len(df),
        "trajectoires": int(len(longues)),
        "persistance": float(longues.sum()) / len(df),
        "longueur_mediane": float(longues.median()) if len(longues) else np.nan,
    }


def main(nb_images):
    images = sorted(i for i in os.listdir(FILM) if i.lower().endswith(".bmp"))[:nb_images]
    h_moyen = np.load(os.path.join(FILM, "mean", "mean_holo.npy"))

    bufs = [cp.zeros((banc.NB_PIX_Y, banc.NB_PIX_X), dtype=t) for t in
            (cp.complex64, cp.complex64, cp.complex64, cp.float32, cp.complex64, cp.complex64)]
    d_volume = cp.zeros((banc.NB_PLANE, banc.NB_PIX_Y, banc.NB_PIX_X), dtype=cp.float32)
    d_binaire = cp.zeros((banc.NB_PLANE, banc.NB_PIX_Y, banc.NB_PIX_X), dtype=cp.bool_)
    boite = taille_boite_voxels(banc.BOITE_XY, banc.BOITE_Z, banc.DX, banc.DZ)

    print("{} images de {}, K = {}".format(len(images), FILM, K))
    par_methode = {nom: [] for nom in banc.METHODES}
    from traitement_holo import read_image
    t0 = time.perf_counter()
    for n, nom_image in enumerate(images, start=1):
        h = read_image(os.path.join(FILM, nom_image), banc.NB_PIX_X, banc.NB_PIX_Y) / h_moyen
        propag.volume_propag_angular_spectrum_to_module(
            cp.asarray(h), *bufs, d_volume,
            banc.WAVELENGTH / banc.INDEX_MEDIUM, banc.MAGNIFICATION, banc.PIX_SIZE,
            banc.NB_PIX_X, banc.NB_PIX_Y, banc.DISTANCE_INI, banc.DZ, banc.NB_PLANE,
            *banc.FILTRE)
        focus.focus(d_volume, d_volume, banc.SUM_SIZE, banc.TYPE_FOCUS)
        for nom, f in detections(d_volume, d_binaire, boite, n).items():
            par_methode[nom].append(f)
        if n % 10 == 0:
            print("   {} images, {:.0f} s".format(n, time.perf_counter() - t0))

    print("\n{:<22}{:>12}{:>14}{:>16}{:>16}{:>14}".format(
        "methode", "K", "detect./img", "trajectoires", "persistance", "longueur med."))
    for nom in banc.METHODES:
        s = persistance(par_methode[nom])
        print("{:<22}{:>12.1f}{:>14.1f}{:>16}{:>15.0f} %{:>14.0f}".format(
            nom, K[nom], s["detections"] / len(images), s["trajectoires"],
            100 * s["persistance"], s["longueur_mediane"]))
    print("\npersistance = part des detections appartenant a une trajectoire d'au moins "
          "{} images".format(LONGUEUR_MINIMALE))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
