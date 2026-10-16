# -*- coding: utf-8 -*-

"""
Filename: benchmark_complet.py

Description:
Toutes les combinaisons seuil x detection, mesurees sur hologrammes simules (verite terrain
connue) puis sur le film reel, avec les temps de chaque etape.

    seuils            global            moyenne + K x ecart-type du volume entier
                      local sigma       fond ecrete + K x max(sigma, plancher), par maille
                      local MAD         mediane + K x 1,4826 x MAD, par maille
                      local mediane     mediane x (1 + K), par maille

    detections        CCL               binarisation, composantes connexes, barycentre
                      maxima            sommets de la boite, centroide pondere

Les quatre seuils et les deux detections sont independants: les huit combinaisons sont
mesurees sur les MEMES volumes de focus, avec un balayage de K propre a chaque seuil.

Sur le film reel il n'y a pas de verite terrain: on mesure le nombre de detections et leur
**persistance** (part appartenant a une trajectoire d'au moins trois images, chainee par
trackpy), qui recompense les objets reels sans recompenser le bruit.

Ecrit mesures/simulation.csv, mesures/film.csv et mesures/temps.csv.

Usage:
    python utils/benchmark_complet.py [nb_images_du_film]

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import glob
import json
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
import processor
import propagation as propag
from traitement_holo import read_image
from CCL3D import (CCA_CUDA_float, CCL3D, calc_threshold, calc_threshold_local,
                   mailles_fond_sigma, maxima_centroides, taille_boite_voxels,
                   type_threshold)

import benchmark_detection as banc

SEUILS = {
    "global": (8.0, 10.0, 12.0, 15.0, 20.0),
    "local sigma": (12.0, 15.0, 20.0, 25.0),
    "local MAD": (15.0, 20.0, 25.0, 30.0),
    "local mediane": (5.0, 6.0, 8.0, 10.0),
}
DETECTIONS = ("CCL", "maxima")
PAS_Z, PAS_XY = 5, 4
RAYON, MEMOIRE, LONGUEUR = 5e-6, 1, 3
SORTIE = os.path.join(RACINE, "mesures")


def f1(r, p):
    if not np.isfinite(r) or not np.isfinite(p) or r + p == 0:
        return 0.0
    return 2 * r * p / (r + p)


def top():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


class Chaine:
    """Garde les tampons et expose les etapes, pour ne les allouer qu'une fois."""

    def __init__(self):
        self.bufs = [cp.zeros((banc.NB_PIX_Y, banc.NB_PIX_X), dtype=t) for t in
                     (cp.complex64,) * 3 + (cp.float32,) + (cp.complex64,) * 2]
        self.vol = cp.zeros((banc.NB_PLANE, banc.NB_PIX_Y, banc.NB_PIX_X), dtype=cp.float32)
        self.binv = cp.zeros((banc.NB_PLANE, banc.NB_PIX_Y, banc.NB_PIX_X), dtype=cp.bool_)
        self.boite = taille_boite_voxels(banc.BOITE_XY, banc.BOITE_Z, banc.DX, banc.DZ)
        self.mz, self.my, self.mx = banc.MAILLE_Z, banc.MAILLE_XY, banc.MAILLE_XY
        self.gz = banc.NB_PLANE // self.mz
        self.gy = banc.NB_PIX_Y // self.my
        self.gx = banc.NB_PIX_X // self.mx
        self.temps = {}

    def chrono(self, cle, duree):
        self.temps.setdefault(cle, []).append(1000 * duree)

    def volume_de_focus(self, h_net):
        t0 = top()
        propag.volume_propag_angular_spectrum_to_module(
            cp.asarray(h_net), *self.bufs, self.vol,
            banc.WAVELENGTH / banc.INDEX_MEDIUM, banc.MAGNIFICATION, banc.PIX_SIZE,
            banc.NB_PIX_X, banc.NB_PIX_Y, banc.DISTANCE_INI, banc.DZ, banc.NB_PLANE,
            *banc.FILTRE)
        t1 = top()
        focus.focus(self.vol, self.vol, banc.SUM_SIZE, banc.TYPE_FOCUS)
        t2 = top()
        self.chrono("propagation", t1 - t0)
        self.chrono("focus", t2 - t1)

    def statistiques(self):
        """Fond et dispersion par maille: une fois par hologramme, K n'y intervient pas."""
        t0 = top()
        t = self.vol[:self.gz * self.mz, :self.gy * self.my, :self.gx * self.mx]
        t = t[::PAS_Z, ::PAS_XY, ::PAS_XY]
        t = t.reshape(self.gz, self.mz // PAS_Z, self.gy, self.my // PAS_XY,
                      self.gx, self.mx // PAS_XY)
        t = t.transpose(0, 2, 4, 1, 3, 5).reshape(self.gz * self.gy * self.gx, -1)
        p50 = cp.percentile(t, 50, axis=1)
        mad = cp.percentile(cp.abs(t - p50[:, None]), 50, axis=1)
        forme = (self.gz, self.gy, self.gx)
        self.p50 = p50.reshape(forme).astype(cp.float32)
        self.mad = mad.reshape(forme).astype(cp.float32)
        del t
        cp.get_default_memory_pool().free_all_blocks()
        self.chrono("mailles mediane + MAD", top() - t0)

        t0 = top()
        self.infos_sig = mailles_fond_sigma(
            self.vol, 1.0, self.my, self.mz, banc.KAPPA, banc.NB_PASSES, banc.PERCENTILE,
            banc.PAS_ECHANTILLON)
        self.chrono("mailles fond ecrete + sigma", top() - t0)

    def seuil(self, type_seuil, K):
        """Rend soit un flottant (seuil global), soit le dictionnaire des grilles."""
        mailles = (self.mz, self.my, self.mx)
        if type_seuil == "global":
            t0 = top()
            s = calc_threshold(self.vol, K)
            self.chrono("seuil global", top() - t0)
            return s
        if type_seuil == "local sigma":
            grille = self.infos_sig['fond'] + np.float32(K) * cp.maximum(
                self.infos_sig['sigma'], cp.float32(self.infos_sig['sigma_plancher']))
            infos = dict(self.infos_sig)
        elif type_seuil == "local MAD":
            grille = self.p50 + np.float32(K * 1.4826) * self.mad
            infos = {'fond': self.p50, 'sigma': self.mad * np.float32(1.4826),
                     'sigma_plancher': 0.0}
        else:                                    # local mediane
            grille = self.p50 * np.float32(1.0 + K)
            infos = {'fond': self.p50, 'sigma': self.p50 * np.float32(K),
                     'sigma_plancher': 0.0}
        infos = dict(infos)
        infos['seuil_mailles'] = grille
        infos['mailles'] = mailles
        return infos

    def detecte(self, seuil, mode, i_image=1):
        if mode == "CCL":
            t0 = top()
            if isinstance(seuil, dict):
                d_seuil, _ = calc_threshold_local(self.vol, 1.0, self.my, self.mz,
                                                  infos=seuil)
            else:
                d_seuil = seuil
            t1 = top()
            lab, nb = CCL3D(self.binv, self.vol, type_threshold.THRESHOLD, d_seuil,
                            banc.CONNECTIVITE)
            t2 = top()
            if isinstance(d_seuil, cp.ndarray):
                del d_seuil
                cp.get_default_memory_pool().free_all_blocks()
            f = CCA_CUDA_float(lab, self.vol, nb, i_image, banc.NB_PIX_X, banc.NB_PIX_Y,
                               banc.NB_PLANE, banc.DX, banc.DY, banc.DZ,
                               z_offset=banc.DISTANCE_INI)
            del lab
            cp.get_default_memory_pool().free_all_blocks()
            t3 = top()
            self.chrono("interpolation du seuil", t1 - t0)
            self.chrono("binarisation + etiquetage", t2 - t1)
            self.chrono("barycentres", t3 - t2)
            return f
        t0 = top()
        f, _ = maxima_centroides(self.vol, seuil, self.boite, banc.DX, banc.DY, banc.DZ,
                                 i_image=i_image, z_offset=banc.DISTANCE_INI,
                                 capacite=900000)
        self.chrono("sommets + centroides", top() - t0)
        cp.get_default_memory_pool().free_all_blocks()
        return f


def mesure_simulation(chaine):
    lignes = []
    for dossier in sorted(glob.glob(os.path.join(RACINE, "holograms simules", "N*"))):
        rep = [d for d in glob.glob(os.path.join(dossier, "*")) if os.path.isdir(d)][0]
        distance = float(json.load(open(os.path.join(rep, "config_simu.json")))[
            "distance_volume_camera"])
        for chemin in sorted(glob.glob(os.path.join(rep, "simulated_hologram", "*.npy"))):
            base = os.path.basename(chemin).replace("holo_", "", 1)[:-4]
            ref = os.path.join(rep, "holo_ref", "holo_ref_{}.npy".format(base))
            vrai, phi = banc.lire_verite(rep, base, distance)
            h = np.load(chemin).astype(np.float32) / np.load(ref).astype(np.float32)
            chaine.volume_de_focus(h)
            chaine.statistiques()
            for type_seuil, grille in SEUILS.items():
                for K in grille:
                    s = chaine.seuil(type_seuil, K)
                    for mode in DETECTIONS:
                        f = chaine.detecte(s, mode)
                        sc = banc.scores(f[:, 1:4] if len(f) else np.zeros((0, 3)),
                                         vrai, phi)
                        lignes.append({
                            "seuil": type_seuil, "detection": mode, "K": K,
                            "densite": len(vrai), "hologramme": base,
                            "rappel": sc["rappel"], "precision": sc["precision"],
                            "couchees": sc["rappel_couchees"],
                            "detections": sc["nb_detections"],
                            "f1": f1(sc["rappel"], sc["precision"])})
        print("   simulation {} faite".format(os.path.basename(dossier)))
    return pd.DataFrame(lignes)


def mesure_film(chaine, meilleurs, nb_images):
    film = os.path.join(RACINE, "Film REF")
    images = sorted(i for i in os.listdir(film) if i.lower().endswith(".bmp"))[:nb_images]
    h_moyen = np.load(os.path.join(film, "mean", "mean_holo.npy"))
    collecte = {cle: [] for cle in meilleurs}
    for n, nom_image in enumerate(images, start=1):
        h = read_image(os.path.join(film, nom_image), banc.NB_PIX_X, banc.NB_PIX_Y) / h_moyen
        chaine.volume_de_focus(h)
        chaine.statistiques()
        for (type_seuil, mode), K in meilleurs.items():
            s = chaine.seuil(type_seuil, K)
            collecte[(type_seuil, mode)].append(chaine.detecte(s, mode, i_image=n))
        if n % 10 == 0:
            print("   film : {} images".format(n))

    lignes = []
    for (type_seuil, mode), K in meilleurs.items():
        listes = collecte[(type_seuil, mode)]
        d = pd.DataFrame([{"frame": int(l[0]), "x": l[1], "y": l[2], "z": l[3]}
                          for f in listes for l in f])
        if d.empty:
            continue
        lien = processor.link_df(d, search_range=RAYON, memory=MEMOIRE)
        tailles = lien.groupby("particle").size()
        longues = tailles[tailles >= LONGUEUR]
        lignes.append({
            "seuil": type_seuil, "detection": mode, "K": K,
            "detections_par_image": len(d) / len(images),
            "trajectoires": int(len(longues)),
            "persistance": float(longues.sum()) / len(d),
            "longueur_mediane": float(longues.median()) if len(longues) else np.nan,
            "persistantes_par_image": float(longues.sum()) / len(images)})
    return pd.DataFrame(lignes)


def main(nb_images):
    os.makedirs(SORTIE, exist_ok=True)
    chaine = Chaine()

    print("Mesure sur hologrammes simules")
    sim = mesure_simulation(chaine)
    sim.to_csv(os.path.join(SORTIE, "simulation.csv"), index=False)

    moy = sim.groupby(["seuil", "detection", "K", "densite"]).mean(
        numeric_only=True).reset_index()
    meilleurs = {}
    print("\nMeilleur K de chaque combinaison (F1 moyen sur les densites)")
    print("{:<16}{:<10}{:>6}{:>11}".format("seuil", "detection", "K", "F1 moyen"))
    for type_seuil, grille in SEUILS.items():
        for mode in DETECTIONS:
            scores_K = {K: moy[(moy.seuil == type_seuil) & (moy.detection == mode)
                               & (moy.K == K)].f1.mean() for K in grille}
            K = max(scores_K, key=scores_K.get)
            meilleurs[(type_seuil, mode)] = K
            print("{:<16}{:<10}{:>6.0f}{:>11.3f}".format(type_seuil, mode, K, scores_K[K]))

    print("\nMesure sur le film reel ({} images)".format(nb_images))
    chaine.temps = {}
    film = mesure_film(chaine, meilleurs, nb_images)
    film.to_csv(os.path.join(SORTIE, "film.csv"), index=False)
    print(film.to_string(index=False))

    temps = pd.DataFrame([{"etape": k, "ms": float(np.mean(v)), "n": len(v)}
                          for k, v in chaine.temps.items()])
    temps.to_csv(os.path.join(SORTIE, "temps.csv"), index=False)
    print("\nTemps par etape, sur le film")
    print(temps.to_string(index=False))
    print("\nresultats dans {}".format(SORTIE))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
