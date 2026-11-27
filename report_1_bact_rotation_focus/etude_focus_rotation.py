# -*- coding: utf-8 -*-

"""
Filename: etude_focus_rotation.py

Description:
Localisation d'une bacterie seule, inclinee de phi = 0 deg (dans l'axe optique) a 90 deg
(couchee), avec les cinq fonctions de focus elementaires d'HOLOTRACKER, par seuil global + CCL.

Donnees: SIMU_1_BACT_ROTATION/phi_XX/, un hologramme par angle (bâtonnet 1,5 x 0,8 um,
incline dans le plan XZ, centre au pixel (512, 512) et a 50 um du capteur).

Pour chaque hologramme:
  1. l'hologramme brut, sans division par holo_ref, filtre par un passe-haut seul (15) ;
  2. un seul volume de module, reconstruit par spectre angulaire ;
  3. pour chaque fonction de focus, le volume de focus, puis:
       - le contraste: max du focus / moyenne du fond, le fond etant tout le volume sauf
         une colonne de 2 x RAYON_FOND de cote centree sur la bacterie, sur toute la
         profondeur ;
       - un balayage du K du seuil global (seuil = moyenne + K x ecart-type du volume),
         et pour chaque K la detection (CCL3D + barycentres) et son appariement a la
         position vraie.

Le K est ensuite optimise **par fonction de focus, un seul K pour tous les angles**, comme
HOLOTRACKER applique un seul K a tout un film: le centre geometrique du plateau ou le F1
moyen sur les angles (rappel et precision sur l'unique objet) est maximal (voir k_optimal).

Sorties, dans le repertoire de ce script:
    balayage_K.csv          une ligne par (phi, focus, K)
    resultats_K_optimal.csv une ligne par (phi, focus), au K retenu pour ce focus
    contraste_focus.csv     une ligne par (phi, focus): max, fond, contraste
    erreur_XY_vs_phi.png, erreur_Z_vs_phi.png, contraste_vs_phi.png,
    contraste_normalise_vs_phi.png, contraste_sigma_vs_phi.png, F1_vs_K.png

Usage:
    python report_1_bact_rotation_focus/etude_focus_rotation.py
    python report_1_bact_rotation_focus/etude_focus_rotation.py --sans-mesure
        (refait l'analyse et les graphes a partir des CSV, sans GPU)

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

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(RACINE, "code_locate"))
sys.path.insert(0, os.path.join(RACINE, "utils"))

# Les reductions CuPy (somme, argmax) passent sinon par CUB, que nvcc doit compiler, et nvcc
# ne compile pas sur la machine de developpement (voir CLAUDE.md). Sans CUB, elles passent
# par NVRTC comme le reste du pipeline. A fixer avant l'import de cupy.
os.environ["CUPY_ACCELERATORS"] = ""

import cupy as cp
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import focus
import propagation as propag
from focus import Focus_type
from CCL3D import CCA_CUDA_float, CCL3D, type_threshold

import benchmark_detection as banc          # appariement et tolerances (3 um XY, 5 um Z)

# --- parametres -------------------------------------------------------------------------
SIMU = os.path.join(RACINE, "SIMU_1_BACT_ROTATION")

DISTANCE_INI = 30e-6            # premier plan reconstruit, depuis le capteur
DZ = 0.2e-6                     # pas entre plans
NB_PLANE = 200                  # 30 -> 69,8 um ; la bacterie est a 50 um
FILTRE = (15, 0)                # filtre spectral en pixels de frequence: passe-haut 15 seul
                                # (0 = pas de passe-bas, voir d_spec_filter_FFT)
SUM_SIZE = 15                   # fenetre des criteres de focus, en pixels
CONNECTIVITE = 26

FOCUS = (Focus_type.SUM_OF_VARIANCE, Focus_type.TENEGRAD, Focus_type.SUM_OF_LAPLACIAN,
         Focus_type.SUM_OF_GRADIENT, Focus_type.SUM_OF_INTENSITY)
# MEAN_ARITH_ALL et MEAN_GEO_ALL (combinaisons des cinq criteres) sont exclus de l'etude

GRILLE_K = np.round(np.geomspace(3.0, 1000.0, 37), 2)   # K balayes, echelle log
RAYON_FOND = 5e-6               # demi-cote de la colonne exclue du fond, en metres

# couleurs (palette categorielle validee) et marqueurs: un couple par focus, fixe
COULEURS = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac")
MARQUEURS = ("o", "s", "^", "D", "v")


def lire_angle(dossier):
    """Hologramme brut, position vraie (X, Y, Z en metres) et parametres d'un angle."""
    cfg = json.load(open(os.path.join(dossier, "config_simu.json"), encoding="utf-8"))
    # Pas de division par holo_ref: l'hologramme est propage tel quel. Le passe-haut retire
    # le fond continu, et ni le seuil global (moyenne + K x ecart-type) ni le rapport
    # max / fond ne dependent de l'echelle de l'hologramme.
    h_brut = tifffile.imread(os.path.join(dossier, "simulated_hologram", "holo_0.tiff"))
    h_brut = h_brut.astype(np.float32)

    v = pd.read_csv(os.path.join(dossier, "object_positions", "bact_0.csv")).iloc[0]
    # X = colonnes, Y = lignes, comme HOLOTRACKER ; Z compte depuis le capteur
    vrai = np.array([v.x_position_m, v.y_position_m,
                     v.z_position_m + cfg["distance_volume_camera"]])
    return h_brut, vrai, float(v.phi_angle), cfg


def contraste(d_focus, vrai, dx):
    """Max du focus, moyenne du fond (volume hors colonne autour de la bacterie)."""
    r = int(round(RAYON_FOND / dx))
    ix, iy = int(round(vrai[0] / dx)), int(round(vrai[1] / dx))
    colonne = d_focus[:, iy - r:iy + r + 1, ix - r:ix + r + 1]
    somme_fond = float(cp.sum(d_focus, dtype=cp.float64)) - float(cp.sum(colonne, dtype=cp.float64))
    nb_fond = d_focus.size - colonne.size
    i_max = int(cp.argmax(d_focus))
    k, j, i = np.unravel_index(i_max, d_focus.shape)
    return {"max_focus": float(d_focus[k, j, i]), "fond_moyen": somme_fond / nb_fond,
            "x_max": i * dx, "y_max": j * dx, "z_max": DISTANCE_INI + k * DZ}


def detecte(d_focus, d_binaire, seuil, dx):
    """Seuil global + composantes connexes + barycentres: positions (n, 3) en metres."""
    lab, nb = CCL3D(d_binaire, d_focus, type_threshold.THRESHOLD, seuil, CONNECTIVITE)
    if nb == 0:
        del lab
        return np.zeros((0, 3))
    f = CCA_CUDA_float(lab, d_focus, nb, 1, d_focus.shape[2], d_focus.shape[1], NB_PLANE,
                       dx, dx, DZ, z_offset=DISTANCE_INI)
    del lab
    cp.get_default_memory_pool().free_all_blocks()
    return f[:, 1:4]


def mesure():
    dossiers = sorted(glob.glob(os.path.join(SIMU, "phi_*")))
    _, _, _, cfg = lire_angle(dossiers[0])
    n = int(cfg["holo_size_xy"])
    dx = cfg["pix_size"] / cfg["magnification"]
    lambda_milieu = cfg["wavelength"] / cfg["index_medium"]

    bufs = [cp.zeros((n, n), dtype=t) for t in
            (cp.complex64,) * 3 + (cp.float32,) + (cp.complex64,) * 2]
    d_module = cp.zeros((NB_PLANE, n, n), dtype=cp.float32)
    d_focus = cp.zeros((NB_PLANE, n, n), dtype=cp.float32)
    d_binaire = cp.zeros((NB_PLANE, n, n), dtype=cp.bool_)

    print("{} angles, volume {} plans de {:.2f} um depuis {:.1f} um, pixel objet {:.4f} um,"
          " sum_size {}".format(len(dossiers), NB_PLANE, DZ * 1e6, DISTANCE_INI * 1e6,
                                dx * 1e6, SUM_SIZE))
    balayage, contrastes = [], []
    for dossier in dossiers:
        h_brut, vrai, phi, cfg_phi = lire_angle(dossier)
        assert cfg_phi["pix_size"] == cfg["pix_size"] and cfg_phi["holo_size_xy"] == n
        propag.volume_propag_angular_spectrum_to_module(
            cp.asarray(h_brut), *bufs, d_module, lambda_milieu, cfg["magnification"],
            cfg["pix_size"], n, n, DISTANCE_INI, DZ, NB_PLANE, *FILTRE)

        for type_focus in FOCUS:
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            focus.focus(d_module, d_focus, SUM_SIZE, type_focus)
            cp.cuda.Stream.null.synchronize()
            temps = time.perf_counter() - t0

            moyenne = float(cp.mean(d_focus))
            ecart_type = float(cp.std(d_focus))
            c = contraste(d_focus, vrai, dx)
            # K au-dela duquel plus aucun voxel ne passe le seuil
            k_limite = (c["max_focus"] - moyenne) / ecart_type
            contrastes.append(dict(phi=phi, focus=type_focus.name, **c,
                                   rapport_max_fond=c["max_focus"] / c["fond_moyen"],
                                   moyenne_volume=moyenne, ecart_type_volume=ecart_type,
                                   k_limite=k_limite, temps_focus_ms=1000 * temps))

            for K in GRILLE_K:
                if K >= k_limite:
                    det = np.zeros((0, 3))
                else:
                    det = detecte(d_focus, d_binaire, moyenne + K * ecart_type, dx)
                trouve, n_app, ecarts = banc.apparie(det, vrai[None, :])
                ligne = dict(phi=phi, focus=type_focus.name, K=K, detections=len(det),
                             trouve=bool(trouve[0]),
                             f1=2 * n_app / (1 + len(det)) if len(det) else 0.0,
                             ex=np.nan, ey=np.nan, ez=np.nan)
                if len(ecarts):
                    ligne.update(ex=ecarts[0][0], ey=ecarts[0][1], ez=ecarts[0][2])
                balayage.append(ligne)
        print("   phi = {:4.1f} deg fait".format(phi))
    return pd.DataFrame(balayage), pd.DataFrame(contrastes)


def k_optimal(balayage):
    """Un K par focus, au centre du plateau ou le F1 moyen sur les angles est maximal.

    Le plateau est la plus longue suite de K consecutifs de la grille qui atteignent le F1
    maximal. Le K retenu est le point de grille le plus proche de son centre geometrique,
    racine(K_min x K_max): c'est celui qui laisse le plus de marge de part et d'autre, vers
    les fausses detections (K trop bas) comme vers la perte de l'objet (K trop haut).
    L'erreur de position varie peu le long du plateau, elle ne sert donc pas a choisir.
    """
    b = balayage.copy()
    b["erreur_3d"] = np.sqrt(b.ex ** 2 + b.ey ** 2 + b.ez ** 2)
    par_k = b.groupby(["focus", "K"]).agg(f1=("f1", "mean"), erreur=("erreur_3d", "mean"))
    par_k = par_k.reset_index()
    choix, plateaux = {}, {}
    for nom, g in par_k.groupby("focus"):
        g = g.sort_values("K")
        au_max = np.isclose(g.f1.to_numpy(), g.f1.max())
        # plus longue suite de True consecutifs
        debut, longueur, meilleur = 0, 0, (0, 0)
        for i, ok in enumerate(au_max):
            if ok:
                if longueur == 0:
                    debut = i
                longueur += 1
                if longueur > meilleur[1]:
                    meilleur = (debut, longueur)
            else:
                longueur = 0
        ks = g.K.to_numpy()[meilleur[0]:meilleur[0] + meilleur[1]]
        centre = np.sqrt(ks[0] * ks[-1])
        choix[nom] = float(ks[np.argmin(np.abs(np.log(ks / centre)))])
        plateaux[nom] = (float(ks[0]), float(ks[-1]), float(g.f1.max()))
    return choix, plateaux, par_k


def graphes(res, contrastes, par_k, choix):
    noms = [f.name for f in FOCUS]
    style = {nom: dict(color=COULEURS[i], marker=MARQUEURS[i], markersize=6, linewidth=1.5)
             for i, nom in enumerate(noms)}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9c9c9c", "font.size": 10})

    def legende(nom):
        return "{} (K = {:g})".format(nom, choix[nom])

    # erreur X et Y
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for ax, col, titre in zip(axes, ("ex", "ey"), ("X (horizontal)", "Y (vertical)")):
        for nom in noms:
            r = res[res.focus == nom].sort_values("phi")
            ax.plot(r.phi, r[col] * 1e6, label=legende(nom), **style[nom])
        ax.axhline(0, color="#9c9c9c", linewidth=0.8)
        ax.set_ylabel("erreur {} (um)".format(titre))
    axes[-1].set_xlabel("phi, angle a l'axe optique (deg)  -  0 = dans l'axe, 90 = couchee")
    axes[0].set_title("Erreur de localisation en X et Y, seuil global au K optimal de chaque focus")
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "erreur_XY_vs_phi.png"), dpi=150)
    plt.close(fig)

    # erreur Z
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom in noms:
        r = res[res.focus == nom].sort_values("phi")
        ax.plot(r.phi, r.ez * 1e6, label=legende(nom), **style[nom])
    ax.axhline(0, color="#9c9c9c", linewidth=0.8)
    ax.set_xlabel("phi, angle a l'axe optique (deg)  -  0 = dans l'axe, 90 = couchee")
    ax.set_ylabel("erreur Z (um)")
    ax.set_title("Erreur de localisation en Z, seuil global au K optimal de chaque focus")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "erreur_Z_vs_phi.png"), dpi=150)
    plt.close(fig)

    # contraste max / fond
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom in noms:
        r = contrastes[contrastes.focus == nom].sort_values("phi")
        ax.plot(r.phi, r.rapport_max_fond, label=nom, **style[nom])
    ax.set_xlabel("phi, angle a l'axe optique (deg)  -  0 = dans l'axe, 90 = couchee")
    ax.set_ylabel("max du focus / moyenne du fond")
    ax.set_title("Contraste du focus selon l'orientation")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "contraste_vs_phi.png"), dpi=150)
    plt.close(fig)

    # meme contraste, divise par sa valeur a phi = 0: compare les variations relatives
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom in noms:
        r = contrastes[contrastes.focus == nom].sort_values("phi")
        ax.plot(r.phi, r.rapport_max_fond / r.rapport_max_fond.iloc[0], label=nom,
                **style[nom])
    ax.axhline(1, color="#9c9c9c", linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("phi, angle a l'axe optique (deg)  -  0 = dans l'axe, 90 = couchee")
    ax.set_ylabel("(max / fond) rapporte a sa valeur a phi = 0")
    ax.set_title("Contraste du focus normalise: variation relative selon l'orientation")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "contraste_normalise_vs_phi.png"), dpi=150)
    plt.close(fig)

    # contraste en ecarts-types: le K au-dela duquel l'objet disparait du seuil global
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom in noms:
        r = contrastes[contrastes.focus == nom].sort_values("phi")
        ax.plot(r.phi, r.k_limite, label="{} (K retenu = {:g})".format(nom, choix[nom]),
                **style[nom])
    ax.set_xlabel("phi, angle a l'axe optique (deg)  -  0 = dans l'axe, 90 = couchee")
    ax.set_ylabel("(max - moyenne) / ecart-type du volume")
    ax.set_title("Contraste du focus en ecarts-types: K maximal avant de perdre la bacterie")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "contraste_sigma_vs_phi.png"), dpi=150)
    plt.close(fig)

    # F1 moyen selon K: la justification du K retenu
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom in noms:
        g = par_k[par_k.focus == nom].sort_values("K")
        ax.plot(g.K, g.f1, label=legende(nom), **dict(style[nom], markersize=4))
        ax.plot([choix[nom]], g[g.K == choix[nom]].f1, marker=style[nom]["marker"],
                markersize=11, color=style[nom]["color"], markeredgecolor="white",
                markeredgewidth=2, linestyle="none")
    ax.set_xscale("log")
    ax.set_xlabel("K du seuil global (moyenne + K x ecart-type du volume)")
    ax.set_ylabel("F1 moyen sur les 19 angles")
    ax.set_title("Choix du K: F1 moyen selon K (gros marqueur = K retenu)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "F1_vs_K.png"), dpi=150)
    plt.close(fig)


def main(sans_mesure):
    if sans_mesure:          # reprend les mesures deja faites, refait seulement l'analyse
        balayage = pd.read_csv(os.path.join(ICI, "balayage_K.csv"))
        contrastes = pd.read_csv(os.path.join(ICI, "contraste_focus.csv"))
    else:
        balayage, contrastes = mesure()
        balayage.to_csv(os.path.join(ICI, "balayage_K.csv"), index=False)
        contrastes.to_csv(os.path.join(ICI, "contraste_focus.csv"), index=False)

    choix, plateaux, par_k = k_optimal(balayage)
    res = pd.concat([balayage[(balayage.focus == nom) & (balayage.K == K)]
                     for nom, K in choix.items()])
    res = res.merge(contrastes[["phi", "focus", "rapport_max_fond", "k_limite"]],
                    on=["phi", "focus"])
    res.to_csv(os.path.join(ICI, "resultats_K_optimal.csv"), index=False)

    print("\n{:<18}{:>8}{:>18}{:>7}{:>9}{:>11}{:>11}{:>11}{:>12}".format(
        "focus", "K", "plateau F1 max", "F1", "trouves", "|eX| um", "|eY| um", "|eZ| um",
        "focus (ms)"))
    for f in FOCUS:
        r = res[res.focus == f.name]
        t = contrastes[contrastes.focus == f.name].temps_focus_ms.median()
        k_min, k_max, _ = plateaux[f.name]
        print("{:<18}{:>8g}{:>24}{:>7.3f}{:>6}/{:<2}{:>11.3f}{:>11.3f}{:>11.3f}{:>12.0f}".format(
            f.name, choix[f.name], "{:g}-{:g} (x{:.1f})".format(k_min, k_max, k_max / k_min),
            r.f1.mean(), int(r.trouve.sum()), len(r),
            np.nanmean(np.abs(r.ex)) * 1e6, np.nanmean(np.abs(r.ey)) * 1e6,
            np.nanmean(np.abs(r.ez)) * 1e6, t))

    graphes(res, contrastes, par_k, choix)
    print("\nresultats dans {}".format(ICI))
    return 0


if __name__ == "__main__":
    sys.exit(main("--sans-mesure" in sys.argv))
