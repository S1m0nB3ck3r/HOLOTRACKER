# -*- coding: utf-8 -*-

"""
Filename: etude_detection_dense.py

Description:
Precision, rappel, F1 et erreur de position en X, Y, Z de la detection par seuil global +
CCL3D, pour chacune des cinq fonctions de focus elementaires, sur les hologrammes denses de
'SIMULATED HOLO/' (N = 10 a 500 bacteries par hologramme, 10 hologrammes par densite).

Pour chaque hologramme:
  1. lecture et nettoyage par benchmark_detection.lire_simulation (division par la moyenne,
     positions vraies dans le repere HOLOTRACKER: X colonnes, Y lignes, Z depuis le capteur) ;
  2. un seul volume de module, reconstruit par spectre angulaire, passe-haut seul (15) ;
  3. pour chaque fonction de focus, le volume de focus, puis pour chaque K de la grille:
     seuil global (moyenne + K x ecart-type du volume), CCL3D, barycentres (CCA), comme
     dans HOLOTRACKER, et appariement a la verite par benchmark_detection.apparie
     (au plus proche, a moins de 3 um en XY et 5 um en Z).

Le K est optimise **par fonction de focus, un seul K pour toutes les densites**, comme
HOLOTRACKER applique un seul K a tout un film: celui qui maximise le F1 moyen sur les
densites (F1 moyen des hologrammes de chaque densite, puis moyenne des densites).

L'erreur de position est celle des detections appariees: biais (moyenne signee) et
ecart-type, par axe, regroupes sur tous les hologrammes d'une densite.

Le rappel est aussi mesure selon l'orientation: bacteries couchees (phi > 60 deg, environ la
moitie avec un tirage isotrope) et debout (phi < 30 deg, environ 13 %), pour voir si une
fonction de focus perd preferentiellement les bacteries couchees.

Sorties, dans le repertoire de ce script:
    detection_balayage_K.csv     une ligne par (hologramme, focus, K)
    detection_K_optimal.csv      une ligne par (densite, focus), au K retenu
    detection_vs_densite.png     precision, rappel, F1 selon la densite
    rappel_orientation_vs_densite.png  rappel des couchees, des debout, et leur rapport
    erreur_XYZ_vs_densite.png    biais et ecart-type en X, Y, Z selon la densite
    detection_F1_vs_K.png        F1 moyen selon K: la justification du K retenu

Usage:
    python report_focus_dense_holo/etude_detection_dense.py
    python report_focus_dense_holo/etude_detection_dense.py --sans-mesure
        (refait l'analyse et les graphes a partir du CSV, sans GPU)

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(RACINE, "code_locate"))
sys.path.insert(0, os.path.join(RACINE, "utils"))

# Les reductions CuPy passent sinon par CUB, que nvcc doit compiler, et nvcc ne compile pas
# sur la machine de developpement (voir CLAUDE.md). A fixer avant l'import de cupy.
os.environ["CUPY_ACCELERATORS"] = ""

import cupy as cp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import focus
import propagation as propag
from focus import Focus_type
from CCL3D import CCA_CUDA_float, CCL3D, type_threshold

import benchmark_detection as banc          # lecture, appariement, optique du banc

# --- parametres -------------------------------------------------------------------------
SIMU = os.path.join(RACINE, "SIMULATED HOLO")

# optique et volume: ceux du banc d'essai (240 plans de 0,5 um depuis le capteur)
NB_PIX = banc.NB_PIX_X
NB_PLANE, DZ, DISTANCE_INI = banc.NB_PLANE, banc.DZ, banc.DISTANCE_INI
DX = banc.DX
FILTRE = (15, 0)                # passe-haut seul (0 = pas de passe-bas)
SUM_SIZE = banc.SUM_SIZE        # 15
CONNECTIVITE = 26

FOCUS = (Focus_type.SUM_OF_VARIANCE, Focus_type.TENEGRAD, Focus_type.SUM_OF_LAPLACIAN,
         Focus_type.SUM_OF_GRADIENT, Focus_type.SUM_OF_INTENSITY)

GRILLE_K = np.round(np.geomspace(2.5, 250.0, 16), 2)    # K balayes, echelle log

# Au-dela de ce nombre de detections par objet vrai, la precision est sous 5 % et
# l'appariement glouton (matrice objets x detections) deviendrait tres long: le K est note
# sans appariement, avec un F1 compte nul.
DETECTIONS_MAX_PAR_OBJET = 20

PHI_COUCHEE, PHI_DEBOUT = 60.0, 30.0    # couchee: phi > 60 deg ; debout: phi < 30 deg

COULEURS = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac")
MARQUEURS = ("o", "s", "^", "D", "v")


def detecte(d_focus, d_binaire, seuil):
    """Seuil global + composantes connexes + barycentres: positions (n, 3) en metres."""
    lab, nb = CCL3D(d_binaire, d_focus, type_threshold.THRESHOLD, seuil, CONNECTIVITE)
    if nb == 0:
        del lab
        return np.zeros((0, 3))
    f = CCA_CUDA_float(lab, d_focus, nb, 1, NB_PIX, NB_PIX, NB_PLANE, DX, DX, DZ,
                       z_offset=DISTANCE_INI)
    del lab
    cp.get_default_memory_pool().free_all_blocks()
    return f[:, 1:4]


def mesure():
    bufs = [cp.zeros((NB_PIX, NB_PIX), dtype=t) for t in
            (cp.complex64,) * 3 + (cp.float32,) + (cp.complex64,) * 2]
    d_module = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.float32)
    d_focus = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.float32)
    d_binaire = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.bool_)

    print("volume {} plans de {:.1f} um depuis {:.1f} um, filtre {}, sum_size {}, {} K".format(
        NB_PLANE, DZ * 1e6, DISTANCE_INI * 1e6, FILTRE, SUM_SIZE, len(GRILLE_K)))
    lignes = []
    for dossier in banc.dossiers_simulation(SIMU):
        for n in banc.numeros_hologrammes(dossier):
            h_net, vrai, phi = banc.lire_simulation(dossier, n)
            couchee, debout = phi > PHI_COUCHEE, phi < PHI_DEBOUT
            propag.volume_propag_angular_spectrum_to_module(
                cp.asarray(h_net), *bufs, d_module, banc.WAVELENGTH / banc.INDEX_MEDIUM,
                banc.MAGNIFICATION, banc.PIX_SIZE, NB_PIX, NB_PIX, DISTANCE_INI, DZ,
                NB_PLANE, *FILTRE)

            for type_focus in FOCUS:
                focus.focus(d_module, d_focus, SUM_SIZE, type_focus)
                moyenne = float(cp.mean(d_focus))
                ecart_type = float(cp.std(d_focus))
                maximum = float(cp.max(d_focus))
                cp.get_default_memory_pool().free_all_blocks()

                for K in GRILLE_K:
                    seuil = moyenne + K * ecart_type
                    ligne = dict(densite=len(vrai), hologramme=n, focus=type_focus.name,
                                 K=K, detections=0, appariees=0, rappel=0.0,
                                 precision=np.nan, f1=0.0,
                                 couchees=int(couchee.sum()), couchees_trouvees=0,
                                 debout=int(debout.sum()), debout_trouvees=0)
                    if seuil < maximum:
                        det = detecte(d_focus, d_binaire, seuil)
                        ligne["detections"] = len(det)
                        if len(det) > DETECTIONS_MAX_PAR_OBJET * len(vrai):
                            ligne.update(rappel=np.nan, couchees_trouvees=np.nan,
                                         debout_trouvees=np.nan)   # non apparie, F1 nul
                        elif len(det):
                            trouve, n_app, ecarts = banc.apparie(det, vrai)
                            r, p = trouve.mean(), n_app / len(det)
                            ligne.update(appariees=n_app, rappel=r, precision=p,
                                         f1=2 * r * p / (r + p) if r + p else 0.0,
                                         couchees_trouvees=int(trouve[couchee].sum()),
                                         debout_trouvees=int(trouve[debout].sum()))
                            # sommes pour regrouper biais et ecart-type sur une densite
                            for axe, col in enumerate(("x", "y", "z")):
                                e = ecarts[:, axe] if n_app else np.zeros(0)
                                ligne["somme_e" + col] = float(e.sum())
                                ligne["somme_e2" + col] = float((e ** 2).sum())
                    lignes.append(ligne)
            cp.get_default_memory_pool().free_all_blocks()
            print("   N = {} hologramme {} fait".format(len(vrai), n))
    return pd.DataFrame(lignes)


def k_optimal(balayage):
    """Un K par focus: celui qui maximise le F1 moyen sur les densites."""
    par_densite = balayage.groupby(["focus", "K", "densite"]).f1.mean().reset_index()
    par_k = par_densite.groupby(["focus", "K"]).f1.mean().reset_index()
    choix = {nom: float(g.loc[g.f1.idxmax(), "K"]) for nom, g in par_k.groupby("focus")}
    return choix, par_k


def resultats_au_k(balayage, choix):
    """Precision, rappel, F1 et erreurs regroupees par densite, au K retenu de chaque focus."""
    lignes = []
    for nom, K in choix.items():
        b = balayage[(balayage.focus == nom) & (balayage.K == K)]
        for densite, g in b.groupby("densite"):
            n_app = g.appariees.sum()
            ligne = dict(focus=nom, K=K, densite=densite, rappel=g.rappel.mean(),
                         precision=g.precision.mean(), f1=g.f1.mean(),
                         detections_par_holo=g.detections.mean(), appariees=int(n_app),
                         rappel_couchees=g.couchees_trouvees.sum() / g.couchees.sum(),
                         rappel_debout=g.debout_trouvees.sum() / g.debout.sum(),
                         nb_couchees=int(g.couchees.sum()), nb_debout=int(g.debout.sum()))
            for col in ("x", "y", "z"):
                biais = g["somme_e" + col].sum() / n_app
                ligne["biais_" + col] = biais
                ligne["ecart_type_" + col] = np.sqrt(g["somme_e2" + col].sum() / n_app
                                                     - biais ** 2)
            lignes.append(ligne)
    return pd.DataFrame(lignes)


def graphes(res, par_k, choix):
    noms = [f.name for f in FOCUS]
    densites = sorted(res.densite.unique())
    style = {nom: dict(color=COULEURS[i], marker=MARQUEURS[i], markersize=6, linewidth=1.5)
             for i, nom in enumerate(noms)}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9c9c9c", "font.size": 10})

    def axe_densite(ax):
        ax.set_xscale("log")
        ax.set_xticks(densites)
        ax.set_xticklabels([str(d) for d in densites], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("bacteries par hologramme")

    def legende(nom):
        return "{} (K = {:g})".format(nom, choix[nom])

    # precision, rappel, F1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, col, titre in zip(axes, ("precision", "rappel", "f1"),
                              ("Precision", "Rappel", "F1")):
        for nom in noms:
            r = res[res.focus == nom].sort_values("densite")
            ax.plot(r.densite, r[col], label=legende(nom), **style[nom])
        axe_densite(ax)
        ax.set_ylim(0, 1.02)
        ax.set_title(titre)
    axes[0].set_ylabel("moyenne sur les 10 hologrammes")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Detection par seuil global + CCL au K optimal de chaque focus")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "detection_vs_densite.png"), dpi=150)
    plt.close(fig)

    # rappel selon l'orientation
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for nom in noms:
        r = res[res.focus == nom].sort_values("densite")
        axes[0].plot(r.densite, r.rappel_couchees, label=legende(nom), **style[nom])
        axes[1].plot(r.densite, r.rappel_debout, **style[nom])
        axes[2].plot(r.densite, r.rappel_couchees / r.rappel_debout, **style[nom])
    for ax, titre in zip(axes, ("Rappel des couchees (phi > 60 deg)",
                                "Rappel des debout (phi < 30 deg)",
                                "Rapport couchees / debout")):
        axe_densite(ax)
        ax.set_title(titre)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].axhline(1, color="#9c9c9c", linewidth=0.8)
    axes[2].set_ylim(0, max(1.1, axes[2].get_ylim()[1]))
    axes[0].set_ylabel("part des bacteries retrouvees")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Les bacteries couchees sont-elles perdues plus souvent ? (K optimal de chaque focus)")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "rappel_orientation_vs_densite.png"), dpi=150)
    plt.close(fig)

    # biais et ecart-type en X, Y, Z
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for j, (col, titre) in enumerate((("x", "X (horizontal)"), ("y", "Y (vertical)"),
                                      ("z", "Z (profondeur)"))):
        for nom in noms:
            r = res[res.focus == nom].sort_values("densite")
            axes[0, j].plot(r.densite, r["biais_" + col] * 1e6, label=legende(nom),
                            **style[nom])
            axes[1, j].plot(r.densite, r["ecart_type_" + col] * 1e6, **style[nom])
        axes[0, j].axhline(0, color="#9c9c9c", linewidth=0.8)
        axes[0, j].set_title("erreur en " + titre)
        axes[1, j].set_ylim(bottom=0)
        axe_densite(axes[1, j])
    axes[0, 0].set_ylabel("biais, moyenne signee (um)")
    axes[1, 0].set_ylabel("ecart-type (um)")
    axes[0, 2].legend(fontsize=8, frameon=False)
    fig.suptitle("Erreur de position des detections appariees, au K optimal de chaque focus")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "erreur_XYZ_vs_densite.png"), dpi=150)
    plt.close(fig)

    # F1 moyen selon K
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom in noms:
        g = par_k[par_k.focus == nom].sort_values("K")
        ax.plot(g.K, g.f1, label=legende(nom), **dict(style[nom], markersize=4))
        ax.plot([choix[nom]], g[g.K == choix[nom]].f1, marker=style[nom]["marker"],
                markersize=11, color=style[nom]["color"], markeredgecolor="white",
                markeredgewidth=2, linestyle="none")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("K du seuil global (moyenne + K x ecart-type du volume)")
    ax.set_ylabel("F1 moyen sur les densites")
    ax.set_title("Choix du K: F1 moyen selon K (gros marqueur = K retenu)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "detection_F1_vs_K.png"), dpi=150)
    plt.close(fig)


def main(sans_mesure):
    chemin = os.path.join(ICI, "detection_balayage_K.csv")
    if sans_mesure:
        balayage = pd.read_csv(chemin)
    else:
        balayage = mesure()
        balayage.to_csv(chemin, index=False)

    choix, par_k = k_optimal(balayage)
    res = resultats_au_k(balayage, choix)
    res.to_csv(os.path.join(ICI, "detection_K_optimal.csv"), index=False)

    for nom in [f.name for f in FOCUS]:
        r = res[res.focus == nom].sort_values("densite")
        print("\n{}  (K = {:g}, F1 moyen sur les densites {:.3f})".format(
            nom, choix[nom], r.f1.mean()))
        print("{:>6}{:>11}{:>8}{:>7}{:>10}{:>8}{:>22}{:>22}{:>22}".format(
            "N", "precision", "rappel", "F1", "couchees", "debout", "X biais +/- sigma um",
            "Y biais +/- sigma um", "Z biais +/- sigma um"))
        for _, l in r.iterrows():
            print("{:>6}{:>11.3f}{:>8.3f}{:>7.3f}{:>10.3f}{:>8.3f}".format(
                l.densite, l.precision, l.rappel, l.f1, l.rappel_couchees,
                l.rappel_debout) + "".join(
                "{:>13.3f} +/-{:5.3f}".format(l["biais_" + c] * 1e6,
                                              l["ecart_type_" + c] * 1e6)
                for c in ("x", "y", "z")))

    graphes(res, par_k, choix)
    print("\nresultats dans {}".format(ICI))
    return 0


if __name__ == "__main__":
    sys.exit(main("--sans-mesure" in sys.argv))
