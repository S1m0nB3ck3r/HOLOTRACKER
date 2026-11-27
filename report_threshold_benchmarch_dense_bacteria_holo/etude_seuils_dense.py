# -*- coding: utf-8 -*-

"""
Filename: etude_seuils_dense.py

Description:
Comparaison des quatre types de seuillage d'HOLOTRACKER, chacun suivi de la detection par
composantes connexes (CCL3D + barycentres), avec le critere de focus SUM_OF_GRADIENT, sur
les hologrammes denses de 'SIMULATED HOLO/' (N = 10 a 500 bacteries, 10 hologrammes par
densite). Le K de chaque type de seuil est optimise **pour chaque densite**.

    global          moyenne + K x ecart-type du volume entier          (calc_threshold)
    local sigma     fond ecrete + K x max(sigma, plancher), par maille (mailles_fond_sigma)
    local MAD       mediane + K x 1,4826 x MAD, par maille              (mailles_fond_mad,
                                                                         defaut du GUI)
    local mediane   mediane x (1 + K), par maille                       (mailles_fond_median)

Les mailles sont celles du GUI: 25 plans sur 64 x 64 pixels. Les seuils locaux sont
interpoles a tout le volume par calc_threshold_local, comme dans core.py.

Pour chaque hologramme:
  1. lecture et nettoyage par benchmark_detection.lire_simulation (division par la moyenne,
     positions vraies dans le repere HOLOTRACKER: X colonnes, Y lignes, Z depuis le capteur) ;
  2. volume de module par spectre angulaire, passe-haut seul (15), puis SUM_OF_GRADIENT ;
  3. les statistiques de mailles de chaque seuil local, une fois: elles ne dependent pas de K ;
  4. pour chaque type de seuil et chaque K: seuil, CCL3D, barycentres, appariement a la
     verite (benchmark_detection.apparie: au plus proche, 3 um en XY, 5 um en Z).

Le K retenu pour un (type de seuil, densite) est celui qui maximise le F1 moyen des 10
hologrammes de cette densite. A titre de comparaison, on note aussi le meilleur K unique
pour toutes les densites, celui qu'il faudrait choisir pour un film dont on ignore la
densite: un seuil dont le K optimal bouge peu avec la densite perd peu a ce choix.

Sorties, dans le repertoire de ce script:
    seuils_balayage_K.csv          une ligne par (hologramme, type de seuil, K)
    seuils_K_optimal_par_densite.csv  une ligne par (type de seuil, densite), au K optimal
    seuils_K_unique.csv            une ligne par (type de seuil, densite), au K unique
    seuils_temps.csv               temps des statistiques et du seuil, par type
    seuils_detection_vs_densite.png    precision, rappel, F1 (K par densite et K unique)
    seuils_K_optimal_vs_densite.png    le K optimal selon la densite, par type
    seuils_rappel_orientation.png      rappel des bacteries couchees et debout
    seuils_erreur_XYZ_vs_densite.png   biais et ecart-type en X, Y, Z

Usage:
    python report_threshold_benchmarch_dense_bacteria_holo/etude_seuils_dense.py
    python report_threshold_benchmarch_dense_bacteria_holo/etude_seuils_dense.py --sans-mesure
        (refait l'analyse et les graphes a partir des CSV, sans GPU)

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import os
import sys
import time

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
from CCL3D import (CCA_CUDA_float, CCL3D, calc_threshold, calc_threshold_local,
                   mailles_fond_mad, mailles_fond_median, mailles_fond_sigma, type_threshold)

import benchmark_detection as banc          # lecture, appariement, optique du banc

# --- parametres -------------------------------------------------------------------------
SIMU = os.path.join(RACINE, "SIMULATED HOLO")

# optique et volume: ceux du banc d'essai (240 plans de 0,5 um depuis le capteur)
NB_PIX = banc.NB_PIX_X
NB_PLANE, DZ, DISTANCE_INI = banc.NB_PLANE, banc.DZ, banc.DISTANCE_INI
DX = banc.DX
FILTRE = (15, 0)                # passe-haut seul (0 = pas de passe-bas)
TYPE_FOCUS = Focus_type.SUM_OF_GRADIENT
SUM_SIZE = 15
CONNECTIVITE = 26

# mailles du seuil local: celles du GUI (core.MAILLE_XY, core.MAILLE_Z)
MAILLE_XY, MAILLE_Z = 64, 25
# ecretage du seuil local sigma: valeurs des scripts hors interface
KAPPA, NB_PASSES, PERCENTILE_PLANCHER, PAS_ECHANTILLON = 3.0, 3, 10, 1

# K balayes, echelle log. Les trois premiers comptent des ecarts-types; le seuil mediane
# multiplie le fond, son echelle est autre.
GRILLES_K = {
    "global": np.round(np.geomspace(2.5, 150.0, 16), 2),
    "local sigma": np.round(np.geomspace(2.5, 150.0, 16), 2),
    "local MAD": np.round(np.geomspace(2.5, 150.0, 16), 2),
    "local mediane": np.round(np.geomspace(0.3, 40.0, 16), 3),
}
SEUILS = tuple(GRILLES_K)

# Au-dela de ce nombre de detections par objet vrai, la precision est sous 5 % et
# l'appariement glouton deviendrait tres long: le K est note sans appariement, F1 nul.
DETECTIONS_MAX_PAR_OBJET = 20
PHI_COUCHEE, PHI_DEBOUT = 60.0, 30.0    # couchee: phi > 60 deg ; debout: phi < 30 deg

COULEURS = ("#257de0", "#f47230", "#00b894", "#f2a600")
MARQUEURS = ("o", "s", "^", "D")


def top():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def statistiques_mailles(d_focus, temps):
    """Grilles de fond et de dispersion des trois seuils locaux, calculees une fois."""
    t0 = top()
    sigma = mailles_fond_sigma(d_focus, 1.0, MAILLE_XY, MAILLE_Z, KAPPA, NB_PASSES,
                               PERCENTILE_PLANCHER, PAS_ECHANTILLON)
    t1 = top()
    mad = mailles_fond_mad(d_focus, 1.0, MAILLE_XY, MAILLE_Z)
    t2 = top()
    mediane = mailles_fond_median(d_focus, 0.0, MAILLE_XY, MAILLE_Z)
    t3 = top()
    cp.get_default_memory_pool().free_all_blocks()
    temps.append(dict(etape="mailles local sigma", ms=1000 * (t1 - t0)))
    temps.append(dict(etape="mailles local MAD", ms=1000 * (t2 - t1)))
    temps.append(dict(etape="mailles local mediane", ms=1000 * (t3 - t2)))
    return {"local sigma": sigma, "local MAD": mad, "local mediane": mediane}


def seuil_pour(type_seuil, K, d_focus, grilles, temps):
    """Le seuil passe a CCL3D: un flottant (global) ou un volume interpole (local)."""
    t0 = top()
    if type_seuil == "global":
        seuil = calc_threshold(d_focus, K)
    else:
        g = grilles[type_seuil]
        infos = dict(g)
        if type_seuil == "local sigma":
            infos["seuil_mailles"] = g["fond"] + np.float32(K) * cp.maximum(
                g["sigma"], cp.float32(g["sigma_plancher"]))
        elif type_seuil == "local MAD":
            infos["seuil_mailles"] = g["fond"] + np.float32(K) * g["sigma"]
        else:                                                   # local mediane
            infos["seuil_mailles"] = g["fond"] * np.float32(1.0 + K)
        seuil, _ = calc_threshold_local(d_focus, K, MAILLE_XY, MAILLE_Z, infos=infos)
    temps.append(dict(etape="seuil " + type_seuil, ms=1000 * (top() - t0)))
    return seuil


def detecte(d_focus, d_binaire, seuil):
    """CCL3D + barycentres: positions (n, 3) en metres."""
    lab, nb = CCL3D(d_binaire, d_focus, type_threshold.THRESHOLD, seuil, CONNECTIVITE)
    if isinstance(seuil, cp.ndarray):
        del seuil
    if nb == 0:
        del lab
        cp.get_default_memory_pool().free_all_blocks()
        return np.zeros((0, 3))
    f = CCA_CUDA_float(lab, d_focus, nb, 1, NB_PIX, NB_PIX, NB_PLANE, DX, DX, DZ,
                       z_offset=DISTANCE_INI)
    del lab
    cp.get_default_memory_pool().free_all_blocks()
    return f[:, 1:4]


def mesure():
    bufs = [cp.zeros((NB_PIX, NB_PIX), dtype=t) for t in
            (cp.complex64,) * 3 + (cp.float32,) + (cp.complex64,) * 2]
    d_volume = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.float32)
    d_binaire = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.bool_)

    print("volume {} plans de {:.1f} um depuis {:.1f} um, filtre {}, {} sum_size {},"
          " mailles {} x {} x {}".format(NB_PLANE, DZ * 1e6, DISTANCE_INI * 1e6, FILTRE,
                                         TYPE_FOCUS.name, SUM_SIZE, MAILLE_Z, MAILLE_XY,
                                         MAILLE_XY))
    lignes, temps = [], []
    for dossier in banc.dossiers_simulation(SIMU):
        for n in banc.numeros_hologrammes(dossier):
            h_net, vrai, phi = banc.lire_simulation(dossier, n)
            couchee, debout = phi > PHI_COUCHEE, phi < PHI_DEBOUT
            propag.volume_propag_angular_spectrum_to_module(
                cp.asarray(h_net), *bufs, d_volume, banc.WAVELENGTH / banc.INDEX_MEDIUM,
                banc.MAGNIFICATION, banc.PIX_SIZE, NB_PIX, NB_PIX, DISTANCE_INI, DZ,
                NB_PLANE, *FILTRE)
            # en place, comme le GUI: le volume de module devient le volume de focus
            focus.focus(d_volume, d_volume, SUM_SIZE, TYPE_FOCUS)
            maximum = float(cp.max(d_volume))
            grilles = statistiques_mailles(d_volume, temps)

            for type_seuil in SEUILS:
                for K in GRILLES_K[type_seuil]:
                    ligne = dict(densite=len(vrai), hologramme=n, seuil=type_seuil, K=K,
                                 detections=0, appariees=0, rappel=0.0, precision=np.nan,
                                 f1=0.0, couchees=int(couchee.sum()), couchees_trouvees=0,
                                 debout=int(debout.sum()), debout_trouvees=0)
                    seuil = seuil_pour(type_seuil, K, d_volume, grilles, temps)
                    if type_seuil == "global" and seuil >= maximum:
                        lignes.append(ligne)               # plus rien ne passe le seuil
                        continue
                    det = detecte(d_volume, d_binaire, seuil)
                    del seuil
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
            del grilles
            cp.get_default_memory_pool().free_all_blocks()
            print("   N = {} hologramme {} fait".format(len(vrai), n))
    return pd.DataFrame(lignes), pd.DataFrame(temps)


def regroupe(b):
    """Indicateurs d'un groupe d'hologrammes (une densite) a un K donne."""
    n_app = b.appariees.sum()
    ligne = dict(rappel=b.rappel.mean(), precision=b.precision.mean(), f1=b.f1.mean(),
                 detections_par_holo=b.detections.mean(), appariees=int(n_app),
                 rappel_couchees=b.couchees_trouvees.sum() / b.couchees.sum(),
                 rappel_debout=b.debout_trouvees.sum() / b.debout.sum())
    for col in ("x", "y", "z"):
        biais = b["somme_e" + col].sum() / n_app if n_app else np.nan
        ligne["biais_" + col] = biais
        ligne["ecart_type_" + col] = (np.sqrt(b["somme_e2" + col].sum() / n_app - biais ** 2)
                                      if n_app else np.nan)
    return ligne


def analyse(balayage):
    """K optimal par (seuil, densite), et meilleur K unique par seuil."""
    f1 = balayage.groupby(["seuil", "densite", "K"]).f1.mean().reset_index()
    par_densite, unique = [], []
    for type_seuil in SEUILS:
        f = f1[f1.seuil == type_seuil]
        # K unique: F1 moyen sur les densites maximal
        k_unique = float(f.groupby("K").f1.mean().idxmax())
        for densite, g in f.groupby("densite"):
            k_opt = float(g.loc[g.f1.idxmax(), "K"])
            b = balayage[(balayage.seuil == type_seuil) & (balayage.densite == densite)]
            par_densite.append(dict(seuil=type_seuil, densite=densite, K=k_opt,
                                    **regroupe(b[b.K == k_opt])))
            unique.append(dict(seuil=type_seuil, densite=densite, K=k_unique,
                               **regroupe(b[b.K == k_unique])))
    return pd.DataFrame(par_densite), pd.DataFrame(unique)


def graphes(opt, uni):
    densites = sorted(opt.densite.unique())
    style = {s: dict(color=COULEURS[i], marker=MARQUEURS[i], markersize=6, linewidth=1.5)
             for i, s in enumerate(SEUILS)}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9c9c9c", "font.size": 10})

    def axe_densite(ax):
        ax.set_xscale("log")
        ax.set_xticks(densites)
        ax.set_xticklabels([str(d) for d in densites], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("bacteries par hologramme")

    # precision, rappel, F1: trait plein = K optimal par densite, tirets = K unique
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, col, titre in zip(axes, ("precision", "rappel", "f1"),
                              ("Precision", "Rappel", "F1")):
        for s in SEUILS:
            o = opt[opt.seuil == s].sort_values("densite")
            u = uni[uni.seuil == s].sort_values("densite")
            ax.plot(o.densite, o[col], label=s + " (K par densite)", **style[s])
            ax.plot(u.densite, u[col], linestyle="--", linewidth=1.2, color=style[s]["color"],
                    marker=style[s]["marker"], markersize=4, markerfacecolor="white",
                    label="{} (K unique = {:g})".format(s, u.K.iloc[0]))
        axe_densite(ax)
        ax.set_ylim(0, 1.02)
        ax.set_title(titre)
    axes[0].set_ylabel("moyenne sur les 10 hologrammes")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left", ncol=1)
    fig.suptitle("Types de seuil, {} + CCL: K optimise par densite (plein) ou unique"
                 " (tirets)".format(TYPE_FOCUS.name))
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "seuils_detection_vs_densite.png"), dpi=150)
    plt.close(fig)

    # K optimal selon la densite: deux panneaux, le seuil mediane n'a pas la meme echelle
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for s in SEUILS:
        o = opt[opt.seuil == s].sort_values("densite")
        ax = axes[1] if s == "local mediane" else axes[0]
        ax.plot(o.densite, o.K, label=s, **style[s])
    for ax in axes:
        axe_densite(ax)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("K optimal (en ecarts-types)")
    axes[1].set_ylabel("K optimal (seuil = mediane x (1 + K))")
    axes[0].set_title("global, local sigma, local MAD")
    axes[1].set_title("local mediane")
    fig.suptitle("K optimal selon la densite: plus il est stable, mieux un K unique convient")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "seuils_K_optimal_vs_densite.png"), dpi=150)
    plt.close(fig)

    # rappel selon l'orientation, au K optimal par densite
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for s in SEUILS:
        o = opt[opt.seuil == s].sort_values("densite")
        axes[0].plot(o.densite, o.rappel_couchees, label=s, **style[s])
        axes[1].plot(o.densite, o.rappel_debout, **style[s])
        axes[2].plot(o.densite, o.rappel_couchees / o.rappel_debout, **style[s])
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
    fig.suptitle("Bacteries couchees et debout, K optimal par densite")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "seuils_rappel_orientation.png"), dpi=150)
    plt.close(fig)

    # biais et ecart-type en X, Y, Z, au K optimal par densite
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for j, (col, titre) in enumerate((("x", "X (horizontal)"), ("y", "Y (vertical)"),
                                      ("z", "Z (profondeur)"))):
        for s in SEUILS:
            o = opt[opt.seuil == s].sort_values("densite")
            axes[0, j].plot(o.densite, o["biais_" + col] * 1e6, label=s, **style[s])
            axes[1, j].plot(o.densite, o["ecart_type_" + col] * 1e6, **style[s])
        axes[0, j].axhline(0, color="#9c9c9c", linewidth=0.8)
        axes[0, j].set_title("erreur en " + titre)
        axes[1, j].set_ylim(bottom=0)
        axe_densite(axes[1, j])
    axes[0, 0].set_ylabel("biais, moyenne signee (um)")
    axes[1, 0].set_ylabel("ecart-type (um)")
    axes[0, 2].legend(fontsize=8, frameon=False)
    fig.suptitle("Erreur de position des detections appariees, K optimal par densite")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "seuils_erreur_XYZ_vs_densite.png"), dpi=150)
    plt.close(fig)


def main(sans_mesure):
    chemin = os.path.join(ICI, "seuils_balayage_K.csv")
    chemin_temps = os.path.join(ICI, "seuils_temps.csv")
    if sans_mesure:
        balayage = pd.read_csv(chemin)
        temps = pd.read_csv(chemin_temps)
    else:
        balayage, temps = mesure()
        balayage.to_csv(chemin, index=False)
        temps.to_csv(chemin_temps, index=False)

    opt, uni = analyse(balayage)
    opt.to_csv(os.path.join(ICI, "seuils_K_optimal_par_densite.csv"), index=False)
    uni.to_csv(os.path.join(ICI, "seuils_K_unique.csv"), index=False)

    for type_seuil in SEUILS:
        o = opt[opt.seuil == type_seuil].sort_values("densite")
        u = uni[uni.seuil == type_seuil].sort_values("densite")
        print("\n{}  (K unique {:g}: F1 moyen {:.3f} ; K par densite: F1 moyen {:.3f})".format(
            type_seuil, u.K.iloc[0], u.f1.mean(), o.f1.mean()))
        print("{:>6}{:>8}{:>11}{:>8}{:>7}{:>10}{:>8}{:>10}{:>18}{:>18}{:>18}".format(
            "N", "K opt", "precision", "rappel", "F1", "couchees", "debout", "F1 K uniq",
            "X biais/sigma um", "Y biais/sigma um", "Z biais/sigma um"))
        for (_, l), (_, lu) in zip(o.iterrows(), u.iterrows()):
            print("{:>6}{:>8g}{:>11.3f}{:>8.3f}{:>7.3f}{:>10.3f}{:>8.3f}{:>10.3f}".format(
                l.densite, l.K, l.precision, l.rappel, l.f1, l.rappel_couchees,
                l.rappel_debout, lu.f1) + "".join(
                "{:>10.3f}/{:<7.3f}".format(l["biais_" + c] * 1e6, l["ecart_type_" + c] * 1e6)
                for c in ("x", "y", "z")))

    print("\nTemps median par hologramme (ms)")
    print(temps.groupby("etape").ms.median().round(0).to_string())

    graphes(opt, uni)
    print("\nresultats dans {}".format(ICI))
    return 0


if __name__ == "__main__":
    sys.exit(main("--sans-mesure" in sys.argv))
