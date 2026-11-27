# -*- coding: utf-8 -*-

"""
Filename: etude_contraste_dense.py

Description:
Contraste de chaque fonction de focus a la position des bacteries, sur les hologrammes
denses de 'SIMULATED HOLO/' (N = 10 a 500 bacteries par hologramme, 10 hologrammes par
densite), par rapport au fond diffus du volume.

Pour chaque hologramme:
  1. lecture et nettoyage par benchmark_detection.lire_simulation (division par la moyenne,
     positions vraies dans le repere HOLOTRACKER: X colonnes, Y lignes, Z depuis le capteur) ;
  2. un seul volume de module, reconstruit par spectre angulaire, passe-haut seul (15) ;
  3. pour chaque fonction de focus, le volume de focus, puis:
       - le fond diffus: mediane du volume de focus (sous-echantillonne). Les bacteries
         occupent moins de 0,1 % des voxels: la mediane ne les voit pas, elle mesure le
         bruit et les franges superposees de toutes les bacteries, qui montent avec N ;
       - pour chaque bacterie, le signal: moyenne (et max) du focus sur les 3 x 3 x 3
         voxels centres sur sa position vraie ;
       - contraste = signal moyen / fond diffus.
     On note aussi le contraste en unites de dispersion robuste du fond,
     (signal - mediane) / (1,4826 x MAD), l'echelle du K du seuil local MAD.

Les bacteries dont le cube 3 x 3 x 3 sort du volume sont ignorees.

Sorties, dans le repertoire de ce script:
    contraste_bacteries.csv           une ligne par (hologramme, bacterie, focus)
    fond_hologrammes.csv              une ligne par (hologramme, focus)
    contraste_vs_densite.png          un panneau par focus, echelle propre a chacun
    contraste_normalise_vs_densite.png  les cinq focus rapportes a leur valeur a N = 10

Usage:
    python report_focus_dense_holo/etude_contraste_dense.py
    python report_focus_dense_holo/etude_contraste_dense.py --sans-mesure
        (refait les graphes a partir des CSV, sans GPU)

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

# Les reductions CuPy (mediane) passent sinon par CUB, que nvcc doit compiler, et nvcc ne
# compile pas sur la machine de developpement (voir CLAUDE.md). A fixer avant l'import.
os.environ["CUPY_ACCELERATORS"] = ""

import cupy as cp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import focus
import propagation as propag
from focus import Focus_type

import benchmark_detection as banc          # lecture des simulations, optique du banc

# --- parametres -------------------------------------------------------------------------
SIMU = os.path.join(RACINE, "SIMULATED HOLO")

# optique et volume: ceux du banc d'essai (240 plans de 0,5 um depuis le capteur)
NB_PIX = banc.NB_PIX_X
NB_PLANE, DZ, DISTANCE_INI = banc.NB_PLANE, banc.DZ, banc.DISTANCE_INI
DX = banc.DX
FILTRE = (15, 0)                # passe-haut seul (0 = pas de passe-bas)
SUM_SIZE = banc.SUM_SIZE        # 15

FOCUS = (Focus_type.SUM_OF_VARIANCE, Focus_type.TENEGRAD, Focus_type.SUM_OF_LAPLACIAN,
         Focus_type.SUM_OF_GRADIENT, Focus_type.SUM_OF_INTENSITY)

DEMI_CUBE = 1                   # cube de 2 x 1 + 1 = 3 voxels de cote
PAS_FOND_Z, PAS_FOND_XY = 2, 4  # sous-echantillonnage du volume pour la mediane du fond

COULEURS = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac")
MARQUEURS = ("o", "s", "^", "D", "v")


def cubes(vrai):
    """Indices (z, y, x) des voxels centraux et masque des bacteries dont le cube tient."""
    ix = np.rint(vrai[:, 0] / DX).astype(int)
    iy = np.rint(vrai[:, 1] / DX).astype(int)
    iz = np.rint((vrai[:, 2] - DISTANCE_INI) / DZ).astype(int)
    d = DEMI_CUBE
    dedans = ((ix >= d) & (ix < NB_PIX - d) & (iy >= d) & (iy < NB_PIX - d)
              & (iz >= d) & (iz < NB_PLANE - d))
    return iz, iy, ix, dedans


def valeurs_cubes(d_focus, iz, iy, ix):
    """Les 27 valeurs du focus de chaque cube, (n_bacteries, 27), en une seule lecture GPU."""
    d = DEMI_CUBE
    o = np.arange(-d, d + 1)
    oz, oy, ox = [a.ravel() for a in np.meshgrid(o, o, o, indexing="ij")]
    z = cp.asarray(iz[:, None] + oz[None, :])
    y = cp.asarray(iy[:, None] + oy[None, :])
    x = cp.asarray(ix[:, None] + ox[None, :])
    return cp.asnumpy(d_focus[z, y, x])


def mesure():
    bufs = [cp.zeros((NB_PIX, NB_PIX), dtype=t) for t in
            (cp.complex64,) * 3 + (cp.float32,) + (cp.complex64,) * 2]
    d_module = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.float32)
    d_focus = cp.zeros((NB_PLANE, NB_PIX, NB_PIX), dtype=cp.float32)

    print("volume {} plans de {:.1f} um depuis {:.1f} um, filtre {}, sum_size {}".format(
        NB_PLANE, DZ * 1e6, DISTANCE_INI * 1e6, FILTRE, SUM_SIZE))
    lignes, fonds = [], []
    for dossier in banc.dossiers_simulation(SIMU):
        for n in banc.numeros_hologrammes(dossier):
            h_net, vrai, phi = banc.lire_simulation(dossier, n)
            densite = len(vrai)
            iz, iy, ix, dedans = cubes(vrai)
            propag.volume_propag_angular_spectrum_to_module(
                cp.asarray(h_net), *bufs, d_module, banc.WAVELENGTH / banc.INDEX_MEDIUM,
                banc.MAGNIFICATION, banc.PIX_SIZE, NB_PIX, NB_PIX, DISTANCE_INI, DZ,
                NB_PLANE, *FILTRE)

            for type_focus in FOCUS:
                focus.focus(d_module, d_focus, SUM_SIZE, type_focus)

                echantillon = d_focus[::PAS_FOND_Z, ::PAS_FOND_XY, ::PAS_FOND_XY].ravel()
                fond = float(cp.median(echantillon))
                mad = float(cp.median(cp.abs(echantillon - fond)))
                fonds.append(dict(densite=densite, hologramme=n, focus=type_focus.name,
                                  fond_median=fond, fond_moyen=float(cp.mean(d_focus)),
                                  mad=mad))
                del echantillon

                v = valeurs_cubes(d_focus, iz[dedans], iy[dedans], ix[dedans])
                signal = v.mean(axis=1)
                for k, i_bact in enumerate(np.flatnonzero(dedans)):
                    lignes.append(dict(
                        densite=densite, hologramme=n, bacterie=int(i_bact),
                        phi=float(phi[i_bact]), z=float(vrai[i_bact, 2]),
                        focus=type_focus.name, signal_moyen=float(signal[k]),
                        signal_max=float(v[k].max()), fond_median=fond,
                        contraste=float(signal[k]) / fond,
                        contraste_mad=(float(signal[k]) - fond) / (1.4826 * mad)))
            cp.get_default_memory_pool().free_all_blocks()
        print("   N = {} fait".format(densite))
    return pd.DataFrame(lignes), pd.DataFrame(fonds)


def graphes(bact):
    noms = [f.name for f in FOCUS]
    densites = sorted(bact.densite.unique())
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9c9c9c", "font.size": 10})

    # mediane et quartiles du contraste des bacteries, par densite
    stats = bact.groupby(["focus", "densite"]).contraste.quantile([0.25, 0.5, 0.75])
    stats = stats.unstack().reset_index()

    # un panneau par focus: les echelles different trop pour un axe commun lineaire
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, nom in enumerate(noms):
        ax = axes.flat[i]
        s = stats[stats.focus == nom].sort_values("densite")
        ax.fill_between(s.densite, s[0.25], s[0.75], color=COULEURS[i], alpha=0.2,
                        linewidth=0, label="quartiles 25-75 %")
        ax.plot(s.densite, s[0.5], color=COULEURS[i], marker=MARQUEURS[i], markersize=6,
                linewidth=1.5, label="mediane")
        ax.axhline(1, color="#9c9c9c", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(densites)
        ax.set_xticklabels([str(d) for d in densites], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(nom, fontsize=10)
        ax.set_xlabel("bacteries par hologramme")
        if i % 3 == 0:
            ax.set_ylabel("focus du cube 3x3x3 / fond diffus")
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.5,
                       "contraste = moyenne du focus sur les\n3 x 3 x 3 voxels centres sur la"
                       " bacterie,\ndivisee par la mediane du volume de\nfocus (fond"
                       " diffus).\n\nLigne grise: contraste = 1 (bacterie\nindiscernable"
                       " du fond).", fontsize=9, va="center", color="#555555")
    fig.suptitle("Contraste des bacteries par rapport au fond diffus, selon la densite")
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "contraste_vs_densite.png"), dpi=150)
    plt.close(fig)

    # les cinq focus sur un axe commun, rapportes a leur mediane a la plus faible densite
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, nom in enumerate(noms):
        s = stats[stats.focus == nom].sort_values("densite")
        ax.plot(s.densite, s[0.5] / s[0.5].iloc[0], label=nom, color=COULEURS[i],
                marker=MARQUEURS[i], markersize=6, linewidth=1.5)
    ax.axhline(1, color="#9c9c9c", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(densites)
    ax.set_xticklabels([str(d) for d in densites])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("bacteries par hologramme")
    ax.set_ylabel("contraste median / sa valeur a N = {}".format(densites[0]))
    ax.set_title("Contraste normalise: perte relative avec la densite")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ICI, "contraste_normalise_vs_densite.png"), dpi=150)
    plt.close(fig)


def main(sans_mesure):
    if sans_mesure:
        bact = pd.read_csv(os.path.join(ICI, "contraste_bacteries.csv"))
        fonds = pd.read_csv(os.path.join(ICI, "fond_hologrammes.csv"))
    else:
        bact, fonds = mesure()
        bact.to_csv(os.path.join(ICI, "contraste_bacteries.csv"), index=False)
        fonds.to_csv(os.path.join(ICI, "fond_hologrammes.csv"), index=False)

    med = bact.groupby(["focus", "densite"]).contraste.median().unstack()
    print("\nContraste median (focus du cube 3x3x3 / fond diffus)")
    print(med.loc[[f.name for f in FOCUS]].round(1).to_string())
    med_mad = bact.groupby(["focus", "densite"]).contraste_mad.median().unstack()
    print("\nContraste median en unites de MAD du fond, (signal - fond) / (1,4826 MAD)")
    print(med_mad.loc[[f.name for f in FOCUS]].round(1).to_string())

    graphes(bact)
    print("\nresultats dans {}".format(ICI))
    return 0


if __name__ == "__main__":
    sys.exit(main("--sans-mesure" in sys.argv))
