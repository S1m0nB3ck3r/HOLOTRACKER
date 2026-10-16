# -*- coding: utf-8 -*-

"""
Filename: script_localisation_pipeline_local_sigma_maxima.py

Description:
seuil LOCAL par mailles: fond + K x sigma, estimes par ecretage, et detection par MAXIMA LOCAUX, position par centroide dans une boite.

Une des six combinaisons seuil x detection, toutes dans ce repertoire:

                               barycentre CCL              maxima locaux
    seuil global               ..._global_CCL.py           ..._global_maxima.py
    seuil local sigma          ..._local_sigma_CCL.py      ..._local_sigma_maxima.py
    seuil local MAD            ..._local_MAD_CCL.py        ..._local_MAD_maxima.py

Les six partagent tout jusqu'au volume de focus et ne different que par deux blocs: le
calcul du SEUIL, puis la DETECTION des centres. Mesures des six (F1 moyen sur hologrammes
simules a verite terrain connue, detections par image et persistance des trajectoires sur
'Film REF', temps par hologramme sur une Quadro RTX 3000):

    seuil + detection        K      F1       det./img   persistance    ms
    global + CCL             15     0,882     72,2        0,948        859
    global + maxima          15     0,884     74,6        0,956        854
    local sigma + CCL        20     0,947     93,9        0,934       1141
    local sigma + maxima     20     0,948     98,6        0,946        951
    local MAD + CCL          20     0,939     91,7 (*)    0,928        924
    local MAD + maxima       20     0,941    111,2        0,926        734

(*) la ligne du film a ete mesuree a K = 25; l'optimum simule du MAD est plat de 20 a 25.
Detail et methode: RAPPORT_SEUILS_ET_DETECTION.pdf, mesures brutes dans mesures/.

Lecture d'ensemble: c'est le SEUIL qui porte le gain (0,882 global contre 0,94 local), la
detection ne separe les deux modes que de deux milliemes. Et le K du seuil global depend de
la densite (son optimum passe de 20 a cinq objets a 10 a deux cents), alors qu'un film est
traite avec un K unique; le K d'un seuil local ne bouge pas.

Le seuil: Le volume est decoupe en mailles de 25 plans sur 64 x 64 pixels (12,5 x 11,2 x 11,2 um).
Dans chaque maille, le fond et le sigma du bruit sont estimes par trois passes d'ecretage a
+/- 3 sigma, qui ecartent les objets: sans elles, une bacterie brillante gonfle le sigma de
sa propre maille et s'efface elle-meme. Un sigma plancher (10e percentile des mailles)
protege les mailles ou l'ecretage a trop mordu.

Le seuil suit donc le fond: il descend ou l'hologramme est propre, monte autour des objets
brillants. Son K ne depend plus de la densite.

C'est le plus precis des trois seuils (F1 0,947), mais le plus cher (245 ms contre 28 pour
le MAD) et le plus riche en parametres: maille XY, maille Z, kappa, nombre de passes,
percentile plancher, pas d'echantillonnage.

La detection: Un voxel est retenu s'il est egal au maximum de sa boite -- il n'a aucun voisin plus haut --
et s'il depasse le seuil. La boite fait la largeur de la reponse du critere de focus a une
bacterie (2,1 um en XY, 5,3 um en Z, mesurees). La position vient du centroide des voxels de
la boite, ponderes par max(focus - seuil, 0).

Ici le seuil ne dessine plus la forme des objets, il ne sert qu'a rejeter les sommets de
bruit. Consequence: une bacterie a peine au-dessus du seuil ne se fragmente plus, elle donne
un sommet et un seul. Aucun volume de seuil n'est construit, et il n'y a ni volume binaire
ni volume d'etiquettes: le noyau interpole la grille des mailles au vol, ce qui rend ce mode
le plus rapide des deux (734 ms contre 924 avec le seuil MAD).

Attention a la colonne NUMBER OF VOXEL: elle compte les voxels de la boite au-dessus du
seuil et elle sature (environ 92 % de la boite). Elle ne mesure pas la taille de l'objet, et
filtrer dessus n'a pas de sens. Les descripteurs utiles de ce mode sont le pic du focus et le
rapport signal sur bruit local, exportes en plus.

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr
Date: 2026-10-16

License:
GNU General Public License v3.0

Copyright (C) [2026] Simon BECKER

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import cupy as cp
import numpy as np
import time
import os
from traitement_holo import *
import propagation as propag
import focus
from focus import Focus_type
from CCL3D import mailles_fond_sigma, taille_boite_voxels, maxima_centroides
import pandas as pd

# --------------------------------------------------------------- ou lire, ou ecrire
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# Le film de reference contient deux sequences: 'bacteria' (202 hologrammes) et
# 'polystyrene_spheres' (101). Changer de sequence, c'est changer cette ligne.
path = os.path.join(project_root, 'Film REF', 'bacteria')
result_filename = os.path.join(path, 'result_local_sigma_maxima.csv')
type_image = 'bmp'

# --------------------------------------------------------------- hologramme et volume
# Toutes les distances sont en METRES, comme dans l'application graphique.
medium_wavelenght = 660e-9 / 1.33
cam_magnification = 40.0
cam_nb_pix_X = 1024
cam_nb_pix_Y = 1024
cam_pix_size = 7e-6
nb_plane = 240              # 240 plans de 0,5 um: le volume va de 0 a 119,5 um
dx = cam_pix_size / cam_magnification
dy = cam_pix_size / cam_magnification
dz = 0.5e-6
distance_ini = 0.0          # distance de propagation du premier plan, en metres
filtre_passe_haut = 15      # filtre spectral, en pixels de la transformee
filtre_passe_bas = 125

# --------------------------------------------------------------- critere de focus
# Memes defauts que le GUI (core.__init__). A K egal, SUM_OF_VARIANCE detecte environ deux
# fois plus d'objets que TENEGRAD: le critere et le K vont ensemble.
type_focus = Focus_type.SUM_OF_VARIANCE
sum_size = 15

# --------------------------------------------------------------- seuil
# Seuil local par mailles, par ecretage iteratif. K = 20 est l'optimum mesure en
# simulation (F1 0,947), et il ne bouge pas avec la densite.
nb_StdVar_threshold = 20.0
maille_xy = 64              # 11,2 um avec un pixel de 0,175 um
maille_z = 25               # 12,5 um avec un pas de 0,5 um
kappa_ecretage = 3.0        # les voxels au-dela de +/- 3 sigma sortent de l'estimation
nb_passes_ecretage = 3
percentile_plancher = 10    # sigma plancher, en percentile des sigma de toutes les mailles
# Un voxel sur pas_echantillon dans chaque direction XY. A 1 les mailles coutent 210 ms par
# hologramme, a 2 elles en coutent 30 pour le meme resultat a un objet marginal pres: le
# fond varie lentement et une maille contient 100 000 voxels.
pas_echantillon = 1

# --------------------------------------------------------------- detection
# Boite des maxima, en metres: largeur de la reponse du critere de focus a une bacterie
# (mesuree 2,1 um en XY et 5,3 um en Z). Donnee en metres et non en voxels, pour qu'elle
# reste valable si le pixel ou le pas en Z changent.
boite_xy = 2.1e-6
boite_z = 5.3e-6

# --------------------------------------------------------------- allocations GPU
d_fft_holo = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_filtered = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_holo_filtered = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
# complex64, comme dans le GUI (core.allocate): ce tampon recoit le champ propage, et le
# volume empile ensuite son module. En float32, l'affectation jetait la partie imaginaire
# sans erreur (un simple ComplexWarning) et le volume contenait |Re(E)| au lieu de |E|:
# 36 % d'ecart, avec un profil axial qui oscille a la phase de la porteuse.
d_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_KERNEL = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_volume_module = cp.zeros(shape = (nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
# Ni volume binaire ni volume d'etiquettes: cette methode n'etiquette rien. Elle n'alloue pas
# non plus de volume de seuil, qui pesait autant que le volume de focus.

boite = taille_boite_voxels(boite_xy, boite_z, dx, dz)
print("boite des maxima : {} plans x {} x {} pixels, soit {:.1f} um en Z et {:.2f} um en XY".format(
    boite[0], boite[1], boite[2], boite[0] * dz * 1e6, boite[1] * dx * 1e6))

# --------------------------------------------------------------- hologramme moyen
# Avec un cache sur disque: le recalculer a chaque lancement coute une lecture complete de
# la sequence. Le cache est explicite ici plutot que cache dans calc_holo_moyen, pour qu'on
# voie d'un coup d'oeil ce qui se passe.
chemin_cache = os.path.join(path, 'mean', 'mean_holo.npy')
if os.path.exists(chemin_cache):
    h_mean_holo = np.load(chemin_cache)
else:
    # taille=... recadre chaque image au centre a la taille du capteur utilisee ici.
    h_mean_holo, _, _ = calc_holo_moyen(
        lister_images(path, type_image),
        type_moyenne='arithmetique',
        taille=(cam_nb_pix_X, cam_nb_pix_Y),
        fichiers_sortie=(chemin_cache,))

i_image = 0
if os.path.exists(result_filename):
    os.remove(result_filename)

# Entete identique a celle du CSV du GUI, plus les deux descripteurs propres aux maxima: le fichier est relisible tel quel
# par HoloTracker Link, qui selectionne ses colonnes par nom.
colonnes = ['HOLOGRAM NUMBER', 'OBJECT NUMBER', 'X POSITION (m)', 'Y POSITION (m)',
            'Z POSITION (m)', 'NUMBER OF VOXEL', 'FOCUS PEAK', 'LOCAL SNR']

# --------------------------------------------------------------- un hologramme a la fois
for image in sorted(os.listdir(path)):
    if (image.split('.')[-1].lower() == type_image.lower()):

        ini_time = time.perf_counter()
        i_image += 1
        h_holo = read_image(os.path.join(path, image), cam_nb_pix_X, cam_nb_pix_Y)

        # nettoyage par l'hologramme moyen (division)
        h_holo = h_holo / h_mean_holo
        d_holo = cp.asarray(h_holo)

        # propagation du volume par spectre angulaire, puis module du champ
        t1 = time.perf_counter()
        propag.volume_propag_angular_spectrum_to_module(
            d_holo, d_fft_holo, d_fft_holo_filtered, d_KERNEL,
            d_holo_filtered, d_fft_holo_propag, d_holo_propag, d_volume_module,
            medium_wavelenght, cam_magnification, cam_pix_size, cam_nb_pix_X, cam_nb_pix_Y,
            distance_ini, dz, nb_plane, filtre_passe_haut, filtre_passe_bas)
        t2 = time.perf_counter()

        # critere de focus, sur place: le volume de modules devient volume de focus
        focus.focus(d_volume_module, d_volume_module, sum_size, type_focus)
        t3 = time.perf_counter()

        # fond et sigma du bruit par maille, d'ou le seuil local
        infos = mailles_fond_sigma(
            d_volume_module, nb_StdVar_threshold, maille_xy, maille_z,
            kappa_ecretage, nb_passes_ecretage, percentile_plancher, pas_echantillon)
        t4 = time.perf_counter()

        # maxima locaux, position par centroide dans la boite. Le seuil est passe tel quel:
        # un flottant est traite comme une grille d'une seule maille, un dictionnaire de
        # mailles est interpole au vol dans le noyau.
        features, descripteurs = maxima_centroides(
            d_volume_module, infos, boite, dx, dy, dz,
            i_image=i_image, z_offset=distance_ini)
        t5 = time.perf_counter()

        positions = pd.DataFrame({
            'HOLOGRAM NUMBER': features[:, 0].astype(np.int64),
            'OBJECT NUMBER': np.arange(1, len(features) + 1),
            'X POSITION (m)': features[:, 1],
            'Y POSITION (m)': features[:, 2],
            'Z POSITION (m)': features[:, 3],
            'NUMBER OF VOXEL': features[:, 4].astype(np.int64),
            'FOCUS PEAK': descripteurs['pic'],
            'LOCAL SNR': descripteurs['rsb']})
        positions.to_csv(result_filename, mode='a', index=False,
                         header=not os.path.exists(result_filename), columns=colonnes)

        # Les temps intermediaires sont asynchrones (le GPU travaille encore quand le
        # chronometre est lu): seul le total, qui se termine par un rapatriement sur le CPU,
        # est exact. Pour un chronometrage etape par etape, synchroniser explicitement.
        print("hologramme {:>3} : {:>4} objets, {:.2f} s "
              "(propagation {:.2f} + focus {:.2f} + mailles {:.2f} + maxima {:.2f})".format(
                  i_image, len(features), t5 - ini_time, t2 - t1, t3 - t2, t4 - t3, t5 - t4))

print("\n{} hologrammes traites, resultats dans {}".format(i_image, result_filename))
