# -*- coding: utf-8 -*-

"""
Filename: script_localisation_pipeline_local_maxima.py

Description:
Variante de script_localisation_pipeline.py qui detecte les objets par MAXIMA LOCAUX du
volume de focus, au lieu du seuil global + composantes connexes.

Les deux scripts partagent tout jusqu'au volume de focus. Ils ne different qu'a la derniere
etape:

    script_localisation_pipeline.py        seuil global -> CCL3D -> barycentre de chaque
                                           composante connexe
    ce script                              seuil local par mailles -> sommets du volume
                                           (un par objet) -> centroide dans une boite

Pourquoi: mesure du projet frere Simu-Bacteria-Holograms (2026-10-13, bacteries de
1,5-2 x 0,8 um, vérité terrain, K fixe comme sur un vrai film), rappel / precision /
bacteries couchees retrouvees:

    densite      seuil global + CCL     maxima + centroide
    10 objets    0,97 / 0,88 / 0,93     0,97 / 1,00 / 0,93
    50           0,93 / 0,95 / 0,88     0,97 / 0,98 / 0,96
    100          0,90 / 0,97 / 0,79     0,95 / 0,98 / 0,91
    200          0,81 / 0,99 / 0,67     0,91 / 0,98 / 0,84

Deux raisons a cet ecart. D'abord le meme nb_StdVar convient a toutes les densites: le seuil
ne dessine plus la forme des objets, il ne sert qu'a rejeter les sommets de bruit, alors que
le seuil global doit etre abaisse quand les objets se multiplient. Ensuite une bacterie juste
au-dessus du seuil ne se fragmente plus: elle laisse plusieurs ilots a une CCL, mais un seul
sommet dans sa boite.

Attention a la colonne NUMBER OF VOXEL: elle compte ici les voxels de la boite au-dessus du
seuil, et elle sature (environ 92 % de la boite). Elle ne mesure pas la taille de l'objet.
Les descripteurs utiles de ce mode sont le pic du focus et le RSB local, exportes en plus.

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr
Date: 2026-10-13

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

# Parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
path = os.path.join(project_root, 'Film REF')
result_filename = os.path.join(path, 'result_local_maxima.csv')
type_image = 'bmp'

# hologram parameters. Toutes les distances sont en METRES.
medium_wavelenght = 660e-9 / 1.33
cam_magnification = 40.0
cam_nb_pix_X = 1024
cam_nb_pix_Y = 1024
# 240 plans de 0,5 um: le volume va de 0 a 119,5 um, comme dans
# script_localisation_pipeline.py. Les deux scripts analysent ainsi le meme volume, et
# leurs resultats sont comparables.
nb_plane = 240
cam_pix_size = 7e-6
dx = cam_pix_size / cam_magnification
dy = cam_pix_size / cam_magnification
dz = 0.5e-6
distance_ini = 0.0

# focus parameters, memes defauts que le GUI (core.__init__)
type_focus = Focus_type.SUM_OF_VARIANCE
sum_size = 15

# Seuil local par mailles. Le K des maxima (15) ne depend pas de la densite, contrairement a
# celui du seuil global. Ces valeurs sont mesurees en simulation avec un filtre spectral
# 15-250; le filtre utilise ici est 15-125, ou le volume de focus est plus lisse: K reste a
# verifier sur le film.
nb_StdVar_threshold = 15.0
maille_xy = 64              # 11,2 um avec un pixel de 0,175 um
maille_z = 25               # 12,5 um avec un pas de 0,5 um
kappa_ecretage = 3.0
nb_passes_ecretage = 3
percentile_plancher = 10
# Un voxel sur pas_echantillon dans chaque direction XY pour estimer fond et sigma. A 1, les
# mailles coutent 210 ms par hologramme; a 2, 30 ms, pour le meme resultat a un objet
# marginal pres (mesure sur 'Film REF'). Le fond varie lentement et une maille contient
# 100 000 voxels: en echantillonner le quart suffit largement.
pas_echantillon = 1

# Boite des maxima, en metres: largeur de la reponse du critere de focus a une bacterie
# (mesuree 2,1 um en XY et 5,3 um en Z). Donnee en metres et non en voxels, pour qu'elle
# reste valable si le pixel ou le pas en Z changent.
boite_xy = 2.1e-6
boite_z = 5.3e-6

# memory allocations
d_fft_holo = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_filtered = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_holo_filtered = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
# complex64: ce tampon recoit le champ propage, dont le volume empile le module (voir la note
# du script a seuil global, ou un float32 donnait |Re(E)| au lieu de |E|).
d_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_KERNEL = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_volume_module = cp.zeros(shape = (nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
# Ni volume binaire ni volume d'etiquettes: cette methode n'etiquette rien. Elle n'alloue pas
# non plus de volume de seuil, qui pesait autant que le volume de focus: le seuil des mailles
# est interpole au vol dans le noyau des maxima.

boite = taille_boite_voxels(boite_xy, boite_z, dx, dz)
print("boite des maxima : {} plans x {} x {} pixels, soit {:.1f} x {:.2f} um".format(
    boite[0], boite[1], boite[2], boite[0] * dz * 1e6, boite[1] * dx * 1e6))

# mean Holo removing, avec un cache sur disque comme dans le script a seuil global
chemin_cache = os.path.join(path, 'mean', 'mean_holo.npy')
if os.path.exists(chemin_cache):
    h_mean_holo = np.load(chemin_cache)
else:
    h_mean_holo, _, _ = calc_holo_moyen(
        lister_images(path, type_image),
        type_moyenne='arithmetique',
        taille=(cam_nb_pix_X, cam_nb_pix_Y),
        fichiers_sortie=(chemin_cache,))

i_image = 0
if os.path.exists(result_filename):
    os.remove(result_filename)

# Entete identique a celle du GUI, plus les deux descripteurs propres aux maxima: le CSV est
# relisible tel quel par HoloTracker Link, qui selectionne ses colonnes par nom.
colonnes = ['HOLOGRAM NUMBER', 'OBJECT NUMBER', 'X POSITION (m)', 'Y POSITION (m)',
            'Z POSITION (m)', 'NUMBER OF VOXEL', 'FOCUS PEAK', 'LOCAL SNR']

for image in sorted(os.listdir(path)):
    if (image.split('.')[-1].lower() == type_image.lower()):

        ini_time = time.perf_counter()
        i_image += 1
        h_holo = read_image(os.path.join(path, image), cam_nb_pix_X, cam_nb_pix_Y)

        # Remove mean image
        h_holo = h_holo / h_mean_holo
        d_holo = cp.asarray(h_holo)

        # volume propagation by angular spectrum method
        t1 = time.perf_counter()
        propag.volume_propag_angular_spectrum_to_module(
            d_holo, d_fft_holo, d_fft_holo_filtered, d_KERNEL,
            d_holo_filtered, d_fft_holo_propag, d_holo_propag, d_volume_module,
            medium_wavelenght, cam_magnification, cam_pix_size, cam_nb_pix_X, cam_nb_pix_Y,
            distance_ini, dz, nb_plane, 15, 125)
        t2 = time.perf_counter()

        # focus on the volume (INPLACE)
        focus.focus(d_volume_module, d_volume_module, sum_size, type_focus)
        t3 = time.perf_counter()

        # fond et sigma du bruit par maille, d'ou le seuil local
        infos = mailles_fond_sigma(
            d_volume_module, nb_StdVar_threshold, maille_xy, maille_z,
            kappa_ecretage, nb_passes_ecretage, percentile_plancher, pas_echantillon)
        t4 = time.perf_counter()

        # maxima locaux, position par centroide dans la boite
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

        # Les temps ci-dessous sont lus APRES une synchronisation implicite: maxima_centroides
        # rapatrie ses resultats sur le CPU, ce qui attend la fin du GPU. Les etapes
        # intermediaires, elles, sont asynchrones: leurs durees se recouvrent, et seul le
        # total est exact. Pour un chronometrage etape par etape, synchroniser explicitement.
        print("hologramme {:>3} : {:>4} objets, sigma plancher {:.4g}, {:.2f} s "
              "(propagation {:.2f} + focus {:.2f} + mailles {:.2f} + maxima {:.2f})".format(
                  i_image, len(features), infos['sigma_plancher'], t5 - ini_time,
                  t2 - t1, t3 - t2, t4 - t3, t5 - t4))

print("\n{} hologrammes traites, resultats dans {}".format(i_image, result_filename))
