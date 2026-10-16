# -*- coding: utf-8 -*-

"""
Filename: script_localisation_pipeline_local_CCL.py

Description:
Troisieme variante de la detection, entre les deux autres scripts: le volume de focus est
binarise au-dessus d'un seuil LOCAL, puis etiquete en composantes connexes, et chaque
composante donne un barycentre.

    script_localisation_pipeline.py              seuil GLOBAL -> CCL3D -> barycentres
    ce script                                    seuil LOCAL  -> CCL3D -> barycentres
    script_localisation_pipeline_local_maxima.py seuil LOCAL  -> sommets -> centroides

Les trois partagent tout jusqu'au volume de focus, et ne different que par la derniere
etape. Ce script isole donc l'effet du SEUIL (global -> local), la ou le script des maxima
ajoute l'effet du MODE DE DETECTION (composantes -> sommets). Comparer les trois sur les
memes hologrammes dit lequel des deux changements apporte quoi.

Le seuil local vaut, maille par maille, fond + K x max(sigma, sigma plancher), interpole en
trilineaire (voir CCL3D.calc_threshold_local). Il suit le fond, donc il descend la ou
l'hologramme est propre et monte autour des objets brillants, ce qu'un seuil global ne sait
pas faire.

Ce que ce mode garde du pipeline actuel: la segmentation (volume binaire et etiquettes,
donc l'affichage), le nombre de voxels par objet, le filtre min/max voxel, et le barycentre
pondere par le focus. Ce qu'il ne corrige pas: une bacterie a peine au-dessus du seuil se
fragmente toujours en plusieurs composantes, et deux objets dont les regions se touchent
restent fusionnes.

Cout: le seuil local construit un volume de seuil de la taille du volume de focus
(800 Mo en 1024 x 1024 x 240) et coute environ 350 ms de plus que le seuil global.

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr
Date: 2026-10-15

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
from CCL3D import calc_threshold_local, CCL3D, CCA_CUDA_float, CCL_filter, type_threshold
import pandas as pd

# Parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
path = os.path.join(project_root, 'Film REF')
result_filename = os.path.join(path, 'result_local_CCL.csv')
type_image = 'bmp'

# hologram parameters. Toutes les distances sont en METRES.
medium_wavelenght = 660e-9 / 1.33
cam_magnification = 40.0
cam_nb_pix_X = 1024
cam_nb_pix_Y = 1024
# 240 plans de 0,5 um: le volume va de 0 a 119,5 um, comme dans les deux autres scripts.
nb_plane = 240
cam_pix_size = 7e-6
dx = cam_pix_size / cam_magnification
dy = cam_pix_size / cam_magnification
dz = 0.5e-6
distance_ini = 0.0

# focus parameters, memes defauts que le GUI (core.__init__)
type_focus = Focus_type.SUM_OF_VARIANCE
sum_size = 15

# Seuil local par mailles. K = 20 est la valeur mesuree en simulation pour ce mode: plus
# haute que celle des maxima (15), parce qu'une composante connexe se fragmente des que le
# seuil descend trop.
nb_StdVar_threshold = 20.0
maille_xy = 64              # 11,2 um avec un pixel de 0,175 um
maille_z = 25               # 12,5 um avec un pas de 0,5 um
kappa_ecretage = 3.0
nb_passes_ecretage = 3
percentile_plancher = 10
pas_echantillon = 1         # voir mailles_fond_sigma: 2 divise leur cout par sept

# CCL parameters
n_connectivity = 26
min_voxel = 0               # 0 = pas de filtre sur la taille des composantes
max_voxel = 0

# memory allocations
d_fft_holo = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_filtered = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_holo_filtered = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
# complex64: ce tampon recoit le champ propage, dont le volume empile le module (voir la
# note du script a seuil global, ou un float32 donnait |Re(E)| au lieu de |E|).
d_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_KERNEL = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_volume_module = cp.zeros(shape = (nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
d_bin_volume_focus = cp.zeros(shape = (nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.bool_)

# mean Holo removing, avec un cache sur disque comme dans les deux autres scripts
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

# Entete identique a celle du GUI: le CSV est relisible tel quel par HoloTracker Link.
colonnes = ['HOLOGRAM NUMBER', 'OBJECT NUMBER', 'X POSITION (m)', 'Y POSITION (m)',
            'Z POSITION (m)', 'NUMBER OF VOXEL']

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

        # seuil local: un seuil par voxel, interpole depuis les mailles
        d_seuil, infos = calc_threshold_local(
            d_volume_module, nb_StdVar_threshold, maille_xy, maille_z,
            kappa_ecretage, nb_passes_ecretage, percentile_plancher, pas_echantillon)
        t4 = time.perf_counter()

        # binarisation au-dessus du seuil local, puis composantes connexes
        d_labels, number_of_labels = CCL3D(d_bin_volume_focus, d_volume_module,
                                           type_threshold.THRESHOLD, d_seuil, n_connectivity)
        t5 = time.perf_counter()

        # Le volume de seuil pese autant que le volume de focus: le liberer des qu'il a
        # servi, avant que l'analyse des etiquettes n'alloue a son tour.
        del d_seuil
        cp.get_default_memory_pool().free_all_blocks()

        # barycentre de chaque composante, pondere par le focus
        features = CCA_CUDA_float(d_labels, d_volume_module, number_of_labels, i_image,
                                  cam_nb_pix_X, cam_nb_pix_Y, nb_plane,
                                  dx, dy, dz, z_offset=distance_ini)
        if min_voxel > 0 or max_voxel > 0:
            features = CCL_filter(features, min_voxel, max_voxel)
        t6 = time.perf_counter()
        del d_labels
        cp.get_default_memory_pool().free_all_blocks()

        positions = pd.DataFrame({
            'HOLOGRAM NUMBER': features[:, 0].astype(np.int64),
            'OBJECT NUMBER': np.arange(1, len(features) + 1),
            'X POSITION (m)': features[:, 1],
            'Y POSITION (m)': features[:, 2],
            'Z POSITION (m)': features[:, 3],
            'NUMBER OF VOXEL': features[:, 4].astype(np.int64)})
        positions.to_csv(result_filename, mode='a', index=False,
                         header=not os.path.exists(result_filename), columns=colonnes)

        # Les temps intermediaires sont asynchrones (le GPU travaille encore quand le
        # chronometre est lu): seul le total, qui se termine par un rapatriement sur le
        # CPU, est exact. Pour un chronometrage etape par etape, synchroniser.
        print("hologramme {:>3} : {:>4} objets, sigma plancher {:.4g}, {:.2f} s "
              "(propagation {:.2f} + focus {:.2f} + seuil local {:.2f} + CCL {:.2f} "
              "+ barycentres {:.2f})".format(
                  i_image, len(features), infos['sigma_plancher'], t6 - ini_time,
                  t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5))

print("\n{} hologrammes traites, resultats dans {}".format(i_image, result_filename))
