# -*- coding: utf-8 -*-

"""
Filename: CCL3D.py

Description:
Connected component labelling and label analysis using cupyx.scipy
Author: Simon BECKER
Date: 2025-07-11

License:
GNU General Public License v3.0

Copyright (C) [2024] Simon BECKER

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
# from typeHolo import *
import numpy as np
from cupyx import jit
from cupyx.scipy import ndimage as cp_ndimage
import numpy as np
from traitement_holo import *
from enum import Enum

class type_threshold(Enum):
    NB_STD_DEV = 1
    THRESHOLD = 2
    NB_LOCAL_MEAN = 3

dobjet = np.dtype([
    ('i_image', np.uint64),
    ('nb_pix', np.uint32),
    ('baryX', np.float32),
    ('baryY', np.float32),
    ('baryZ', np.float32),
    ('pSum', np.float32),
    ('pxSumX', np.float32),
    ('pxSumY', np.float32),
    ('pxSumZ', np.float32),
    ('xMin', np.uint32),
    ('xMax', np.uint32),
    ('yMin', np.uint32),
    ('yMax', np.uint32),
    ('zMin', np.uint32),
    ('zMax', np.uint32),
])

cp_dobjet = cp.dtype([
    ('i_image', np.uint64),
    ('nb_pix', np.uint32),
    ('baryX', np.float32),
    ('baryY', np.float32),
    ('baryZ', np.float32),
    ('pSum', np.float32),
    ('pxSumX', np.float32),
    ('pxSumY', np.float32),
    ('pxSumZ', np.float32),
    ('xMin', np.uint32),
    ('xMax', np.uint32),
    ('yMin', np.uint32),
    ('yMax', np.uint32),
    ('zMin', np.uint32),
    ('zMax', np.uint32),
])

#définition des matrices de connectivité du connected_component_labelling
connectivity_6 = cp_ndimage.generate_binary_structure(3, 1)
connectivity_18 = cp_ndimage.generate_binary_structure(3, 2)
connectivity_26 = cp_ndimage.generate_binary_structure(3, 3)

### Calcule du seuil global sur tout un volume
def calc_threshold(d_focus_volume, nbStdVar):
    mean = cp.mean(d_focus_volume)
    stdVar = cp.std(d_focus_volume)
    threshold = mean.item() + nbStdVar * stdVar.item()
    return(threshold)

#binarise un volume de taille sizeX, sizeY, sizeZ
@jit.rawkernel()
def cuda_Binaries_Focus_Volume(d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY and kk < sizeZ):

        if d_focus_volume[kk, jj, ii] > threshold:
            d_bin_volume[kk, jj, ii] = True
        else:
            d_bin_volume[kk, jj, ii] = False

    jit.syncthreads()

#binarise un plan de taille sizeX, sizeY
@jit.rawkernel()
def cuda_Binaries_Focus_Plan(d_bin_plan, d_focus_plan, threshold, sizeX, sizeY):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    jj = index // sizeX
    ii = index - jj * sizeX

    if (ii < sizeX and jj < sizeY):

        if d_focus_plan[jj, ii] > threshold:
            d_bin_plan[jj, ii] = True
        else:
            d_bin_plan[jj, ii] = False

    jit.syncthreads()

def binaries_Focus_Volume_local_contrast(d_bin_volume, d_focus_volume, threshold_nb_mean, sizeMeanXY, sizeMeanZ):

    sizeX = d_focus_volume.shape[2]
    sizeY = d_focus_volume.shape[1]
    sizeZ = d_focus_volume.shape[0]

    #si sumSize est pair rajouter 1
    sizeMeanXY = (sizeMeanXY //2 ) * 2 + 1
    sizeMeanZ = (sizeMeanZ //2 ) * 2 + 1

    #allocation volume de convolution (cube dont la somme est 1 de taille sizeSumX, sizeSumY, sizeSumZ)
    value = 1.0 / ( sizeMeanXY * sizeMeanXY * sizeMeanZ )
    convolve_volume = cp.full(fill_value=value, dtype=cp.float32, shape=(sizeMeanZ, sizeMeanXY, sizeMeanXY))

    #convolution du plan avec convolve_volume
    d_threashold_volume = cp_ndimage.convolve(input = d_focus_volume, weights = convolve_volume, mode = 'reflect') * threshold_nb_mean
    cp.greater(d_focus_volume, d_threashold_volume, out = d_bin_volume)

def div_by_mean_convolution(d_focus_volume, d_focus_volume_2, sizeMeanXY, sizeMeanZ):

    sizeX = d_focus_volume.shape[2]
    sizeY = d_focus_volume.shape[1]
    sizeZ = d_focus_volume.shape[0]

    #si sumSize est pair rajouter 1
    sizeMeanXY = (sizeMeanXY //2 ) * 2 + 1
    sizeMeanZ = (sizeMeanZ //2 ) * 2 + 1

    #allocation volume de convolution (cube dont la somme est 1 de taille sizeSumX, sizeSumY, sizeSumZ)
    value = 1.0 / ( sizeMeanXY * sizeMeanXY * sizeMeanZ )
    convolve_volume = cp.full(fill_value=value, dtype=cp.float32, shape=(sizeMeanZ, sizeMeanXY, sizeMeanXY))

    d_focus_volume_2 = cp.copy(d_focus_volume)

    #convolution du plan avec convolve_volume
    cp_ndimage.convolve(input = d_focus_volume_2, output = d_focus_volume, weights = convolve_volume, mode = 'reflect')

def CCL3D(d_bin_volume, d_focus_volume,  t_threshold, threshold, n_connectivity = 6):

    if n_connectivity == 26 :
        pattern_connectivity = connectivity_26
    elif n_connectivity == 18:
        pattern_connectivity = connectivity_18
    else :
        pattern_connectivity = connectivity_6

    sizeZ, sizeY, sizeX = d_focus_volume.shape
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ)//1024 +1
    #calcul du seuil

    if t_threshold == type_threshold.NB_STD_DEV :
        threshold = calc_threshold(d_focus_volume, threshold)
        cuda_Binaries_Focus_Volume[n_blocks, n_threads](d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ)
    elif t_threshold == type_threshold.THRESHOLD :
        cuda_Binaries_Focus_Volume[n_blocks, n_threads](d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ)
    else:
        binaries_Focus_Volume_local_contrast(d_bin_volume, d_focus_volume, threshold, 20, 10)

    # NOTE: un np.sum(cp.asnumpy(d_bin_volume)) figurait ici. Il rapatriait tout le volume
    # binaire sur le CPU (~200 Mo pour 200 plans en 1024x1024) a chaque hologramme, pour
    # une valeur qui n'etait utilisee que par deux print commentes. Si le comptage est
    # necessaire un jour, noter que cp.count_nonzero passe par CUB et echoue sur une
    # installation ou nvcc ne peut pas compiler (cf. la note dans projection_bool).
    labels, num_label = cp_ndimage.label(d_bin_volume, pattern_connectivity)

    return(labels, num_label)

@jit.rawkernel()
def device_CCA(d_volume_label, d_volume_focus, d_features, sizeX, sizeY, sizeZ, dx, dy, dz, z_offset):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY and kk < sizeZ):
        label = d_volume_label[kk, jj, ii]
        if (label != 0):
            #calcul des index, coordonnées et valeurs à ajouter aux features
            index = label-1
            posX = ii * dx
            posY = jj * dy
            # z_offset = distance de propagation du premier plan (distance_ini). Sans lui,
            # la coordonnee Z exportee est relative au premier plan reconstruit alors que
            # la colonne du CSV s'intitule 'Z POSITION (m)'.
            posZ = z_offset + kk * dz
            focus = d_volume_focus[kk, jj, ii]
            pxSumX = posX * focus
            pxSumY = posY * focus
            pxSumZ = posZ * focus
            """
            #index des données de type features
            # (nb_pix: 0,
            # pxSumX:1, pxSumY:2, pxSumZ:3, pSum:4
            #   )
            """
            #les valeurs actu
            #excuecution des fonctions atomics
            #nb_pix
            jit.atomic_add(d_features, (index, 0), 1)
            #pxSumX, pxSumY, pxSumZ, pSum
            jit.atomic_add(d_features, (index, 1), pxSumX)
            jit.atomic_add(d_features, (index, 2), pxSumY)
            jit.atomic_add(d_features, (index, 3), pxSumZ)
            jit.atomic_add(d_features, (index, 4), focus)
            
    jit.syncthreads()

@jit.rawkernel()
def device_CCA_plane(d_plane_label, d_plane_focus, d_features, sizeX, sizeY, dx, dy, dz, planeNumber, z_offset):
    #version optimisé de device_CCA pour ne travailler que sur un plan et économiser de la méméoire GPU
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    jj = index // sizeX
    ii = index - jj * sizeX

    if (ii < sizeX and jj < sizeY):
        label = d_plane_label[jj,ii]
        if (label != 0):
            #calcul des index, coordonnées et valeurs à ajouter aux features
            index = label-1
            posX = ii * dx
            posY = jj * dy
            posZ = z_offset + planeNumber * dz
            focus = d_plane_focus[jj,ii]
            pxSumX = posX * focus
            pxSumY = posY * focus
            pxSumZ = posZ * focus
            """
            #index des données de type features
            # (nb_pix: 0,
            # pxSumX:1, pxSumY:2, pxSumZ:3, pSum:4
            #   )
            """
            #les valeurs actu
            #excuecution des fonctions atomics
            #nb_pix
            jit.atomic_add(d_features, (index, 0), 1)
            #pxSumX, pxSumY, pxSumZ, pSum
            jit.atomic_add(d_features, (index, 1), pxSumX)
            jit.atomic_add(d_features, (index, 2), pxSumY)
            jit.atomic_add(d_features, (index, 3), pxSumZ)
            jit.atomic_add(d_features, (index, 4), focus)
            
    jit.syncthreads()

def CCA_CUDA(h_volume_label, d_volume_focus, number_of_labels, i_image, sizeX, sizeY, sizeZ, dx, dy, dz, z_offset = 0.0):

    # colonnes: 0=nb_pix, 1=pxSumX, 2=pxSumY, 3=pxSumZ, 4=pSum
    # cp.zeros remplace une boucle Python qui ecrivait les 5 colonnes label par label,
    # soit 5 x number_of_labels lancements de noyau pour une simple initialisation.
    d_features = cp.zeros(shape = (number_of_labels, 5), dtype = cp.float32)

    sizeX, sizeY, sizeZ = h_volume_label.shape
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ)//1024 +1
    d_volume_label = cp.asarray(h_volume_label)

    device_CCA[n_blocks, n_threads](d_volume_label, d_volume_focus, d_features, sizeX, sizeY, sizeZ, dx, dy, dz, z_offset)
    
    h_features = cp.asnumpy(d_features)

    features = np.ndarray(shape = (number_of_labels,), dtype = dobjet)

    #copie h_features vers features
    for i in range(number_of_labels):
        features[i]['i_image'] = i_image
        features[i]['nb_pix'] = h_features[i, 0]
        features[i]['pxSumX'] = h_features[i, 1]
        features[i]['pxSumY'] = h_features[i, 2]
        features[i]['pxSumZ'] = h_features[i, 3]
        features[i]['pSum'] =   h_features[i, 4]

    #calcul des barycentres

    features['baryX'] = features['pxSumX'] / features['pSum']
    features['baryY'] = features['pxSumY'] / features['pSum']
    features['baryZ'] = features['pxSumZ'] / features['pSum']
    features['i_image'] = i_image

    return features

def CCA_CUDA_float(volume_label, d_volume_focus, number_of_labels, i_image, sizeX, sizeY, sizeZ, dx, dy, dz, z_offset = 0.0):
    """Barycentres ponderes par le focus, en metres.

    z_offset : distance de propagation du premier plan (distance_ini), ajoutee a la
    coordonnee Z pour que celle-ci soit absolue et non relative au premier plan.
    """

    # colonnes: 0=nb_pix, 1=pxSumX, 2=pxSumY, 3=pxSumZ, 4=pSum
    # cp.zeros remplace une boucle Python qui ecrivait les 5 colonnes label par label,
    # soit 5 x number_of_labels lancements de noyau pour une simple initialisation.
    d_CCA = cp.zeros(shape = (number_of_labels, 5), dtype = cp.float32)

    sizeZ, sizeY, sizeX = volume_label.shape
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ)//1024 +1

    if isinstance(volume_label, np.ndarray):
        v_label = cp.asarray(volume_label)
        device_CCA[n_blocks, n_threads](v_label, d_volume_focus, d_CCA, sizeX, sizeY, sizeZ, dx, dy, dz, z_offset)
    else:
        device_CCA[n_blocks, n_threads](volume_label, d_volume_focus, d_CCA, sizeX, sizeY, sizeZ, dx, dy, dz, z_offset)

    h_CCA = cp.asnumpy(d_CCA)

    features = np.ndarray(shape = (number_of_labels, 5), dtype = np.float32)

    #calcul barycentres et copie données
    for i in range(number_of_labels):
        features[i, 0] = i_image
        features[i, 1] = h_CCA[i, 1] / h_CCA[i, 4] #calcul baryX = pxSumX / pSum
        features[i, 2] = h_CCA[i, 2] / h_CCA[i, 4] #calcul baryY = pxSumY / pSum
        features[i, 3] = h_CCA[i, 3] / h_CCA[i, 4] #calcul baryZ = pxSumZ / pSum
        features[i, 4] = h_CCA[i, 0] #nb_pix

    return features

def CCL_filter(features_in, nb_vox_min, nb_vox_max):
    """Garde les objets dont le nombre de voxels est dans [nb_vox_min, nb_vox_max].
    Une borne a 0 signifie 'pas de limite de ce cote'.

    Masque booleen: l'implementation precedente reconstruisait le tableau par np.append
    a chaque objet retenu, soit un cout quadratique en nombre d'objets.
    """
    if features_in.shape[0] == 0:
        return features_in

    nb_vox = features_in[:, 4]
    keep = np.ones(features_in.shape[0], dtype=bool)
    if nb_vox_min != 0:
        keep &= nb_vox >= nb_vox_min
    if nb_vox_max != 0:
        keep &= nb_vox <= nb_vox_max

    return features_in[keep]

