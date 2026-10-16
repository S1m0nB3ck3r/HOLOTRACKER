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


### Fond et sigma par mailles: la base du seuil local
def mailles_fond_sigma(d_focus_volume, nbStdVar, maille_xy=64, maille_z=25,
                       kappa=3.0, nb_passes=3, percentile_plancher=10, pas_echantillon=1):
    """Fond et sigma du bruit dans chaque maille, et le seuil local qui en decoule.

    Le volume est decoupe en mailles de maille_z plans sur maille_xy x maille_xy pixels. Dans
    chaque maille, fond et sigma sont estimes avec un ecretage iteratif a +/- kappa sigma, qui
    exclut les objets: sans lui, une bacterie brillante gonfle le sigma de sa propre maille et
    releve le seuil qui devrait la detecter.

    Le sigma plancher (percentile des sigma de mailles) evite qu'une maille tres calme donne un
    seuil ridiculement bas, ou le bruit passerait.

    pas_echantillon > 1 n'examine qu'un voxel sur pas_echantillon dans chaque direction XY. Le
    fond varie lentement et une maille contient des dizaines de milliers de voxels: a pas 2 il
    en reste un quart, soit 25 000, bien assez pour une moyenne et un ecart-type. Chaque passe
    d'ecretage relit la maille entiere, donc le cout suit directement ce pas.

    Retourne le dictionnaire d'infos: grilles de fond et de sigma, sigma plancher, seuil par
    maille et taille des mailles. **C'est ce dictionnaire, et non un volume de seuil, qu'il
    faut passer a maxima_centroides et mettre en cache en mode lot** (quelques milliers de
    valeurs au lieu de 800 Mo).
    """
    nz, ny, nx = d_focus_volume.shape
    gz, gy, gx = nz // maille_z, ny // maille_xy, nx // maille_xy
    if gz == 0 or gy == 0 or gx == 0:
        raise ValueError("volume {} plus petit qu'une maille ({}, {}, {})".format(
            d_focus_volume.shape, maille_z, maille_xy, maille_xy))

    fond = cp.zeros(shape=(gz, gy, gx), dtype=cp.float32)
    sigma = cp.zeros(shape=(gz, gy, gx), dtype=cp.float32)
    p = int(pas_echantillon)

    # Une tranche de mailles a la fois: regrouper tout le volume en (maille, voxels) demanderait
    # une copie de sa taille.
    for k in range(gz):
        tranche = d_focus_volume[k * maille_z:(k + 1) * maille_z, :gy * maille_xy, :gx * maille_xy]
        # (z, y, x) -> (maille_y, maille_x, voxels de la maille)
        tranche = tranche.reshape(maille_z, gy, maille_xy, gx, maille_xy)
        if p > 1:
            tranche = tranche[::p, :, ::p, :, ::p]
        tranche = tranche.transpose(1, 3, 0, 2, 4).reshape(gy, gx, -1)
        m = tranche.mean(axis=2)
        s = tranche.std(axis=2)
        for _ in range(nb_passes):
            # Masque en float32 et non en booleen: une somme de booleens passe par CUB, qui
            # exige nvcc (voir la note de CLAUDE.md et projection_bool).
            garde = (cp.abs(tranche - m[:, :, None]) <= kappa * s[:, :, None]).astype(cp.float32)
            n = cp.maximum(garde.sum(axis=2), 1.0)
            m = (tranche * garde).sum(axis=2) / n
            s = cp.sqrt((cp.square(tranche - m[:, :, None]) * garde).sum(axis=2) / n)
        fond[k] = m
        sigma[k] = s

    plancher = float(cp.percentile(sigma.ravel(), percentile_plancher))
    return {'fond': fond, 'sigma': sigma, 'sigma_plancher': plancher,
            'seuil_mailles': fond + nbStdVar * cp.maximum(sigma, plancher),
            'mailles': (maille_z, maille_xy, maille_xy)}


### Fond et dispersion par mailles, par statistiques de rang: mediane et MAD
def mailles_fond_mad(d_focus_volume, nbStdVar, maille_xy=64, maille_z=25,
                     pas_z=5, pas_xy=4):
    """Seuil local = mediane(maille) + nbStdVar x 1,4826 x MAD(maille).

    Le MAD est la mediane des ecarts a la mediane. Il faudrait que la moitie des voxels
    d'une maille changent pour le deplacer, donc les objets ne le touchent pas: la
    robustesse est obtenue par construction, la ou mailles_fond_sigma l'obtient par trois
    passes d'ecretage. Le facteur 1,4826 convertit le MAD en equivalent ecart-type pour un
    bruit gaussien (MAD = 0,6745 sigma), ce qui garde a nbStdVar le meme sens et le meme
    ordre de grandeur qu'avec l'ecretage.

    C'est la construction standard de la detection de sources: DETECT_THRESH de SExtractor
    en astronomie, detecteurs OS-CFAR en radar, mad_std d'astropy.

    Trois parametres au lieu de six: disparaissent kappa, le nombre de passes et le sigma
    plancher (le MAD d'un vrai bruit n'est jamais nul). Mesure a l'appui, sur hologrammes
    simules a verite terrain connue: F1 moyen de 0,939 contre 0,947 pour l'ecretage, soit
    huit milliemes, dans le bruit de mesure; et 28 ms au lieu de 245.

    pas_z et pas_xy sous-echantillonnent la maille avant les deux tris; ils doivent diviser
    maille_z et maille_xy. Les voxels voisins du volume de focus sont fortement correles (le
    critere moyenne deja sur 11 x 11 pixels), si bien qu'une maille ne contient qu'une
    centaine d'echantillons reellement independants: en lire 1 280 ou 102 400 ne change pas
    l'estimation. Mesure: seuil different de 2,4 %, detections identiques, 24 ms au lieu de
    1 800.

    Retourne le meme dictionnaire que mailles_fond_sigma, pour que la suite de la chaine ne
    voie pas la difference.
    """
    nz, ny, nx = d_focus_volume.shape
    gz, gy, gx = nz // maille_z, ny // maille_xy, nx // maille_xy
    if gz == 0 or gy == 0 or gx == 0:
        raise ValueError("volume {} plus petit qu'une maille ({}, {}, {})".format(
            d_focus_volume.shape, maille_z, maille_xy, maille_xy))
    if maille_z % pas_z or maille_xy % pas_xy:
        raise ValueError("pas_z ({}) doit diviser maille_z ({}) et pas_xy ({}) "
                         "maille_xy ({})".format(pas_z, maille_z, pas_xy, maille_xy))

    tranche = d_focus_volume[:gz * maille_z, :gy * maille_xy, :gx * maille_xy]
    tranche = tranche[::pas_z, ::pas_xy, ::pas_xy]
    tranche = tranche.reshape(gz, maille_z // pas_z, gy, maille_xy // pas_xy,
                              gx, maille_xy // pas_xy)
    # (maille_z, maille_y, maille_x, voxels de la maille), puis deux tris par maille
    tranche = tranche.transpose(0, 2, 4, 1, 3, 5).reshape(gz * gy * gx, -1)
    mediane = cp.percentile(tranche, 50, axis=1)
    mad = cp.percentile(cp.abs(tranche - mediane[:, None]), 50, axis=1)
    forme = (gz, gy, gx)
    mediane = mediane.reshape(forme).astype(cp.float32)
    sigma = (mad * cp.float32(1.4826)).reshape(forme).astype(cp.float32)

    return {'fond': mediane, 'sigma': sigma, 'sigma_plancher': 0.0,
            'seuil_mailles': mediane + cp.float32(nbStdVar) * sigma,
            'mailles': (maille_z, maille_xy, maille_xy)}


### Fond par mailles, estime par la mediane: variante simple du precedent
def mailles_fond_median(d_focus_volume, facteur, maille_xy=64, maille_z=25,
                        pas_z=5, pas_xy=4):
    """Fond median de chaque maille, et seuil local = fond x (1 + facteur).

    Variante de mailles_fond_sigma qui supprime l'ecart-type, l'ecretage et le sigma
    plancher. Trois mesures la justifient, toutes faites sur 'Film REF':

      - le seuil vaut 6,9 fois le fond local (p10 5,7, p90 9,6): une erreur de quelques
        pour cent sur le fond ne le deplace presque pas ;
      - l'ecart-type suit le fond (correlation 0,91), donc il n'apporte pas d'information
        independante: fond x (1 + facteur) remplace fond + K x sigma ;
      - une mediane resiste aux objets par construction. Comparee au fond ecrete en trois
        passes, elle s'en ecarte de -5 % en mediane et de +5 % au pire, la ou une moyenne
        brute derape a +336 % dans les mailles contenant un objet brillant.

    Le facteur n'a donc PAS la meme echelle que le nbStdVar de mailles_fond_sigma: il
    multiplie le fond au lieu de compter des ecarts-types. Un facteur de 6 correspond
    grossierement a K = 15.

    pas_z et pas_xy sous-echantillonnent la maille avant le tri; ils doivent diviser
    maille_z et maille_xy. Une maille de 25 x 64 x 64 voxels en garde ainsi 320, assez
    pour une mediane et sept fois moins cher a trier.

    Retourne le meme dictionnaire que mailles_fond_sigma, pour que la suite de la chaine
    ne voie pas la difference. 'sigma' y vaut l'ecart entre le fond et le seuil, ce qui
    garde un sens pour le rapport signal sur bruit des objets.
    """
    nz, ny, nx = d_focus_volume.shape
    gz, gy, gx = nz // maille_z, ny // maille_xy, nx // maille_xy
    if gz == 0 or gy == 0 or gx == 0:
        raise ValueError("volume {} plus petit qu'une maille ({}, {}, {})".format(
            d_focus_volume.shape, maille_z, maille_xy, maille_xy))
    if maille_z % pas_z or maille_xy % pas_xy:
        raise ValueError("pas_z ({}) doit diviser maille_z ({}) et pas_xy ({}) "
                         "maille_xy ({})".format(pas_z, maille_z, pas_xy, maille_xy))

    tranche = d_focus_volume[:gz * maille_z, :gy * maille_xy, :gx * maille_xy]
    tranche = tranche[::pas_z, ::pas_xy, ::pas_xy]
    tranche = tranche.reshape(gz, maille_z // pas_z, gy, maille_xy // pas_xy,
                              gx, maille_xy // pas_xy)
    # (maille_z, maille_y, maille_x, voxels de la maille) puis un tri par maille
    tranche = tranche.transpose(0, 2, 4, 1, 3, 5).reshape(gz * gy * gx, -1)
    fond = cp.percentile(tranche, 50, axis=1).reshape(gz, gy, gx).astype(cp.float32)

    seuil_mailles = fond * cp.float32(1.0 + facteur)
    return {'fond': fond, 'sigma': seuil_mailles - fond, 'sigma_plancher': 0.0,
            'seuil_mailles': seuil_mailles, 'mailles': (maille_z, maille_xy, maille_xy)}


### Seuil local etendu a tout le volume
def calc_threshold_local(d_focus_volume, nbStdVar, maille_xy=64, maille_z=25,
                         kappa=3.0, nb_passes=3, percentile_plancher=10, pas_echantillon=1,
                         infos=None):
    """Seuil local interpole a la taille du volume: fond + nbStdVar * max(sigma, plancher).

    Le seuil des mailles (voir mailles_fond_sigma) est interpole en trilineaire (grid_mode:
    les valeurs sont au centre des mailles), ce qui donne un seuil qui varie doucement.

    Retourne (d_seuil, infos). **Cette fonction ne sert qu'au seuillage voxel par voxel**,
    c'est-a-dire a une binarisation suivie de composantes connexes. La detection par maxima
    locaux n'en a pas besoin: elle interpole le seuil au vol dans son noyau, ce qui evite de
    construire ce volume (800 Mo et ~115 ms en 1024x1024x200).

    infos: grilles deja calculees, a reutiliser au lieu de refaire l'ecretage. Utile en mode
    lot (les grilles du premier hologramme servent aux suivants) et pour balayer nbStdVar,
    qui n'intervient que dans infos['seuil_mailles'].
    """
    if infos is None:
        infos = mailles_fond_sigma(d_focus_volume, nbStdVar, maille_xy, maille_z,
                                   kappa, nb_passes, percentile_plancher, pas_echantillon)
    nz, ny, nx = d_focus_volume.shape
    gz, gy, gx = nz // maille_z, ny // maille_xy, nx // maille_xy
    seuil_mailles = infos['seuil_mailles']

    d_seuil = cp.empty(shape=(nz, ny, nx), dtype=cp.float32)
    d_seuil[:gz * maille_z, :gy * maille_xy, :gx * maille_xy] = cp_ndimage.zoom(
        seuil_mailles, (maille_z, maille_xy, maille_xy), order=1, grid_mode=True, mode='nearest')
    # Si le volume n'est pas un multiple des mailles, le reste est aligne sur la derniere maille
    # complete, exactement comme l'interpolation traite ses propres bords.
    if gz * maille_z < nz:
        d_seuil[gz * maille_z:] = d_seuil[gz * maille_z - 1]
    if gy * maille_xy < ny:
        d_seuil[:, gy * maille_xy:, :] = d_seuil[:, gy * maille_xy - 1:gy * maille_xy, :]
    if gx * maille_xy < nx:
        d_seuil[:, :, gx * maille_xy:] = d_seuil[:, :, gx * maille_xy - 1:gx * maille_xy]

    return d_seuil, infos


### Taille de la boite des maxima, donnee en metres, rendue en voxels
def taille_boite_voxels(boite_xy, boite_z, dx, dz):
    """Boite (Z, Y, X) en voxels, de cote impair pour etre centree sur son voxel.

    boite_xy et boite_z sont en metres, comme tout le reste du projet: la boite suit la
    largeur de la reponse du critere de focus (mesuree ~2,1 um en XY et 5,3 um en Z), donc
    elle ne depend pas de l'echantillonnage choisi.
    """
    impair = lambda v: max(3, int(round(v)) // 2 * 2 + 1)
    return (impair(boite_z / dz), impair(boite_xy / dx), impair(boite_xy / dx))


@jit.rawkernel(device=True)
def device_seuil_maille(grille, gz, gy, gx, mz, my, mx, k, j, i):
    """Seuil interpole en trilineaire au voxel (k, j, i), depuis la grille des mailles.

    Reproduit cupyx.scipy.ndimage.zoom(..., order=1, grid_mode=True, mode='nearest'):
    la valeur d'une maille est au centre de celle-ci, et au-dela des centres des mailles
    du bord le seuil reste constant. Verifie contre zoom a 4e-7 pres.
    """
    fz = (cp.float32(k) + 0.5) / cp.float32(mz) - 0.5
    fy = (cp.float32(j) + 0.5) / cp.float32(my) - 0.5
    fx = (cp.float32(i) + 0.5) / cp.float32(mx) - 0.5
    fz = min(max(fz, cp.float32(0.0)), cp.float32(gz - 1))
    fy = min(max(fy, cp.float32(0.0)), cp.float32(gy - 1))
    fx = min(max(fx, cp.float32(0.0)), cp.float32(gx - 1))
    k0 = cp.int32(fz)
    j0 = cp.int32(fy)
    i0 = cp.int32(fx)
    k1 = min(k0 + 1, gz - 1)
    j1 = min(j0 + 1, gy - 1)
    i1 = min(i0 + 1, gx - 1)
    tz = fz - cp.float32(k0)
    ty = fy - cp.float32(j0)
    tx = fx - cp.float32(i0)
    return ((1 - tz) * ((1 - ty) * ((1 - tx) * grille[k0, j0, i0] + tx * grille[k0, j0, i1])
                        + ty * ((1 - tx) * grille[k0, j1, i0] + tx * grille[k0, j1, i1]))
            + tz * ((1 - ty) * ((1 - tx) * grille[k1, j0, i0] + tx * grille[k1, j0, i1])
                    + ty * ((1 - tx) * grille[k1, j1, i0] + tx * grille[k1, j1, i1])))


@jit.rawkernel()
def device_maxima_candidats(d_focus, grille, gz, gy, gx, mz, my, mx,
                            sizeZ, sizeY, sizeX, rz, ry, rx,
                            compteur, out_z, out_y, out_x, out_seuil, capacite):
    """Sommets du volume de focus: voxel au-dessus du seuil local ET sans voisin plus haut.

    Le seuil est interpole ici meme, maille par maille: aucun volume de seuil n'est construit.
    Le balayage de la boite n'a lieu que pour les voxels qui depassent le seuil, c'est-a-dire
    une infime fraction du volume. C'est ce qui remplace un filtre max sur tout le volume.
    """
    # cp.int32 explicite: l'indice d'un fil est NON SIGNE. Sans ce cast, i + ddi avec un
    # ddi negatif repasse par le haut (i = 0 moins 6 donne 4 milliards), et l'ecretage
    # min(max(...)) renvoie alors le bord OPPOSE du volume. Les sommets du bord etaient
    # compares aux voxels d'en face.
    index = cp.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    planSize = sizeX * sizeY
    k = index // planSize
    j = (index - k * planSize) // sizeX
    i = index - j * sizeX - k * planSize

    if (i < sizeX and j < sizeY and k < sizeZ):
        valeur = d_focus[k, j, i]
        seuil = device_seuil_maille(grille, gz, gy, gx, mz, my, mx, k, j, i)
        if valeur > seuil:
            voisin_plus_haut = False
            for ddk in range(-rz, rz + 1):
                kk = min(max(k + ddk, 0), sizeZ - 1)
                for ddj in range(-ry, ry + 1):
                    jj = min(max(j + ddj, 0), sizeY - 1)
                    for ddi in range(-rx, rx + 1):
                        ii = min(max(i + ddi, 0), sizeX - 1)
                        if d_focus[kk, jj, ii] > valeur:
                            voisin_plus_haut = True
            if not voisin_plus_haut:
                # atomic_add rend l'ancienne valeur: chaque sommet obtient sa place.
                pos = jit.atomic_add(compteur, 0, 1)
                if pos < capacite:
                    out_z[pos] = k
                    out_y[pos] = j
                    out_x[pos] = i
                    out_seuil[pos] = seuil


@jit.rawkernel()
def device_centroides_boite(d_focus, pos_z, pos_y, pos_x, seuil_sommet, n_objets,
                            sizeZ, sizeY, sizeX, rz, ry, rx, d_accumulateur):
    """Centroide de chaque boite, pondere par max(focus - seuil du sommet, 0).

    Un fil par (objet, voxel de la boite). Colonnes de l'accumulateur:
    0 = somme des poids, 1 = somme ponderee en Z, 2 = en Y, 3 = en X, 4 = voxels au-dessus
    du seuil.
    """
    # cp.int32 explicite, meme raison que dans device_maxima_candidats: les decalages ddk,
    # ddj, ddi sont negatifs sur la moitie de la boite.
    index = cp.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    taille_boite = (2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1)

    if index < n_objets * taille_boite:
        obj = index // taille_boite
        reste = index - obj * taille_boite
        nby = 2 * ry + 1
        nbx = 2 * rx + 1
        ddk = reste // (nby * nbx) - rz
        ddj = (reste - (ddk + rz) * nby * nbx) // nbx - ry
        ddi = reste - (ddk + rz) * nby * nbx - (ddj + ry) * nbx - rx

        k = min(max(pos_z[obj] + ddk, 0), sizeZ - 1)
        j = min(max(pos_y[obj] + ddj, 0), sizeY - 1)
        i = min(max(pos_x[obj] + ddi, 0), sizeX - 1)

        poids = d_focus[k, j, i] - seuil_sommet[obj]
        if poids > 0:
            jit.atomic_add(d_accumulateur, (obj, 0), poids)
            jit.atomic_add(d_accumulateur, (obj, 1), poids * cp.float32(k))
            jit.atomic_add(d_accumulateur, (obj, 2), poids * cp.float32(j))
            jit.atomic_add(d_accumulateur, (obj, 3), poids * cp.float32(i))
            jit.atomic_add(d_accumulateur, (obj, 4), cp.float32(1.0))


### Detection par maxima locaux, position par centroide dans la boite
def maxima_centroides(d_focus_volume, seuil, boite, dx, dy, dz, i_image=1, z_offset=0.0,
                      capacite=200000):
    """Un objet par sommet du volume de focus, au lieu d'une composante connexe par region.

    Un voxel est retenu s'il est egal au maximum de sa boite (il n'a aucun voisin plus haut)
    et s'il depasse le seuil. Le seuil ne dessine plus la forme des objets, il ne sert qu'a
    rejeter les sommets de bruit: c'est pourquoi le meme nbStdVar convient a toutes les
    densites, alors que celui du seuil global doit suivre le nombre d'objets.

    La position vient du centroide des voxels de la boite, ponderes par max(focus - seuil, 0).
    Mesure en simulation: meilleur que la parabole sur 3 points (deux fois moins d'erreur
    mediane en XY) et que le barycentre des composantes connexes, surtout en Z. Le seuil est
    pris au sommet pour toute la boite: il varie sur une maille (10 um) et la boite fait 2 um.

    seuil est soit le dictionnaire rendu par mailles_fond_sigma (seuil local), soit un simple
    flottant (seuil global). Aucun volume de seuil n'est construit dans les deux cas: le
    noyau interpole la grille des mailles au vol, et un flottant est traite comme une grille
    d'une seule maille.

    capacite borne le nombre de sommets retenus par hologramme. Au-dela, la fonction leve une
    erreur explicite plutot que de rendre un resultat tronque en silence.

    Retourne un tableau (n_objets, 5) float32 aux memes colonnes que CCA_CUDA_float
    (i_image, baryX, baryY, baryZ, nb_pix, en metres) et un dictionnaire avec, par objet, le
    pic du focus et son rapport signal sur bruit. Le RSB vaut (pic - fond local) / sigma local
    avec un seuil local, et pic / seuil avec un seuil global.

    Attention a nb_pix: c'est ici le nombre de voxels de la boite au-dessus du seuil, et il
    sature (92 % de la boite mesures sur un hologramme reel). Il ne mesure pas la taille de
    l'objet, contrairement au nb_pix des composantes connexes. Filtrer dessus n'a pas de sens;
    le pic et le RSB sont les descripteurs utiles de ce mode.
    """
    nz, ny, nx = d_focus_volume.shape
    rz, ry, rx = boite[0] // 2, boite[1] // 2, boite[2] // 2

    if isinstance(seuil, dict):
        infos = seuil
        grille = infos['seuil_mailles']
        mz, my, mx = infos['mailles']
    else:
        # Un seuil global est une grille d'une seule maille, couvrant tout le volume: le meme
        # noyau sert aux deux modes.
        infos = None
        grille = cp.full(shape=(1, 1, 1), fill_value=float(seuil), dtype=cp.float32)
        mz, my, mx = nz, ny, nx
    gz, gy, gx = grille.shape

    compteur = cp.zeros(shape=(1,), dtype=cp.int32)
    out_z = cp.zeros(shape=(capacite,), dtype=cp.int32)
    out_y = cp.zeros(shape=(capacite,), dtype=cp.int32)
    out_x = cp.zeros(shape=(capacite,), dtype=cp.int32)
    out_seuil = cp.zeros(shape=(capacite,), dtype=cp.float32)

    n_threads = 1024
    n_blocks = (nx * ny * nz + n_threads - 1) // n_threads
    device_maxima_candidats[n_blocks, n_threads](
        d_focus_volume, grille, cp.int32(gz), cp.int32(gy), cp.int32(gx),
        cp.int32(mz), cp.int32(my), cp.int32(mx),
        cp.int32(nz), cp.int32(ny), cp.int32(nx), cp.int32(rz), cp.int32(ry), cp.int32(rx),
        compteur, out_z, out_y, out_x, out_seuil, cp.int32(capacite))

    n_objets = int(compteur[0])
    if n_objets > capacite:
        raise MemoryError(
            "{} sommets trouves pour une capacite de {}. Le seuil est probablement trop bas; "
            "sinon relancer avec capacite={}.".format(n_objets, capacite, 2 * n_objets))
    if n_objets == 0:
        return (np.ndarray(shape=(0, 5), dtype=np.float32),
                {'pic': np.array([]), 'rsb': np.array([])})

    iz, iy, ix = out_z[:n_objets], out_y[:n_objets], out_x[:n_objets]
    seuil_sommet = out_seuil[:n_objets]

    # colonnes: 0 = somme des poids, 1 = somme ponderee en Z, 2 = en Y, 3 = en X, 4 = voxels
    d_accumulateur = cp.zeros(shape=(n_objets, 5), dtype=cp.float32)
    taille_boite = boite[0] * boite[1] * boite[2]
    n_blocks = (n_objets * taille_boite + n_threads - 1) // n_threads
    device_centroides_boite[n_blocks, n_threads](
        d_focus_volume, iz, iy, ix, seuil_sommet, cp.int32(n_objets),
        cp.int32(nz), cp.int32(ny), cp.int32(nx), cp.int32(rz), cp.int32(ry), cp.int32(rx),
        d_accumulateur)

    acc = cp.asnumpy(d_accumulateur)
    somme = np.maximum(acc[:, 0], 1e-20)
    pic = cp.asnumpy(d_focus_volume[iz, iy, ix])

    if infos is not None:
        # Maille contenant le sommet, puis RSB au sens du projet de simulation
        kz = cp.minimum(iz // mz, infos['fond'].shape[0] - 1)
        ky = cp.minimum(iy // my, infos['fond'].shape[1] - 1)
        kx = cp.minimum(ix // mx, infos['fond'].shape[2] - 1)
        fond = cp.asnumpy(infos['fond'][kz, ky, kx])
        sig = cp.asnumpy(cp.maximum(infos['sigma'][kz, ky, kx],
                                    cp.float32(infos['sigma_plancher'])))
        rsb = (pic - fond) / np.maximum(sig, 1e-20)
    else:
        rsb = pic / max(float(seuil), 1e-20)

    features = np.ndarray(shape=(n_objets, 5), dtype=np.float32)
    features[:, 0] = i_image
    features[:, 1] = acc[:, 3] / somme * dx
    features[:, 2] = acc[:, 2] / somme * dy
    features[:, 3] = z_offset + acc[:, 1] / somme * dz
    features[:, 4] = acc[:, 4]
    return features, {'pic': pic, 'rsb': rsb}

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
        if isinstance(threshold, cp.ndarray):
            # Seuil local: un seuil par voxel, rendu par calc_threshold_local. Le noyau
            # ci-dessous ne prend qu'un scalaire, et une comparaison vectorisee fait le
            # meme travail sans allouer de tampon supplementaire.
            cp.greater(d_focus_volume, threshold, out=d_bin_volume)
        else:
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

