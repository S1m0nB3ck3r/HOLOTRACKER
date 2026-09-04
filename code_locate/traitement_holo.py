# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import cupy as cp
import time
import math
from cupyx import jit
import cupy as cp
from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn

def read_image(path_image, sizeX = 0, sizeY = 0):
        
        h_holo = np.asarray(Image.open(path_image))

        if ((sizeX != 0) and (sizeY != 0)):

            sx = np.size(h_holo, axis = 1)
            sy = np.size(h_holo, axis = 0)

            offsetX = (sx - sizeX)//2
            offsetY = (sy - sizeY)//2

            h_holo = h_holo[offsetY:offsetY+sizeY:1, offsetX:offsetX+sizeX:1]
        
        h_holo = h_holo.astype(np.float32)
        return(h_holo)

def display(plan):

    if isinstance(plan, cp.ndarray):
        h_plan = cp.asnumpy(plan)
        min = h_plan.min()
        max = h_plan.max()
        img = Image.fromarray((h_plan - min) * 255 / (max - min))

    else:
        min = plan.min()
        max = plan.max()
        img = Image.fromarray((plan - min) * 255 / (max - min))
    
    img.show(title = "plan")
    img.close()

@cp.fuse()
def div_holo(A, B):
    if (B!=0.0):
        C = A/B
    else:
        C = 0.0
    return C

def module(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.sqrt(cp.square(cp.real(planComplex)) + cp.square(cp.imag(planComplex))))
    else:
        return(np.sqrt(np.square(np.real(planComplex)) + np.square(np.imag(planComplex))))

def intensite(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.square(cp.real(planComplex)) + cp.square(cp.imag(planComplex)))
    else:
        return(np.square(np.real(planComplex)) + np.square(np.imag(planComplex)))

def phase(planComplex):
    # arctan2 et non arctan: arctan(im/re) replie la phase sur ]-pi/2, pi/2[ et perd
    # le quadrant, en plus de diviser par zero sur les pixels de partie reelle nulle.
    if isinstance(planComplex, cp.ndarray):
        return(cp.arctan2(cp.imag(planComplex), cp.real(planComplex)))
    else:
        return(np.arctan2(np.imag(planComplex), np.real(planComplex)))

def get_sub_plane(x, y, z, boxSizeXY, boxSizeZ, d_volume):

    sizeX, sizeY, sizeZ = d_volume.shape
    planXY = np.zeros(shape=(boxSizeXY, boxSizeXY))
    planXZ = np.zeros(shape=(boxSizeXY, boxSizeZ))
    planYZ = np.zeros(shape=(boxSizeXY, boxSizeZ))

    #test des limites des coordonées xyz
    xMin = int(x - boxSizeXY//2)
    xMax = int(x + boxSizeXY//2)
    yMin = int(y - boxSizeXY//2)
    yMax = int(y + boxSizeXY//2)
    zMin = int(z - boxSizeZ//2)
    zMax = int(z + boxSizeZ//2)

    xMin = xMin if xMin > 0 else 0
    xMax = xMax if xMax < sizeX else sizeX 
    yMin = yMin if yMin > 0 else 0 
    yMax = yMax if yMax < sizeY else sizeY
    zMin = zMin if zMin > 0 else 0 
    zMax = zMax if zMax < sizeZ else sizeZ

    if isinstance(d_volume, cp.ndarray):
        if (d_volume.dtype == cp.complex64):
            planXY_t = cp.asnumpy(intensite(d_volume[xMin : xMax, yMin : yMax, z ]))
            planXY[0:boxSizeXY, 0:boxSizeXY] = planXY_t
            planXZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[xMin : xMax, y, zMin : zMax]))
            planYZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[x , yMin : yMax, zMin : zMax ]))
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = cp.asnumpy(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[x , yMin : yMax, zMin : zMax ])
    else:
        if (d_volume.dtype == np.complex64):
            planXY[0:boxSizeXY, 0:boxSizeXY]  = intensite(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[x , yMin : yMax, zMin : zMax ])
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = d_volume[xMin : xMax, yMin : yMax, z ]
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[xMin : xMax, y, zMin : zMax]
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[x , yMin : yMax, zMin : zMax ]

    min = planXY.min()
    max = planXY.max()
    planXY = (planXY - min) * 255 / (max - min)
            
    min = planXZ.min()
    max = planXZ.max()
    planXZ = (planXZ - min) * 255 / (max - min)

    min = planYZ.min()
    max = planYZ.max()
    planYZ = (planYZ - min) * 255 / (max - min)

    planYZ.reshape((boxSizeXY, boxSizeZ))

    return np.concatenate((planXY, planXZ, planYZ), axis = 1)

def lister_images(repertoire, extension):
    """Chemins tries des images d'une extension donnee ('tif', 'bmp', ...).

    Le tri est important: il fixe l'ordre des hologrammes, et donc la numerotation des
    images dans le CSV de sortie.
    """
    ext = '.' + extension.lower().lstrip('.')
    noms = sorted(f for f in os.listdir(repertoire) if f.lower().endswith(ext))
    return [os.path.join(repertoire, n) for n in noms]


def sauver_hologramme(tableau, chemin):
    """Ecrit un tableau 2D. Le format est deduit de l'extension du chemin:

      .tif  flottant 32 bits, sans perte: c'est ce qui sert aux calculs
      .npy  format numpy, sans perte
      autre apercu 8 bits NORMALISE sur la dynamique reelle du tableau

    L'apercu est normalise et non tronque: un simple clip(0, 255) donne une image noire
    des que les valeurs ne sont pas deja a l'echelle 0-255, ce qui est le cas des images
    flottantes issues du pre-traitement.
    """
    ext = os.path.splitext(chemin)[1].lower()
    os.makedirs(os.path.dirname(chemin) or '.', exist_ok=True)

    if ext == '.npy':
        np.save(chemin, tableau)
    elif ext in ('.tif', '.tiff'):
        Image.fromarray(tableau.astype(np.float32), mode='F').save(chemin)
    else:
        v_min, v_max = float(tableau.min()), float(tableau.max())
        if v_max > v_min:
            apercu = (tableau - v_min) * 255.0 / (v_max - v_min)
        else:
            apercu = np.zeros_like(tableau)
        Image.fromarray(apercu.astype(np.uint8), mode='L').save(chemin)
    return chemin


def calc_holo_moyen(chemins, type_moyenne="arithmetique", taille=None,
                    fichiers_sortie=(), progress_callback=None):
    """Hologramme moyen d'une pile d'images.

    C'est LA fonction de calcul de moyenne du projet. Elle remplace trois versions qui
    coexistaient (le bouton de Locate, le script hors GUI, l'outil de pre-traitement) et
    ne differaient que par des details peripheriques, devenus des arguments.

    Parameters
    ----------
    chemins : liste des chemins d'images, dans l'ordre. Voir lister_images().
    type_moyenne : "arithmetique"  -> moyenne classique, somme / n
                   "geometrique"   -> exp(moyenne(log(x))), soit la racine n-ieme du
                                      produit. Le passage par le logarithme n'est pas une
                                      autre moyenne: c'est sa forme numeriquement stable.
                                      Elle donne le meme poids relatif a chaque image et
                                      est moins sensible aux valeurs extremes.
                   ("arithmetic", "log" et "logarithmic" sont acceptes: ce sont les noms
                    employes par les anciennes versions.)
    taille : (sizeX, sizeY) pour recadrer chaque image au CENTRE avant de l'accumuler,
             ou None pour prendre les images entieres.
    fichiers_sortie : chemins ou ecrire la moyenne. Le format de chacun vient de son
             extension (voir sauver_hologramme). Exemple: (".../mean.tif", ".../mean.bmp")
             pour obtenir a la fois le fichier de calcul et son apercu.
    progress_callback : callable(i, n), appele apres chaque image.

    Returns
    -------
    (moyenne, carte_min, carte_max), en float32.

    Les cartes min/max donnent, pour chaque pixel, la plus petite et la plus grande valeur
    rencontree sur toute la pile. Elles sont toujours calculees: leur cout est negligeable
    devant la lecture des images, et elles permettent de determiner analytiquement les
    bornes de normalisation sans repasser sur les donnees (voir utils/pre_treatment.py).
    Un appelant qui n'en a pas besoin les ignore.
    """
    geometrique = str(type_moyenne).lower() in ("geometrique", "geometric", "log",
                                                "logarithmic", "logarithmique")
    if not geometrique and str(type_moyenne).lower() not in ("arithmetique", "arithmetic"):
        raise ValueError(
            "type_moyenne inconnu: '{}'. Utiliser 'arithmetique' ou 'geometrique'.".format(
                type_moyenne))

    n = len(chemins)
    if n == 0:
        raise ValueError("Aucune image a moyenner")

    accumulateur = carte_min = carte_max = None
    for i, chemin in enumerate(chemins):
        image = lire_hologramme(chemin, taille)

        if accumulateur is None:
            accumulateur = np.zeros_like(image)
            carte_min = np.full_like(image, np.inf)
            carte_max = np.full_like(image, -np.inf)
        elif image.shape != accumulateur.shape:
            raise ValueError(
                "L'image {} fait {} alors que les precedentes font {}. Utiliser "
                "l'argument 'taille' pour les recadrer a une taille commune.".format(
                    os.path.basename(chemin), image.shape, accumulateur.shape))

        # Cartes min/max sur les images BRUTES, avant le passage au logarithme.
        np.minimum(carte_min, image, out=carte_min)
        np.maximum(carte_max, image, out=carte_max)

        if geometrique:
            # On borne par le plus petit flottant positif plutot que d'ajouter un epsilon:
            # cela laisse les valeurs non nulles intactes au lieu de toutes les decaler.
            accumulateur += np.log(np.clip(image, np.finfo(np.float32).tiny, None))
        else:
            accumulateur += image

        if progress_callback is not None:
            progress_callback(i + 1, n)

    moyenne = accumulateur / n
    if geometrique:
        moyenne = np.exp(moyenne)

    moyenne = moyenne.astype(np.float32)
    for chemin in fichiers_sortie:
        sauver_hologramme(moyenne, chemin)

    return moyenne, carte_min.astype(np.float32), carte_max.astype(np.float32)


def lire_hologramme(chemin, taille=None):
    """Lit une image en float64, sans perte de dynamique.

    taille : (sizeX, sizeY) pour recadrer au centre, ou None pour l'image entiere.

    Aucune conversion en 8 bits: un Image.open(...).convert('L') ecraserait un TIF
    flottant (valeurs entre 0 et 1) ou un TIF 16 bits.
    """
    image = np.asarray(Image.open(chemin), dtype=np.float64)
    if image.ndim == 3:
        image = image.mean(axis=2)          # couleur -> niveaux de gris

    if taille is not None:
        sizeX, sizeY = int(taille[0]), int(taille[1])
        sy, sx = image.shape
        offsetX = (sx - sizeX) // 2
        offsetY = (sy - sizeY) // 2
        image = image[offsetY:offsetY + sizeY, offsetX:offsetX + sizeX]
    return image

def sum_plans(d_volum_focus):
    return(d_volum_focus.sum(axis = 0), d_volum_focus.sum(axis = 1), d_volum_focus.sum(axis = 2))

@jit.rawkernel()
def d_filter_FFT_3D(d_VOLUME_IN, d_VOLUME_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    # Les volumes sont ordonnes (Z, Y, X): l'acces se fait en [kk, jj, ii], et la borne
    # sur kk manquait, ce qui laissait des voxels hors volume etre ecrits.
    if (ii < sizeX and jj < sizeY and kk < sizeZ):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2
        centreZ = sizeZ // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))
        distanceZ = cp.abs(centreZ - kk)

        if ((distanceCentre > dMinXY) and (distanceCentre < dMaxXY ) and (distanceZ > dMinZ) and (distanceZ < dMaxZ )):
            d_VOLUME_OUT[kk, jj, ii] = d_VOLUME_IN[kk, jj, ii]
        else:
            d_VOLUME_OUT[kk, jj, ii] = 0.0 + 0.0j

def filtre_volume(d_FFT_volume_IN, d_FFT_volume_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ):

    nthread = 1024
    # ceil(a // b) ne fait rien: la division entiere tronque avant l'arrondi. Le nombre
    # de blocs etait donc insuffisant des que le total n'est pas multiple de nthread,
    # et les derniers voxels n'etaient jamais traites. (Ici sizeX etait de plus utilise
    # deux fois a la place de sizeY.)
    nBlock = (sizeX * sizeY * sizeZ + nthread - 1) // nthread

    d_filter_FFT_3D[nBlock, nthread](d_FFT_volume_IN, d_FFT_volume_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ)

def normalise_to_U8_volume(d_volume_IN):

    min = cp.min(d_volume_IN)
    max = cp.max(d_volume_IN)

    #d_volume_out = cp.zeros(dtype = cp.uint8, shape = d_volume_IN.shape)

    return(((d_volume_IN - min) * 255 / (max - min)).astype(cp.uint8))

def projection_bool(d_bin_volume, axis):
    """Projection 'au moins un voxel vrai' du volume binaire le long de l'axe demande.

    Le volume est ordonne (Z, Y, X):
      axis=0 -> plan (Y, X) : projection le long des plans de reconstruction
      axis=1 -> plan (Z, X) : projection le long de Y
      axis=2 -> plan (Z, Y) : projection le long de X

    ATTENTION: ne pas remplacer par cp.any(d_bin_volume, axis=...). Sur une installation
    ou nvcc ne peut pas compiler (version de Visual Studio non supportee par le CUDA
    Toolkit), les reductions cupy passant par CUB echouent - c'est le cas de
    cp.any(..., axis=2), de cp.count_nonzero et de ndarray.sum. Le noyau ci-dessous passe
    par NVRTC et fonctionne dans tous les cas.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis doit valoir 0, 1 ou 2 (recu: {axis})")

    sizeZ, sizeY, sizeX = d_bin_volume.shape
    nthread = 1024
    if axis == 0:
        d_projection = cp.zeros(shape=(sizeY, sizeX), dtype=cp.uint8)
        n_out = sizeY * sizeX
    elif axis == 1:
        d_projection = cp.zeros(shape=(sizeZ, sizeX), dtype=cp.uint8)
        n_out = sizeZ * sizeX
    else:
        d_projection = cp.zeros(shape=(sizeZ, sizeY), dtype=cp.uint8)
        n_out = sizeZ * sizeY

    # (n + nthread - 1) // nthread: le motif ceil(a // b) utilise auparavant tronquait,
    # laissant les derniers pixels de la projection non calcules.
    nBlock = (n_out + nthread - 1) // nthread
    d_projection_bool[nBlock, nthread](d_bin_volume, d_projection, sizeZ, sizeY, sizeX, axis)

    return d_projection

@jit.rawkernel()
def d_projection_bool(d_bin_volume, d_projection, sizeZ, sizeY, sizeX, axis):
    # Volume ordonne (Z, Y, X). Les noms de tailles etaient auparavant inverses
    # (sizeX, sizeY, sizeZ = d_bin_volume.shape), ce qui rendait le code illisible
    # meme si l'arithmetique tombait juste.
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if axis == 0:
        jj = int(index // sizeX)
        ii = int(index - jj * sizeX)
        if (jj < sizeY and ii < sizeX):
            val = 0
            for kk in range(sizeZ):
                if d_bin_volume[kk, jj, ii]:
                    val = 1
            d_projection[jj, ii] = val

    elif axis == 1:
        kk = int(index // sizeX)
        ii = int(index - kk * sizeX)
        if (kk < sizeZ and ii < sizeX):
            val = 0
            for jj in range(sizeY):
                if d_bin_volume[kk, jj, ii]:
                    val = 1
            d_projection[kk, ii] = val

    elif axis == 2:
        kk = int(index // sizeY)
        jj = int(index - kk * sizeY)
        if (kk < sizeZ and jj < sizeY):
            val = 0
            for ii in range(sizeX):
                if d_bin_volume[kk, jj, ii]:
                    val = 1
            d_projection[kk, jj] = val
