# -*- coding: utf-8 -*-

"""
Filename: propagation.py

Description:
Groups of functions needed for retro-propagates holograms (Fresnell, Rayleigh-Sommerfeld and angular spectrum).
Author: Simon BECKER
Date: 2024-07-09

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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
from numpy.fft import fft2 as np_fft2
from numpy.fft import ifft2 as np_ifft2
from numpy.fft import fftshift as np_fftshift
from numpy.fft import ifftshift as np_ifftshift

import cupy as cp
from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from cupyx import jit
from cupyx.scipy import ndimage as cp_ndimage
from traitement_holo import *

#########################################################################################################################################
################################               traitements des KERNELS               ####################################################
#########################################################################################################################################

@jit.rawkernel()
def d_filter_FFT(d_plan_IN, d_plan_OUT, sizeX, sizeY, dMin, dMax):
    #### Filtre passe bande   dMin>F>dMax avec dMin et dMax en pixel
    ### d_plan_IN et d_plan_OUT des plans complex sur GPU de taille sizeX et sizeY 

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    # ii = index - jj * sizeX. sizeY figurait ici: exact tant que l'image est carree,
    # faux des que sizeX != sizeY.
    jj = cp.int32(index) // cp.int32(sizeX)
    ii = cp.int32(index) - cp.int32(jj * sizeX)

    if (ii < sizeX and jj < sizeY):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))

        if ((distanceCentre > dMin) and (distanceCentre < dMax )):
            d_plan_OUT[jj, ii] = d_plan_IN[jj, ii]
        else:
            d_plan_OUT[jj, ii] = 0.0 + 0.0j

@jit.rawkernel()
def d_spec_filter_FFT(d_plan_IN, d_plan_OUT, sizeX, sizeY, dMin, dMax):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    # ii = index - jj * sizeX (voir d_filter_FFT ci-dessus).
    jj = cp.int32(index) // cp.int32(sizeX)
    ii = cp.int32(index) - cp.int32(jj * sizeX)

    if (ii < sizeX and jj < sizeY):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))

        if (dMin != 0 and distanceCentre < dMin) or (dMax != 0 and distanceCentre > dMax):
            d_plan_OUT[jj, ii] = 0.0 + 0.0j
        else:
            d_plan_OUT[jj, ii] = d_plan_IN[jj, ii]

        # if ((distanceCentre > dMin) and (distanceCentre < dMax )):
        #     #d_plan_OUT[ii, jj] = d_plan_IN[ii, jj] * cp.log(1 + (distanceCentre - dMin)/ (dMax - dMin))
        #     d_plan_OUT[jj, ii] = d_plan_IN[jj, ii]
        # else:
        #     d_plan_OUT[jj, ii] = 0.0 + 0.0j

#		spec_mask_filter_device[xy] = log(1 + (R - R_low));

#########################################################################################################################################
################################               calculs des KERNELS                  #####################################################
#########################################################################################################################################

@jit.rawkernel()
def d_calc_kernel_propag_Rayleigh_Sommerfeld(d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    
    ### calcul du noyau de propagation d_KERNEL selon la méthode Rayleigh_Sommerfeld
    ### H_propag = FFT-1 ( FFT(HOLO) * FFT(KERNEL_RAYLEIGH_SOMMERFELD)

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = nb_pix_X * nb_pix_Y

    jj = cp.int32(index) // cp.int32(nb_pix_X)
    ii = cp.int32(index) - cp.int32(jj * nb_pix_X)

    if (ii < nb_pix_X and jj < nb_pix_Y):

        K = cp.float32(2.0 * cp.pi / lambda_milieu)
        X = cp.float32( cp.int32(ii) - cp.int32(nb_pix_X // 2) )
        Y = cp.float32( cp.int32(jj) - cp.int32(nb_pix_Y // 2) )
        dpix = cp.float32(pixSize / magnification)
        mod = cp.float32(distance) / cp.float32(lambda_milieu * (distance * distance + X * X *dpix * dpix + Y * Y * dpix*dpix))
        # phase contient deja K = 2*pi/lambda: le facteur exp(2j*pi*phase) qui figurait
        # ici appliquait donc 2*pi une seconde fois.
        phase = cp.float32(K * cp.sqrt(distance * distance + X * X * dpix * dpix + Y * Y * dpix * dpix))
        d_KERNEL[jj, ii] = cp.complex64(mod * cp.exp(1.0j*phase))

    
@jit.rawkernel()
def d_calc_kernel_angular_spectrum_jit(d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):

    ### calcul du plan de phase d_KERNEL du noyau de propagation du spectre angulaire
    ###  H_propag = FFT-1 ( FFT(HOLO) * KERNEL_ANGULAR_SPECTRUM ) 

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = nb_pix_X * nb_pix_Y

    jj = index // nb_pix_X
    ii = index - jj * nb_pix_X

    if (ii < nb_pix_X and jj < nb_pix_Y):
        du = magnification / (pixSize * cp.float32(nb_pix_X))
        dv = magnification / (pixSize * cp.float32(nb_pix_Y))

        offset_u = nb_pix_X//2
        offset_v = nb_pix_Y//2

        U = ( cp.int32(ii) - cp.int32(offset_u) )*du
        V = ( cp.int32(jj) - cp.int32(offset_v) )*dv
        
        arg = 1.0 - cp.square(lambda_milieu * U) - cp.square(lambda_milieu *V )
        
        if(arg>0):
            d_KERNEL[jj, ii] = cp.exp(2 * 1j * cp.pi * distance * cp.sqrt(arg) / lambda_milieu)
        else:
            d_KERNEL[jj, ii] = 0.0+0j

@jit.rawkernel()
def d_propag_fresnel_phase1_jit(d_HOLO_IN, d_HOLO_OUT, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    jj = index // nb_pix_X
    ii = index - jj * nb_pix_X

    if (ii < nb_pix_X and jj < nb_pix_Y):
        offsetX = nb_pix_X//2
        offsetY = nb_pix_Y//2

        X = ii - offsetX
        Y = jj - offsetY

        dp_X = magnification * lambda_milieu * distance / (nb_pix_X * pixSize)
        dp_Y = magnification * lambda_milieu * distance / (nb_pix_Y * pixSize)

        arg = (cp.pi * 1.0j / (lambda_milieu * distance))  * (X*X*dp_X*dp_X + Y*Y*dp_Y*dp_Y)
        mod = 1.0j * cp.exp(2.0j*cp.pi*distance/lambda_milieu)/(lambda_milieu*distance)
        d_HOLO_OUT[jj, ii]=mod * cp.exp(arg) * d_HOLO_IN[jj, ii]

@jit.rawkernel()
def d_propag_fresnel_phase2_jit(d_HOLO_IN, d_HOLO_OUT, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    jj = index // nb_pix_X
    ii = index - jj * nb_pix_X

    if (ii < nb_pix_X and jj < nb_pix_Y):
        offsetX = nb_pix_X//2
        offsetY = nb_pix_Y//2

        X = ii - offsetX
        Y = jj - offsetY

        dp_X = pixSize / magnification
        dp_Y = pixSize / magnification

        arg = (cp.pi * 1.0j / (lambda_milieu * distance)) * (X*X*dp_X*dp_X + Y*Y*dp_Y*dp_Y)
        d_HOLO_OUT[jj, ii] = cp.exp(arg) * d_HOLO_IN[jj, ii]

#########################################################################################################################################
################################               propagations                  ############################################################
#########################################################################################################################################

def propag_angular_spectrum(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance, f_pix_min, f_pix_max):
    
    ### propagation de type spectre angulaire (méthode à 2 FFTs): propagation à utiliser pour les distance faibles
    ### dx et dy ne dépendent pas de la distance de propagation
    
    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread
    # Affectations en place '[:]': les tampons sont fournis par l'appelant. Ecrire
    # 'd_FFT_HOLO = ...' ne faisait que reaffecter la variable locale, l'appelant ne
    # voyait rien (sequelle des signatures C/CUDA d'origine).
    d_FFT_HOLO[:] = fftshift(fft2(d_HOLO, norm = 'ortho'))

    if ((f_pix_min != 0) or (f_pix_max != 0)):
        # La configuration de lancement [nBlock, nthread] manquait: l'appel levait un
        # TypeError des que le filtrage etait demande.
        d_filter_FFT[nBlock, nthread](d_FFT_HOLO, d_FFT_HOLO, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)

    d_calc_kernel_angular_spectrum_jit[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
    d_FFT_HOLO_PROPAG[:] = d_FFT_HOLO * d_KERNEL
    d_HOLO_PROPAG[:] = ifft2(ifftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')
    return(d_HOLO_PROPAG)

def propag_fresnell(d_HOLO, d_HOLO_2, d_FFT, d_HOLO_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    
    ### propagation de type Fresnell (méthode à 1 FFT): propagation à utiliser pour les distances grandes 
    ### dx et dy dépendent de la distance de propagation
    ### méthode pas encore testée...un peu merdique

    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread
    d_propag_fresnel_phase1_jit[nBlock, nthread](d_HOLO, d_HOLO_2, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
    d_FFT = fftshift(fft2(d_HOLO_2))
    d_propag_fresnel_phase2_jit[nBlock, nthread](d_FFT, d_HOLO_PROPAG, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)

def propag_Rayleigh_Sommerfeld(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    
    ### propagation de type Rayleigh Sommerfeld (méthode à 3 FFTs): propagation à utiliser pour les distances grandes 
    ### dx et dy ne dépendent pas de la distance de propagation

    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread
    # Affectations en place (voir propag_angular_spectrum).
    d_FFT_HOLO[:] = fftshift(fft2(d_HOLO, norm = 'ortho'))
    d_calc_kernel_propag_Rayleigh_Sommerfeld[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
    d_FFT_KERNEL[:] = fftshift(fft2(d_KERNEL, norm = 'ortho'))
    d_FFT_HOLO_PROPAG[:] = d_FFT_HOLO * d_FFT_KERNEL
    d_HOLO_PROPAG[:] = ifft2(ifftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')
    return(d_HOLO_PROPAG)

#########################################################################################################################################
################################               Calculs volumes               ############################################################
#########################################################################################################################################

def volume_propag_angular_spectrum_complex(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distancePropagIni, pasPropag, nbPropag, f_pix_min, f_pix_max):

    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread

    d_FFT_HOLO[:] = fftshift(fft2(d_HOLO, norm = 'ortho'))

    # 'or' et non 'and': une seule des deux bornes suffit a demander un filtrage, comme
    # dans spec_filter_FFT. (Un print de debogage figurait aussi ici.)
    if ((f_pix_min != 0) or (f_pix_max != 0)):
        d_spec_filter_FFT[nBlock, nthread](d_FFT_HOLO, d_FFT_HOLO, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)

    for i in range(nbPropag):
        distance = distancePropagIni + i * pasPropag
        d_calc_kernel_angular_spectrum_jit[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        d_FFT_HOLO_PROPAG[:] = d_FFT_HOLO * d_KERNEL
        # Volume ordonne (Z, Y, X) comme partout ailleurs: le plan i s'ecrit en [i,:,:].
        d_HOLO_VOLUME_PROPAG[i,:,:] = ifft2(ifftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')

def spec_filter_FFT(d_FFT_HOLO, d_FFT_HOLO_FILTERED, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max):

    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread
    
    if f_pix_min != 0 or f_pix_max != 0:
        # Apply filtering: d_FFT_HOLO -> d_FFT_HOLO_FILTERED
        d_spec_filter_FFT[nBlock, nthread](d_FFT_HOLO, d_FFT_HOLO_FILTERED, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)
    else:
        # No filtering: d_FFT_HOLO_FILTERED = d_FFT_HOLO (both should be identical)
        d_FFT_HOLO_FILTERED[:] = cp.copy(d_FFT_HOLO)
    

def volume_propag_angular_spectrum_to_module(d_HOLO, d_FFT_HOLO, d_FFT_HOLO_FILTERED, d_KERNEL, d_HOLO_FILTERED, d_FFT_HOLO_PROPAG, d_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG_MODULE,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distancePropagIni, pasPropag, nbPropag, f_pix_min, f_pix_max):
    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread

    d_FFT_HOLO[:] = fftshift(fft2(d_HOLO, norm = 'ortho'))

    spec_filter_FFT(d_FFT_HOLO, d_FFT_HOLO_FILTERED,
                    nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)
        
    d_HOLO_FILTERED[:] = cp.abs(ifft2(ifftshift(d_FFT_HOLO_FILTERED), norm = 'ortho'))

    for i in range(nbPropag):
        distance = distancePropagIni + i * pasPropag
        d_calc_kernel_angular_spectrum_jit[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        d_FFT_HOLO_PROPAG[:] = d_FFT_HOLO_FILTERED * d_KERNEL
        d_HOLO_PROPAG[:] = ifft2(ifftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')
        d_HOLO_VOLUME_PROPAG_MODULE[i,:,:] = cp.abs(d_HOLO_PROPAG)

def volume_propag_Rayleigh_Sommerfeld(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, pasPropag, nbPropag):

    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread

    # Affectations en place, et volume ordonne (Z, Y, X): le plan i s'ecrit en [i,:,:].
    d_FFT_HOLO[:] = fftshift(fft2(d_HOLO, norm = 'ortho'))
    for i in range(nbPropag):
        distance = (i + 1)* pasPropag
        d_calc_kernel_propag_Rayleigh_Sommerfeld[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        d_FFT_KERNEL[:] = fftshift(fft2(d_KERNEL, norm = 'ortho'))
        d_FFT_HOLO_PROPAG[:] = d_FFT_HOLO * d_FFT_KERNEL
        d_HOLO_VOLUME_PROPAG[i,:,:] = ifft2(ifftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')

def volume_propag_fresnell(d_HOLO, d_Holo_temp, d_FFT, d_HOLO_VOLUME_PROPAG, 
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, pasPropag, nbPropag):

    nthread = 1024
    # ceil(a // b) est un faux arrondi: la division entiere tronque avant l'appel a ceil,
    # donc les derniers pixels n'etaient pas couverts quand nb_pix_X*nb_pix_Y n'est pas
    # multiple de nthread (cas exact en 1024x1024, faux des qu'on change de capteur).
    nBlock = (nb_pix_X * nb_pix_Y + nthread - 1) // nthread

    for i in range(nbPropag):
        distance = (i + 1)* pasPropag
        d_propag_fresnel_phase1_jit[nBlock, nthread](d_HOLO, d_Holo_temp, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        d_FFT[:] = fftshift(fft2(d_Holo_temp, norm = 'ortho'))
        # Volume ordonne (Z, Y, X): le plan i s'ecrit en [i,:,:].
        d_propag_fresnel_phase2_jit[nBlock, nthread](d_FFT, d_HOLO_VOLUME_PROPAG[i,:,:], lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)

