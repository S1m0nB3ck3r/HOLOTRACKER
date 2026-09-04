# -*- coding: utf-8 -*-
"""
Filename: pipeline_reference.py

Description:
Exécution instrumentée du pipeline de localisation, utilisée à la fois pour produire le
fichier de référence et pour le test de non-régression. Les deux passent par la MÊME
fonction, sinon la référence et le test pourraient diverger sans qu'on le voie.

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import os
import sys
import json

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code_locate"))

import hashlib

import numpy as np
import cupy as cp

from core import HoloTrackerCore
import propagation as propag
import display

REFERENCE_DIR = os.path.join(TESTS_DIR, "reference")
PARAMS_FILE = os.path.join(REFERENCE_DIR, "params.json")
EXPECTED_FILE = os.path.join(REFERENCE_DIR, "expected.json")

# Nombre de classes des histogrammes. Les bornes ne sont PAS recalculées à chaque
# exécution: elles sont figées dans le fichier de référence, sinon les classes se
# décaleraient d'une exécution à l'autre et les effectifs ne seraient plus comparables.
HIST_BINS = 40

# Toutes les combinaisons proposees par l'interface (voir ui.py). Les images produites
# sont comparees par empreinte exacte: deux executions consecutives donnent des images
# identiques au bit pres, verifie sur les 48 combinaisons.
DISPLAY_TYPES = [
    "RAW_HOLOGRAM", "CLEANED_HOLOGRAM", "FILTERED_HOLOGRAM",
    "FFT_HOLOGRAM", "FFT_FILTERED_HOLOGRAM", "VOLUME_PLANE_NUMBER",
    "XY_SUM_PROJECTION", "XZ_SUM_PROJECTION", "YZ_SUM_PROJECTION",
    "XY_MAX_PROJECTION", "XZ_MAX_PROJECTION", "YZ_MAX_PROJECTION",
]
ADDITIONAL_DISPLAYS = ["None", "Centroid positions", "Segmentation", "Segmentation + Centroid"]

# Plan utilise pour VOLUME_PLANE_NUMBER: le milieu du volume de reference (200 plans).
DISPLAY_PLANE = 100

# Points ou la valeur du pixel est relevee (get_pixel_value), en coordonnees image.
# Volontairement varies: un coin, le centre, et deux points quelconques.
PIXEL_PROBES = [(0, 0), (512, 512), (137, 890), (1000, 20)]

# Taille de la vignette extraite autour d'un objet (extract_object_slices).
SLICE_VOX_XY = 32
SLICE_VOX_Z = 32


def load_params():
    """Paramètres FIGÉS du test.

    Volontairement séparés de last_param.json: ce dernier est réécrit par l'interface à
    chaque utilisation, un test qui le lirait changerait de résultat sans qu'on touche au
    code.
    """
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return json.load(f)


def test_image_path():
    """Première image de Film REF/cleaned (l'ordre alphabétique des .tif, pas des .csv)."""
    directory = os.path.join(PROJECT_ROOT, "Film REF", "cleaned")
    images = sorted(f for f in os.listdir(directory) if f.lower().endswith(".tif"))
    if not images:
        raise FileNotFoundError(f"Aucune image .tif dans {directory}")
    return directory, images[0]


def _describe(array, hist_range=None, bins=HIST_BINS):
    """Statistiques et histogramme d'un tableau (numpy ou cupy).

    hist_range : bornes imposées (celles de la référence). Si None, on prend min/max.
    Note: cp.percentile et cp.count_nonzero ne compilent pas sur toutes les installations
    (nvcc indisponible), on s'en tient à mean/std/min/max/sum et cp.histogram.
    """
    xp = cp if isinstance(array, cp.ndarray) else np
    lo = float(xp.min(array))
    hi = float(xp.max(array))
    if hist_range is None:
        hist_range = (lo, hi)
    counts, _ = xp.histogram(array, bins=bins, range=tuple(hist_range))
    return {
        "min": lo,
        "max": hi,
        "mean": float(xp.mean(array)),
        "std": float(xp.std(array)),
        "sum": float(xp.sum(array.astype(xp.float64) if xp is np else array)),
        "hist_range": [float(hist_range[0]), float(hist_range[1])],
        "hist": [int(c) for c in (cp.asnumpy(counts) if xp is cp else counts)],
    }


def _capture_display(core, directory, filename):
    """Empreinte de chaque image d'affichage, pour les 12 x 4 combinaisons de l'interface.

    L'empreinte est un SHA-256 tronque du tableau de pixels: c'est suffisant pour detecter
    tout changement de rendu, et cela evite de stocker des images dans le depot.

    IMPORTANT: a appeler AVANT toute reutilisation de d_volume_module, qui contient a ce
    moment le volume de focus dont depend l'affichage.
    """
    images = {}
    for display_type in DISPLAY_TYPES:
        for additional in ADDITIONAL_DISPLAYS:
            image = display.get_display_image(
                core,                directory, filename, display_type,
                plane_number=DISPLAY_PLANE, additional_display=additional)
            cle = f"{display_type}|{additional}"
            if image is None:
                images[cle] = None
                continue
            pixels = np.asarray(image)
            images[cle] = {
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()[:16],
            }
    return images


def _capture_pixel_values(core, directory, filename):
    """Valeur du pixel sous le curseur, pour chaque mode d'affichage et quelques points.

    C'est le second chemin de rendu (get_pixel_value): il relit les memes donnees que
    l'affichage mais par un code distinct, il merite donc sa propre couverture.
    """
    valeurs = {}
    for display_type in DISPLAY_TYPES:
        for (x, y) in PIXEL_PROBES:
            v = display.get_pixel_value(core, directory, filename, display_type, DISPLAY_PLANE, x, y)
            valeurs[f"{display_type}|{x},{y}"] = None if v is None else float(v)
    return valeurs


def _capture_object_slices(core, objects):
    """Vignettes XY, XZ et YZ extraites autour du plus gros objet detecte.

    Le plus gros objet est choisi parce qu'il est stable: un objet de 2 voxels pourrait
    changer de rang entre deux executions, pas celui de 4056.
    """
    if not objects:
        return None
    plus_gros = max(objects, key=lambda o: o[3])
    x_um, y_um, z_um = (c * 1e6 for c in plus_gros[:3])
    coupes = display.extract_object_slices(core, x_um, y_um, z_um, SLICE_VOX_XY, SLICE_VOX_Z)
    out = {"objet_um": [round(x_um, 6), round(y_um, 6), round(z_um, 6)],
           "nb_voxels": plus_gros[3]}
    for nom, tableau in coupes.items():
        a = np.asarray(tableau)
        out[nom] = {"shape": list(a.shape),
                    "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]}
    return out


def run_pipeline(params, ranges=None):
    """Exécute le pipeline complet sur l'image de test et renvoie ses grandeurs.

    ranges : bornes d'histogramme issues de la référence, pour que les effectifs soient
    comparables. None lors de la création de la référence.
    """
    ranges = ranges or {}
    directory, filename = test_image_path()

    core = HoloTrackerCore()
    core.set_parameters(**params)
    core.allocate()
    core.enter_test_mode()

    message = core.process_hologram_complete_pipeline(directory, filename)
    if "error" in core.results:
        raise RuntimeError(f"Le pipeline a signalé une erreur: {core.results['error']}")

    features = core.results["features"]
    objects = [[float(f[1]), float(f[2]), float(f[3]), int(f[4])] for f in features]
    objects.sort(key=lambda o: (round(o[0], 12), round(o[1], 12), round(o[2], 12)))

    out = {
        "image": filename,
        "n_objects": int(core.results["number_of_objects"]),
        "threshold": float(core.threshold),
        "objects": objects,
        "holo_cleaned": _describe(core.h_cleaned_holo, ranges.get("holo_cleaned")),
        "holo_filtered": _describe(core.h_filtered_holo, ranges.get("holo_filtered")),
        "volume_focus": _describe(core.d_volume_module, ranges.get("volume_focus")),
    }

    # Affichage: capture AVANT la re-propagation ci-dessous, qui écrase le volume de focus.
    out["display"] = _capture_display(core, directory, filename)
    out["pixel_values"] = _capture_pixel_values(core, directory, filename)
    out["object_slices"] = _capture_object_slices(core, objects)

    # Volume propagé AVANT le critère de focus. Le pipeline applique focus() en place, il
    # faut donc refaire la propagation seule, avec exactement le même appel.
    dx = float(core.pixel_size) / float(core.objective_magnification)
    propag.volume_propag_angular_spectrum_to_module(
        core.d_holo, core.d_fft_holo, core.d_fft_holo_filtered, core.d_KERNEL,
        core.d_filtered_holo, core.d_fft_holo_propag, core.d_holo_propag,
        core.d_volume_module,
        float(core.wavelength) / float(core.medium_optical_index),
        float(core.objective_magnification), float(core.pixel_size),
        int(core.holo_size_x), int(core.holo_size_y),
        float(core.distance_ini), float(core.step), int(core.number_of_planes),
        int(core.high_pass), int(core.low_pass))
    out["volume_propag"] = _describe(core.d_volume_module, ranges.get("volume_propag"))
    out["voxel_size_m"] = [dx, dx, float(core.step)]

    core.cleanup_test_mode()
    return out
