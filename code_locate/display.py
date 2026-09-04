# -*- coding: utf-8 -*-

"""
Filename: display.py

Description:
Fabrication des images affichées par l'interface: vues de l'hologramme, plans et
projections du volume reconstruit, superpositions de centroïdes et de segmentation,
vignettes autour d'un objet, valeur du pixel sous le curseur.

Ce module ne calcule rien: il lit les tableaux déjà produits par HoloTrackerCore et les
transforme en images. Le découpage sépare deux métiers qui cohabitaient dans core.py:
la reconstruction holographique d'un côté, le rendu de l'autre.

Chaque fonction reçoit le coeur en premier argument et y lit ses données
(core.d_volume_module, core.h_cleaned_holo, core.results...). L'appel se lit donc:

    import display
    image = display.get_display_image(core, repertoire, fichier, "XY_MAX_PROJECTION")

Fonctions publiques:
    get_display_image         image d'un mode d'affichage, avec superposition éventuelle
    get_pixel_value           valeur brute sous le curseur, pour le même mode
    extract_object_slices     vignettes XY, XZ et YZ autour d'un objet
    get_default_display_type  mode conseillé selon les paramètres

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

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

import os

import numpy as np
from PIL import Image

try:
    import cupy as cp
    # projection_bool: projection "au moins un voxel vrai" du volume binaire, utilisée
    # par la superposition Segmentation.
    from traitement_holo import projection_bool
except ImportError:
    cp = None
    projection_bool = None

# Journal: voir holo_log.py. Les exceptions ignorées y laissent une trace,
# avec fichier, fonction et ligne, sans interrompre l'application.
import logging
log = logging.getLogger(__name__)


def stat_plane(data, label=""):
    """
    Debug function: compute and print statistics for a CuPy array
    If complex, computes abs() first
    """
    try:
        # Handle complex data
        if hasattr(data, 'dtype') and np.iscomplexobj(cp.asnumpy(data[:1, :1])):
            data_abs = cp.abs(data)
            complex_info = " (complex -> abs)"
        else:
            data_abs = data
            complex_info = ""
            
        # Compute statistics
        data_sum = float(cp.sum(data_abs))
        data_min = float(cp.min(data_abs))
        data_max = float(cp.max(data_abs))
        data_mean = float(cp.mean(data_abs))
        data_std = float(cp.std(data_abs))
        
        # Trace de débogage: dans le journal (holotracker.log), pas sur la sortie standard.
        # Cette fonction est appelée à chaque rafraîchissement de l'affichage; en print(),
        # elle noyait la console et la sortie des tests.
        log.debug("STAT %s%s: sum=%.3f, min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                  label, complex_info, data_sum, data_min, data_max, data_mean, data_std)
        
        return {
            'sum': data_sum,
            'min': data_min, 
            'max': data_max,
            'mean': data_mean,
            'std': data_std
        }
    except Exception as e:
        # Error computing statistics

        return None

def _to_uint8(core, arr):
    """Normalize array to uint8 for display (0-255 range)"""
    arr = np.array(arr)  # ensure numpy
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = arr.astype(np.float64)

    mx = arr.max() if arr.size else 0.0
    mn = arr.min() if arr.size else 0.0
    if mx > mn:
        out = ((arr - mn) * 255.0 / (mx - mn)).astype(np.uint8)
    elif mx > 0:
        out = (arr * 255.0 / mx).astype(np.uint8)
    else:
        out = np.zeros_like(arr, dtype=np.uint8)
    return out

def open_hologram_image(core, directory, filename):
    filepath = os.path.join(directory, filename)
    img = Image.open(filepath)
    # Resize if parameters are present
    x = int(core.holo_size_x)
    y = int(core.holo_size_y)
    img = img.resize((x, y))
    return img

def get_display_image(core, directory, filename, display_type, plane_number=0, additional_display="None", use_log=False):
    """Retourne l'image à afficher selon le type demandé avec marqueurs de détection"""
    try:

        # Provide a safe getter for raw hologram (prefer in-memory)
        def _get_raw_array():
            if core.h_raw_holo is not None:
                return core.h_raw_holo.copy()
            # fallback to reading file
            raw_hologram = open_hologram_image(core, directory, filename)
            return np.array(raw_hologram.convert('L'), dtype=np.float64)

        # Common fallback behavior: print minimal notice and return raw hologram display
        def _fallback(type_name):
            # print(f"no '{type_name}' available")
            arr = _get_raw_array()
            if use_log:
                arr = np.log(arr + 1e-10)
            return Image.fromarray(_to_uint8(core, arr))

        # RAW_HOLOGRAM: use in-memory if available
        if display_type == "RAW_HOLOGRAM":
            arr = _get_raw_array()
            stat_plane(arr, label="RAW_HOLOGRAM")
            if use_log:
                arr = np.log(arr + 1e-10)
            image = Image.fromarray(_to_uint8(core, arr))
            return _apply_additional_display(core, image, additional_display, display_type, plane_number)

        # CLEANED_HOLOGRAM
        if display_type == "CLEANED_HOLOGRAM":
            if core.memory_allocated and core.h_cleaned_holo is not None:
                arr = core.h_cleaned_holo.copy()
                stat_plane(arr, label="CLEANED_HOLOGRAM")
                if use_log:
                    arr = np.log(arr + 1e-10)
                image = Image.fromarray(_to_uint8(core, arr))
                return _apply_additional_display(core, image, additional_display, display_type, plane_number)
            return _fallback("CLEANED_HOLOGRAM")

        # FILTERED_HOLOGRAM
        if display_type == "FILTERED_HOLOGRAM":
            # Try CPU version first (more efficient)
            if core.memory_allocated and core.h_filtered_holo is not None:
                try:
                    h_filtered = core.h_filtered_holo.copy()
                    if np.iscomplexobj(h_filtered):
                        h_filtered = np.abs(h_filtered)
                    stat_plane(h_filtered, label="FILTERED_HOLOGRAM")
                    if use_log:
                        h_filtered = np.log(h_filtered + 1e-10)
                    image = Image.fromarray(_to_uint8(core, h_filtered))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)
            # Fallback to GPU version
            if core.memory_allocated and core.d_filtered_holo is not None:
                try:
                    h_filtered = cp.asnumpy(core.d_filtered_holo)
                    if np.iscomplexobj(h_filtered):
                        h_filtered = np.abs(h_filtered)
                    stat_plane(h_filtered, label="FILTERED_HOLOGRAM")
                    if use_log:
                        h_filtered = np.log(h_filtered + 1e-10)
                    image = Image.fromarray(_to_uint8(core, h_filtered))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number)
                except Exception:
                    return _fallback("FILTERED_HOLOGRAM")
            return _fallback("FILTERED_HOLOGRAM")

        # FFT_HOLOGRAM
        if display_type == "FFT_HOLOGRAM":
            if core.memory_allocated and core.d_fft_holo is not None:
                try:
                    h_fft = cp.asnumpy(core.d_fft_holo)
                    if np.iscomplexobj(h_fft):
                        h_fft = np.abs(h_fft)
                    stat_plane(h_fft, label="FFT_HOLOGRAM")
                    if use_log:
                        h_fft = np.log(h_fft + 1e-10)
                    image = Image.fromarray(_to_uint8(core, h_fft))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number)
                except Exception:
                    return _fallback("FFT_HOLOGRAM")
            return _fallback("FFT_HOLOGRAM")

        # FFT_FILTERED_HOLOGRAM
        if display_type == "FFT_FILTERED_HOLOGRAM":
            if core.memory_allocated and core.d_fft_holo_filtered is not None:
                try:
                    h_fft_filtered = cp.asnumpy(core.d_fft_holo_filtered)
                    if np.iscomplexobj(h_fft_filtered):
                        h_fft_filtered = np.abs(h_fft_filtered)
                    stat_plane(h_fft_filtered, label="FFT_FILTERED_HOLOGRAM")
                    if use_log:
                        h_fft_filtered = np.log(h_fft_filtered + 1e-10)
                    image = Image.fromarray(_to_uint8(core, h_fft_filtered))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number)
                except Exception:
                    return _fallback("FFT_FILTERED_HOLOGRAM")
            return _fallback("FFT_FILTERED_HOLOGRAM")

        # VOLUME_PLANE_NUMBER and projections - keep original logic but use _fallback on error
        if display_type == "VOLUME_PLANE_NUMBER":
            if core.memory_allocated:
                try:
                    volume_gpu = core.d_volume_module
                    if plane_number < volume_gpu.shape[0]:
                        plane = cp.asnumpy(volume_gpu[plane_number, :, :])
                        plane = np.abs(plane)
                        stat_plane(plane, label=f"VOLUME_PLANE_{plane_number}")
                        if use_log:
                            plane = np.log(plane + 1e-10)
                        image = Image.fromarray(_to_uint8(core, plane))
                        return _apply_additional_display(core, image, additional_display, display_type, plane_number)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)
            return _fallback("VOLUME_PLANE_NUMBER")

        if display_type == "XY_SUM_PROJECTION":
            if core.memory_allocated:
                try:
                    projection = cp.sum(core.d_volume_module, axis=0)
                    projection = cp.asnumpy(projection)
                    stat_plane(projection, label="XY_SUM_PROJECTION")
                    if use_log:
                        projection = np.log(projection + 1e-10)
                    image = Image.fromarray(_to_uint8(core, projection))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number, 'XY')
                except Exception:
                    log.debug("exception ignorée", exc_info=True)
            return _fallback("XY_SUM_PROJECTION")

        if display_type == "XZ_SUM_PROJECTION":
            if core.memory_allocated:
                try:
                    projection = cp.sum(core.d_volume_module, axis=1)
                    projection = cp.asnumpy(projection)
                    stat_plane(projection, label="XZ_SUM_PROJECTION")
                    if use_log:
                        projection = np.log(projection + 1e-10)
                    image = Image.fromarray(_to_uint8(core, projection))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number, 'XZ')
                except Exception:
                    log.debug("exception ignorée", exc_info=True)
            return _fallback("XZ_SUM_PROJECTION")

        if display_type == "YZ_SUM_PROJECTION":
            if core.memory_allocated:
                try:

                    # Test with smaller slice first
                    test_slice = core.d_volume_module[:10, :10, :]
                    test_result = cp.sum(test_slice, axis=2)

                    # Try CUDA operation first, fallback to CPU if it fails
                    try:
                        projection = cp.sum(core.d_volume_module, axis=2)
                        projection = cp.asnumpy(projection)

                    except Exception as cuda_error:
                        print(f"CUDA error in YZ_SUM_PROJECTION, falling back to CPU: {cuda_error}")
                        # Fallback to CPU computation
                        volume_cpu = cp.asnumpy(core.d_volume_module)
                        projection = np.sum(volume_cpu, axis=2)

                    stat_plane(projection, label="YZ_SUM_PROJECTION")
                    if use_log:
                        projection = np.log(projection + 1e-10)
                    image = Image.fromarray(_to_uint8(core, projection))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number, 'YZ')
                except Exception as e:
                    print(f"Error in YZ_SUM_PROJECTION: {e}")
                    pass
            return _fallback("YZ_SUM_PROJECTION")

        if display_type == "XY_MAX_PROJECTION":
            if core.memory_allocated:
                try:
                    projection = cp.max(core.d_volume_module, axis=0)
                    projection = cp.asnumpy(projection)
                    stat_plane(projection, label="XY_MAX_PROJECTION")
                    if use_log:
                        projection = np.log(projection + 1e-10)
                    image = Image.fromarray(_to_uint8(core, projection))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number, 'XY')
                except Exception:
                    log.debug("exception ignorée", exc_info=True)
            return _fallback("XY_MAX_PROJECTION")

        if display_type == "XZ_MAX_PROJECTION":
            if core.memory_allocated:
                try:
                    projection = cp.max(core.d_volume_module, axis=1)
                    projection = cp.asnumpy(projection)
                    stat_plane(projection, label="XZ_MAX_PROJECTION")
                    if use_log:
                        projection = np.log(projection + 1e-10)
                    image = Image.fromarray(_to_uint8(core, projection))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number, 'XZ')
                except Exception:
                    log.debug("exception ignorée", exc_info=True)
            return _fallback("XZ_MAX_PROJECTION")

        if display_type == "YZ_MAX_PROJECTION":
            if core.memory_allocated:
                try:
                    # print(f"Debug: Computing YZ_MAX_PROJECTION, volume shape: {core.d_volume_module.shape}")
                    # Try CUDA operation first, fallback to CPU if it fails
                    try:
                        projection = cp.max(core.d_volume_module, axis=2)
                        projection = cp.asnumpy(projection)
                    except Exception as cuda_error:
                        print(f"CUDA error in YZ_MAX_PROJECTION, falling back to CPU: {cuda_error}")
                        # Fallback to CPU computation
                        volume_cpu = cp.asnumpy(core.d_volume_module)
                        projection = np.max(volume_cpu, axis=2)

                    stat_plane(projection, label="YZ_MAX_PROJECTION")
                    if use_log:
                        projection = np.log(projection + 1e-10)
                    image = Image.fromarray(_to_uint8(core, projection))
                    return _apply_additional_display(core, image, additional_display, display_type, plane_number, 'YZ')
                except Exception as e:
                    # print(f"Error in YZ_MAX_PROJECTION: {e}")
                    log.debug("exception ignorée", exc_info=True)
            return _fallback("YZ_MAX_PROJECTION")

        # Default: show raw hologram
        arr = _get_raw_array()
        if use_log:
            arr = np.log(arr + 1e-10)
        return Image.fromarray(_to_uint8(core, arr))

    except Exception as e:
        # Minimal error reporting
        # print(f"Error in get_display_image: {e}")
        try:
            arr = core.h_raw_holo if core.h_raw_holo is not None else np.zeros((100,100), dtype=np.float64)
            if use_log:
                arr = np.log(arr + 1e-10)
            return Image.fromarray(_to_uint8(core, arr))
        except:
            return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))

def _apply_additional_display(core, image, additional_display, display_type, plane_number, projection_type=None):
    """Apply additional display overlay based on the selected option"""
    try:
        # No overlays for FFT displays
        if display_type in ["FFT_HOLOGRAM", "FFT_FILTERED_HOLOGRAM"]:
            return image

        if additional_display == "None":
            return image
        elif additional_display == "Centroid positions":
            # Show ONLY centroid markers (red dots), no segmentation
            return _add_centroid_overlay(core, image, display_type, plane_number, projection_type)
        elif additional_display == "Segmentation":
            # Show ONLY segmentation (blue voxels), no centroid markers
            return _add_segmentation_overlay(core, image, display_type, plane_number, projection_type)
        elif additional_display == "Segmentation + Centroid":
            # Show BOTH segmentation (blue voxels) AND centroid markers (red dots)
            return _add_segmentation_and_centroid_overlay(core, image, display_type, plane_number, projection_type)
        else:
            return image
    except Exception as e:
        # print(f"Error applying additional display: {e}")
        return image

def _add_centroid_overlay(core, image, display_type, plane_number, projection_type=None):
    """Add centroid position markers (red dots) to the image"""
    try:
        if display_type == "VOLUME_PLANE_NUMBER":
            # For volume plane: show only markers for the specific plane
            return _add_centroid_markers_for_plane(core, image, plane_number)
        elif projection_type:
            # For projections: use the appropriate projection method
            return add_detection_markers_to_image_2d_projection(core, image, core.results, projection_type)
        else:
            # For hologram images: use standard marker method
            return add_detection_markers_to_image(core, image, core.results)
    except Exception as e:
        # print(f"Error adding centroid overlay: {e}")
        return image

def _add_segmentation_overlay(core, image, display_type, plane_number, projection_type=None):
    """Add segmentation overlay to the image"""
    try:
        if core.d_bin_volume_focus is None:
            return image

        if display_type == "VOLUME_PLANE_NUMBER":
            # For volume plane: show segmentation for the specific plane
            if plane_number < core.d_bin_volume_focus.shape[0]:
                bin_plane = cp.asnumpy(core.d_bin_volume_focus[plane_number, :, :])
                return _blend_segmentation_with_image(core, image, bin_plane)
        elif projection_type == 'XY':
            # Sum or max projection along Z axis (axis=0) using projection_bool
            bin_projection = projection_bool(core.d_bin_volume_focus, axis=0)
            bin_projection = cp.asnumpy(bin_projection)
            return _blend_segmentation_with_image(core, image, bin_projection)
        elif projection_type == 'XZ':
            # Sum or max projection along Y axis (axis=1) using projection_bool
            bin_projection = projection_bool(core.d_bin_volume_focus, axis=1)
            bin_projection = cp.asnumpy(bin_projection)
            return _blend_segmentation_with_image(core, image, bin_projection)
        elif projection_type == 'YZ':
            # Sum or max projection along X axis (axis=2) using projection_bool
            bin_projection = projection_bool(core.d_bin_volume_focus, axis=2)
            bin_projection = cp.asnumpy(bin_projection)
            return _blend_segmentation_with_image(core, image, bin_projection)
        else:
            # For hologram images: use XY projection with projection_bool
            bin_projection = projection_bool(core.d_bin_volume_focus, axis=0)
            bin_projection = cp.asnumpy(bin_projection)
            return _blend_segmentation_with_image(core, image, bin_projection)

        return image
    except Exception as e:
        # print(f"Error adding segmentation overlay: {e}")
        return image

def _add_segmentation_and_centroid_overlay(core, image, display_type, plane_number, projection_type=None):
    """Add both segmentation overlay AND centroid markers to the image

    This shows:
    - Blue voxels (segmentation from _add_segmentation_overlay)
    - Red dots (centroid markers from _add_centroid_overlay)
    """
    try:
        # First add segmentation (blue voxels)
        image = _add_segmentation_overlay(core, image, display_type, plane_number, projection_type)

        # Then add centroid markers (red dots) on top
        image = _add_centroid_overlay(core, image, display_type, plane_number, projection_type)

        return image
    except Exception as e:
        # print(f"Error adding segmentation + centroid overlay: {e}")
        return image

def _add_centroid_markers_for_plane(core, image, plane_number):
    """Add centroid markers for a specific plane only"""
    try:
        import cv2

        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # Add red circles for each detection on this specific plane
        if 'features' in core.results and core.results['features'] is not None:
            features = core.results['features']
            for feature in features:
                # Extract coordinates (baryX, baryY, baryZ are now in meters)
                x_m = feature[1]  # baryX in meters
                y_m = feature[2]  # baryY in meters
                z_m = feature[3]  # baryZ in meters

                # Convert from meters to micrometers
                x_um = x_m * 1e6
                y_um = y_m * 1e6
                z_um = z_m * 1e6

                # Convert Z position to plane number
                cam_pix_size = float(core.pixel_size)  # Use same parameter name
                cam_magnification = float(core.objective_magnification)  # Use same parameter name
                dz = float(core.step)  # Step size in meters

                # Calculate the plane number for this feature
                feature_plane = int(z_um / (dz * 1e6))  # Convert to plane index

                # Only show markers for features in the current plane (with some tolerance)
                if abs(feature_plane - plane_number) <= 1:  # Allow ±1 plane tolerance
                    # Convert from micrometers to pixels
                    effective_pixel_size = cam_pix_size / cam_magnification
                    x_pix = int(x_um * 1e-6 / effective_pixel_size)
                    y_pix = int(y_um * 1e-6 / effective_pixel_size)

                    # Draw red filled dot (not a circle outline)
                    cv2.circle(img_array, (x_pix, y_pix), 3, (255, 0, 0), -1)  # Red filled circle, radius 3

        return Image.fromarray(img_array)
    except Exception as e:
        # print(f"Error adding markers for plane: {e}")
        return image

def _blend_segmentation_with_image(core, image, segmentation_data):
    """Blend segmentation data as overlay with the original image"""
    try:
        import cv2

        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # Debug information

        # Ensure segmentation_data matches image dimensions
        img_h, img_w = img_array.shape[:2]
        seg_h, seg_w = segmentation_data.shape

        if seg_h != img_h or seg_w != img_w:
            # Resize segmentation to match image
            segmentation_data = cv2.resize(segmentation_data.astype(np.float32), (img_w, img_h))

        # Create binary mask for segmentation
        if segmentation_data.max() > 0:
            # Create binary mask where segmentation is present
            mask = (segmentation_data > 0).astype(np.uint8)

        else:
            mask = np.zeros_like(segmentation_data, dtype=np.uint8)

        # Create result image starting with original
        result = img_array.copy()

        # Apply solid blue color where segmentation is present
        result[mask > 0, 2] = 255  # Blue channel at maximum
        result[mask > 0, 0] = 0    # Red channel to 0
        result[mask > 0, 1] = 0    # Green channel to 0

        return Image.fromarray(result)
    except Exception as e:
        # print(f"Error blending segmentation: {e}")
        return image

def add_detection_markers_to_image(core, image, detections):
    """Add red markers to show detected objects on hologram"""
    try:
        import cv2

        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # Add red circles for each detection
        if 'features' in core.results and core.results['features'] is not None:
            features = core.results['features']

        else:

            print(f"  core.results keys: {list(core.results.keys()) if core.results else 'None'}")

        # Add red circles for each detection
        if 'features' in core.results and core.results['features'] is not None:
            features = core.results['features']

            for i, feature in enumerate(features):
                try:
                    # Extract coordinates (baryX, baryY are in meters, convert to pixels)
                    x_m = feature[1]  # baryX in meters
                    y_m = feature[2]  # baryY in meters

                    # Convert using the same dx, dy resolution as used in processing
                    dx = float(core.pixel_size) / float(core.objective_magnification)  # meters per pixel
                    dy = float(core.pixel_size) / float(core.objective_magnification)  # meters per pixel

                    # Convert from physical coordinates (meters) to pixel coordinates
                    x_pix = int(x_m / dx)
                    y_pix = int(y_m / dy)

                    # Draw red filled dot (not a circle outline) if coordinates are within image bounds
                    if 0 <= x_pix < img_array.shape[1] and 0 <= y_pix < img_array.shape[0]:
                        cv2.circle(img_array, (x_pix, y_pix), 3, (255, 0, 0), -1)  # Red filled circle, radius 3

                except Exception as e:
                    if i < 3:

                        print(f"   Feature data: {feature}")
                        import traceback
                        traceback.print_exc()

        return Image.fromarray(img_array)

    except ImportError:
        # print("OpenCV not available, cannot add detection markers")
        return image
    except Exception as e:
        # print(f"Error adding detection markers: {e}")
        return image

def add_detection_markers_to_image_2d_projection(core, image, results, projection_type):
    """Ajoute des marqueurs de détection (points rouges) sur une projection 2D selon le type de projection

    Args:
        image: Image PIL ou array numpy
        results: Résultats de détection
        projection_type: Type de projection ('XY', 'XZ', 'YZ')
    """
    try:
        import cv2

        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # Add markers for each detected particle
        if 'features' in core.results and core.results['features'] is not None:
            features = core.results['features']
            for feature in features:
                # Extract coordinates (now in meters, convert to micrometers)
                x_m = feature[1]  # baryX in meters
                y_m = feature[2]  # baryY in meters
                z_m = feature[3]  # baryZ in meters

                # Convert from meters to micrometers
                x_um = x_m * 1e6
                y_um = y_m * 1e6
                z_um = z_m * 1e6

                # Convert using the same dx, dy resolution as used in processing
                dx = float(core.pixel_size) / float(core.objective_magnification)  # meters per pixel
                dy = float(core.pixel_size) / float(core.objective_magnification)  # meters per pixel
                dz = float(core.step)  # Step size in meters

                # X and Y coordinates in pixels (lateral dimensions)
                x_pix = int(x_m / dx)
                y_pix = int(y_m / dy)

                # Z coordinate in pixels (depth dimension) - convert from meters to plane index
                z_pix = int(z_m / dz)  # Direct conversion from meters to plane index

                # Choose coordinates based on projection type
                if projection_type == 'XY':
                    coord_x, coord_y = x_pix, y_pix
                elif projection_type == 'XZ':
                    coord_x, coord_y = x_pix, z_pix
                elif projection_type == 'YZ':
                    coord_x, coord_y = y_pix, z_pix
                else:
                    coord_x, coord_y = x_pix, y_pix

                # Clamp coordinates to image boundaries to prevent IndexError
                coord_x = max(0, min(coord_x, img_array.shape[1] - 1))
                coord_y = max(0, min(coord_y, img_array.shape[0] - 1))

                # Draw marker: red filled circle (no green circle anymore)
                # Use radius=3 with filled circle (-1) for a visible but compact red dot
                cv2.circle(img_array, (coord_x, coord_y), 3, (255, 0, 0), -1)  # Red filled circle

        return Image.fromarray(img_array)

    except ImportError:
        # print("OpenCV not available, cannot add detection markers")
        return image
    except Exception as e:
        # print(f"Error adding detection markers to projection: {e}")
        return image

def get_default_display_type(core):
    """Get default display type based on parameters"""
    remove_mean = core.remove_mean
    # Check if remove_mean is enabled (regardless of mean_hologram availability)
    if remove_mean:
        return "CLEANED_HOLOGRAM"
    else:
        return "RAW_HOLOGRAM"

def get_pixel_value(core, directory, filename, display_type, plane_number, x, y):
    """Get the pixel value from the original data for the given coordinates"""
    try:
        # Helper: get value from array with bounds checking
        def _get_array_value(arr, x, y):
            if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                h, w = arr.shape[-2:]
                if 0 <= x < w and 0 <= y < h:
                    return arr[y, x] if len(arr.shape) == 2 else arr[-1, y, x]
            return None

        # RAW_HOLOGRAM
        if display_type == "RAW_HOLOGRAM":
            if core.h_raw_holo is not None:
                return _get_array_value(core.h_raw_holo, x, y)
            # Fallback to loading file
            raw_hologram = open_hologram_image(core, directory, filename)
            arr = np.array(raw_hologram.convert('L'), dtype=np.float64)
            return _get_array_value(arr, x, y)

        # CLEANED_HOLOGRAM
        elif display_type == "CLEANED_HOLOGRAM":
            if core.h_cleaned_holo is not None:
                return _get_array_value(core.h_cleaned_holo, x, y)

        # FILTERED_HOLOGRAM
        elif display_type == "FILTERED_HOLOGRAM":
            # Try CPU version first (more efficient)
            if core.h_filtered_holo is not None:
                return _get_array_value(core.h_filtered_holo, x, y)
            # Fallback to GPU version
            if core.d_filtered_holo is not None:
                try:
                    h_filtered = cp.asnumpy(core.d_filtered_holo)
                    return _get_array_value(h_filtered, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        # FFT_HOLOGRAM
        elif display_type == "FFT_HOLOGRAM":
            if core.d_fft_holo is not None:
                try:
                    h_fft = cp.asnumpy(core.d_fft_holo)
                    return _get_array_value(h_fft, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        # FFT_FILTERED_HOLOGRAM
        elif display_type == "FFT_FILTERED_HOLOGRAM":
            if core.d_fft_holo_filtered is not None:
                try:
                    h_fft_filtered = cp.asnumpy(core.d_fft_holo_filtered)
                    return _get_array_value(h_fft_filtered, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        # VOLUME_PLANE_NUMBER
        elif display_type == "VOLUME_PLANE_NUMBER":
            if core.d_volume_module is not None:
                try:
                    if plane_number < core.d_volume_module.shape[0]:
                        plane = cp.asnumpy(core.d_volume_module[plane_number, :, :])
                        return _get_array_value(plane, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        # Projections
        elif display_type in ["XY_SUM_PROJECTION", "XY_MAX_PROJECTION"]:
            if core.d_volume_module is not None:
                try:
                    if display_type == "XY_SUM_PROJECTION":
                        projection = cp.sum(core.d_volume_module, axis=0)
                    else:
                        projection = cp.max(core.d_volume_module, axis=0)
                    projection = cp.asnumpy(projection)
                    return _get_array_value(projection, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        elif display_type in ["XZ_SUM_PROJECTION", "XZ_MAX_PROJECTION"]:
            if core.d_volume_module is not None:
                try:
                    if display_type == "XZ_SUM_PROJECTION":
                        projection = cp.sum(core.d_volume_module, axis=1)
                    else:
                        projection = cp.max(core.d_volume_module, axis=1)
                    projection = cp.asnumpy(projection)
                    return _get_array_value(projection, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        elif display_type in ["YZ_SUM_PROJECTION", "YZ_MAX_PROJECTION"]:
            if core.d_volume_module is not None:
                try:
                    if display_type == "YZ_SUM_PROJECTION":
                        projection = cp.sum(core.d_volume_module, axis=2)
                    else:
                        projection = cp.max(core.d_volume_module, axis=2)
                    projection = cp.asnumpy(projection)
                    return _get_array_value(projection, x, y)
                except Exception:
                    log.debug("exception ignorée", exc_info=True)

        return "N/A"

    except Exception as e:
        return f"Error: {e}"

def extract_object_slices(core, pos_x_um, pos_y_um, pos_z_um, vox_xy, vox_z):
    """Extract 3 slice views (XY, XZ, YZ) around an object from the reconstructed volume"""
    if not core.memory_allocated or core.d_volume_module is None:
        raise ValueError("Test mode not initialized or no volume available")

    # print(f" Core: Extracting slices at ({pos_x_um:.3f}, {pos_y_um:.3f}, {pos_z_um:.3f}) µm")

    # Convert micrometers to pixel coordinates using the same formula as in add_detection_markers
    volume_shape = core.d_volume_module.shape  # (Z, Y, X)

    # Use the same conversion formula as used in processing (dx, dy)
    dx = float(core.pixel_size) / float(core.objective_magnification)  # meters per pixel
    dy = float(core.pixel_size) / float(core.objective_magnification)  # meters per pixel

    # Convert coordinates from micrometers to meters, then to pixels
    pos_x_m = pos_x_um / 1e6  # Convert µm to meters
    pos_y_m = pos_y_um / 1e6  # Convert µm to meters
    pos_z_m = pos_z_um / 1e6  # Convert µm to meters

    # Convert X,Y coordinates (camera pixel coordinates)
    pos_x_pix = int(pos_x_m / dx)
    pos_y_pix = int(pos_y_m / dy)

    # For Z coordinate, use the step parameter.
    # Z est absolu (il inclut distance_ini): retirer l'offset pour retrouver le plan.
    dz = float(core.step)  # meters
    pos_z_pix = int((pos_z_m - float(core.distance_ini)) / dz)

    # Clamp to valid range
    pos_x_pix = max(0, min(pos_x_pix, volume_shape[2] - 1))
    pos_y_pix = max(0, min(pos_y_pix, volume_shape[1] - 1))
    pos_z_pix = max(0, min(pos_z_pix, volume_shape[0] - 1))

    # print(f"📍 Corrected pixel coordinates: ({pos_x_pix}, {pos_y_pix}, {pos_z_pix})")
    # print(f"📐 Volume shape: {volume_shape}")
    # print(f" Spatial resolution: dx={dx*1e6:.3f} µm/pixel, dy={dy*1e6:.3f} µm/pixel")

    # Calculate effective pixel size for debugging
    effective_pixel_size = dx * 1e6  # Convert from meters to micrometers
    # print(f" Effective pixel size: {effective_pixel_size:.3f} µm/pixel")
    # Skip the problematic print line

    # Calculate slice boundaries with padding
    half_xy = vox_xy // 2
    half_z = vox_z // 2

    # XY slice (constant Z)
    z_center = max(0, min(pos_z_pix, volume_shape[0] - 1))
    xy_slice = _extract_slice_with_padding(core, 
        core.d_volume_module[z_center, :, :], 
        pos_y_pix, pos_x_pix, vox_xy, vox_xy
    )

    # XZ slice (constant Y) 
    y_center = max(0, min(pos_y_pix, volume_shape[1] - 1))
    xz_slice = _extract_slice_with_padding(core, 
        core.d_volume_module[:, y_center, :],
        pos_z_pix, pos_x_pix, vox_z, vox_xy
    )

    # YZ slice (constant X)
    x_center = max(0, min(pos_x_pix, volume_shape[2] - 1))
    yz_slice = _extract_slice_with_padding(core, 
        core.d_volume_module[:, :, x_center],
        pos_z_pix, pos_y_pix, vox_z, vox_xy
    )

    # print(f" Extracted slices: XY({xy_slice.shape}), XZ({xz_slice.shape}), YZ({yz_slice.shape})")

    return {
        'xy_slice': cp.asnumpy(xy_slice),
        'xz_slice': cp.asnumpy(xz_slice),
        'yz_slice': cp.asnumpy(yz_slice)
    }

def _extract_slice_with_padding(core, slice_2d, center_row, center_col, size_row, size_col):
    """Extract a sub-slice with zero padding if near boundaries"""
    half_row = size_row // 2
    half_col = size_col // 2

    # Calculate boundaries
    row_start = center_row - half_row
    row_end = center_row + half_row + (size_row % 2)
    col_start = center_col - half_col
    col_end = center_col + half_col + (size_col % 2)

    # Get original slice dimensions
    orig_rows, orig_cols = slice_2d.shape

    # Create output array filled with zeros
    output = cp.zeros((size_row, size_col), dtype=slice_2d.dtype)

    # Calculate valid regions
    valid_row_start = max(0, row_start)
    valid_row_end = min(orig_rows, row_end)
    valid_col_start = max(0, col_start)
    valid_col_end = min(orig_cols, col_end)

    # Calculate offsets in output array
    out_row_start = valid_row_start - row_start
    out_row_end = out_row_start + (valid_row_end - valid_row_start)
    out_col_start = valid_col_start - col_start
    out_col_end = out_col_start + (valid_col_end - valid_col_start)

    # Copy valid data
    output[out_row_start:out_row_end, out_col_start:out_col_end] = \
        slice_2d[valid_row_start:valid_row_end, valid_col_start:valid_col_end]

    return output
