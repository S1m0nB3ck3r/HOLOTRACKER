# -*- coding: utf-8 -*-
"""
pre_treatment.py
----------------
Computes the mean image (arithmetic or log/geometric) of a directory of images,
then subtracts or divides each image by that mean.
Cleaned images are saved to directory_out.

Parameters
----------
directory_in  : str  – Directory containing the source images.
directory_out : str  – Output directory (created if it does not exist).
image_type    : str  – Image file extension to process, e.g. "tif", "png", "bmp".
mean_type     : str  – "arithmetic" or "log" (geometric mean in log space).
clean_type    : str  – "soustr" (subtraction) or "div" (division).

Command-line usage
------------------
python pre_treatment.py --dir_in PATH_IN --dir_out PATH_OUT
                        --image_type tif --mean_type log --clean_type soustr
"""

import os
import sys
import argparse
import numpy as np

# Le calcul de la moyenne est partage avec HoloTracker Locate: une seule implementation
# pour tout le projet. code_locate/ n'est pas un paquet installable, on l'ajoute au chemin
# comme le font holo_log.py et les tests.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "code_locate"))
import traitement_holo
from PIL import Image


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def list_images(directory: str, image_type: str) -> list[str]:
    """Return a sorted list of image paths matching the given extension."""
    ext = image_type.lstrip(".").lower()
    paths = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(f".{ext}")
    ]
    if not paths:
        raise FileNotFoundError(
            f"No '.{ext}' images found in: {directory}"
        )
    return paths


def read_image_float32(path: str) -> np.ndarray:
    """Open an image and convert it to a float32 array."""
    return np.asarray(Image.open(path)).astype(np.float32)


def save_image_tiff(array: np.ndarray, path: str) -> None:
    """
    Save a numpy array as a 32-bit floating-point TIFF.
    Uses PIL Image mode 'F' (single-precision float32).
    """
    img = Image.fromarray(array.astype(np.float32), mode="F")
    img.save(path)


# ---------------------------------------------------------------------------
# Mean image computation
# ---------------------------------------------------------------------------

def compute_mean_image(
    image_paths: list[str],
    mean_type: str,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the mean image and per-pixel min/max maps over all raw images.

    mean_type="arithmetic" : classical mean      -> mean(I_i)
    mean_type="log"        : geometric mean      -> exp(mean(log(I_i)))

    progress_callback : callable(i, n), called after each image.

    Returns
    -------
    mean    : np.ndarray  - mean image (float32)
    min_map : np.ndarray  - per-pixel minimum over all raw images
    max_map : np.ndarray  - per-pixel maximum over all raw images

    These two maps allow the exact extrema of the cleaned images to be derived
    analytically without any additional pass over the data.

    Thin wrapper: the computation itself lives in code_locate/traitement_holo.py, which is
    the single mean implementation of the whole project. Three copies used to coexist (the
    Locate button, the standalone script, this module) and differed only in peripheral
    details, now arguments.
    """
    if mean_type.lower() not in ("arithmetic", "log"):
        raise ValueError(f"Invalid mean_type: '{mean_type}'. Use 'arithmetic' or 'log'.")

    print(f"Computing mean image ({mean_type}) over {len(image_paths)} images...")
    if progress_callback is None:
        n = len(image_paths)

        def progress_callback(i, total):
            if i % max(1, total // 10) == 0 or i == total:
                print(f"  {i}/{total} images processed ({100 * i / total:.0f} %)")

    return traitement_holo.calc_holo_moyen(
        image_paths, type_moyenne=mean_type, progress_callback=progress_callback)


# ---------------------------------------------------------------------------
# Image cleaning
# ---------------------------------------------------------------------------

def clean_image(img: np.ndarray, mean: np.ndarray, clean_type: str) -> np.ndarray:
    """
    Clean an image with respect to the mean image.

    clean_type="soustr" : result = img - mean
    clean_type="div"    : result = img / mean  (zero mean pixels are replaced by epsilon)
    """
    clean_type = clean_type.lower()
    if clean_type == "soustr":
        return img - mean
    elif clean_type == "div":
        # Avoid division by zero
        eps = np.finfo(np.float32).tiny
        denom = np.where(mean == 0.0, eps, mean)
        return img / denom
    else:
        raise ValueError(f"Invalid clean_type: '{clean_type}'. Use 'soustr' or 'div'.")


def compute_normalization_bounds(
    min_map: np.ndarray,
    max_map: np.ndarray,
    mean: np.ndarray,
    clean_type: str,
) -> tuple[float, float]:
    """
    Analytically determine the global min and max of the cleaned images
    from the per-pixel min_map / max_map accumulated during the mean pass.

    For each pixel (x, y):
      - subtraction : I_cleaned[x,y] = I[x,y] - mean[x,y]
          → min_cleaned[x,y] = min_map[x,y] - mean[x,y]
          → max_cleaned[x,y] = max_map[x,y] - mean[x,y]
      - division    : I_cleaned[x,y] = I[x,y] / mean[x,y]
          → min_cleaned[x,y] = min_map[x,y] / mean[x,y]
          → max_cleaned[x,y] = max_map[x,y] / mean[x,y]

    global_min / global_max are then the min/max of these 2D maps,
    which is exact without any additional pass over the data.
    """
    clean_type = clean_type.lower()
    if clean_type == "soustr":
        cleaned_min_map = min_map - mean
        cleaned_max_map = max_map - mean
    elif clean_type == "div":
        eps = np.finfo(np.float32).tiny
        denom = np.where(mean == 0.0, eps, mean)
        cleaned_min_map = min_map / denom
        cleaned_max_map = max_map / denom
    else:
        raise ValueError(f"Invalid clean_type: '{clean_type}'.")

    global_min = float(cleaned_min_map.min())
    global_max = float(cleaned_max_map.max())
    return global_min, global_max


def clean_and_normalize(
    img: np.ndarray,
    mean: np.ndarray,
    clean_type: str,
    global_min: float,
    global_max: float,
) -> np.ndarray:
    """
    Clean an image and normalize it to [0.0, 1.0].

    Normalization uses the global extrema computed analytically by
    compute_normalization_bounds, guaranteeing that 0.0 and 1.0 correspond
    exactly to the darkest and brightest values across the entire dataset.
    """
    cleaned = clean_image(img, mean, clean_type).astype(np.float64)
    if global_max > global_min:
        normalized = (cleaned - global_min) / (global_max - global_min)
    else:
        normalized = np.zeros_like(cleaned)
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    directory_in: str,
    directory_out: str,
    image_type: str,
    mean_type: str,
    clean_type: str,
) -> None:
    """
    Full pipeline:
      1. List images in directory_in.
      2. Compute the mean image according to mean_type.
      3. Save the mean image to directory_out/mean_image.tif.
      4. For each image: clean according to clean_type, normalize to [0,1] and save.
    """
    # --- Validation ---
    if not os.path.isdir(directory_in):
        raise NotADirectoryError(f"Source directory not found: {directory_in}")

    os.makedirs(directory_out, exist_ok=True)

    # --- List images ---
    image_paths = list_images(directory_in, image_type)
    print(f"{len(image_paths)} images found in: {directory_in}")

    # --- Mean image + per-pixel maps (single pass) ---
    mean_image, min_map, max_map = compute_mean_image(image_paths, mean_type)

    mean_output_path = os.path.join(directory_out, "mean_image.tif")
    save_image_tiff(mean_image, mean_output_path)
    print(f"Mean image saved: {mean_output_path}")

    # --- Analytical computation of normalization bounds ---
    global_min, global_max = compute_normalization_bounds(
        min_map, max_map, mean_image, clean_type
    )
    print(f"Normalization bounds: min={global_min:.6f}  max={global_max:.6f}")

    # --- Clean, normalize [0,1] and save ---
    print(f"\nCleaning + normalizing images (clean_type='{clean_type}')...")
    for i, path in enumerate(image_paths):
        img = read_image_float32(path)
        result = clean_and_normalize(img, mean_image, clean_type, global_min, global_max)

        # Keep original filename, force .tif extension
        basename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(directory_out, f"{basename}_cleaned.tif")
        save_image_tiff(result, out_path)

        if (i + 1) % max(1, len(image_paths) // 10) == 0 or (i + 1) == len(image_paths):
            print(f"  {i + 1}/{len(image_paths)} saved ({100 * (i + 1) / len(image_paths):.0f} %)")

    print(f"\nDone. Cleaned images saved to: {directory_out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subtract or divide the mean image from every image in a directory."
    )
    parser.add_argument("--dir_in",     required=True,  help="Source image directory.")
    parser.add_argument("--dir_out",    required=True,  help="Output directory.")
    parser.add_argument("--image_type", required=True,  help="Image extension (e.g. tif, png, bmp).")
    parser.add_argument(
        "--mean_type",
        required=True,
        choices=["arithmetic", "log"],
        help="Mean type: 'arithmetic' or 'log' (geometric).",
    )
    parser.add_argument(
        "--clean_type",
        required=True,
        choices=["soustr", "div"],
        help="Cleaning operation: 'soustr' (subtraction) or 'div' (division).",
    )

    args = parser.parse_args()

    run(
        directory_in=args.dir_in,
        directory_out=args.dir_out,
        image_type=args.image_type,
        mean_type=args.mean_type,
        clean_type=args.clean_type,
    )
