"""Processor utilities for code_link GUI.

Contains helpers to load localization CSVs, normalize columns, convert positions to micrometers,
and a thin wrapper around trackpy linking.
"""
from typing import Optional, Tuple
import os
import math
import pandas as pd

try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except Exception:
    tp = None
    TRACKPY_AVAILABLE = False


def _norm(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isalnum())


# The original code attempted to auto-detect column names. The user's CSV format is fixed and
# simple; we'll assume these exact column headers and map them directly. This keeps the loader
# minimal and easier to reason about.



def load_localisation_csv(path: str) -> pd.DataFrame:
    """Load a localisation CSV and return a dataframe with columns: frame,x,y,z (units: meters)

    - Detects columns using heuristics
    - Converts numerical columns to numeric
    - Attempts to detect units (header containing '(m)' or typical magnitudes). If values appear to be in micrometers
      the function converts them to meters. The returned dataframe uses SI units (meters) throughout.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Discard the placeholder rows that older versions of HoloTracker Locate wrote for
    # holograms where no object was detected: 'OBJECT NUMBER' == 0 with x=y=z=0.
    # Kept as a fallback so CSV files produced before the fix stay usable: without it
    # trackpy links those zeros together into a motionless trajectory at the origin.
    if 'OBJECT NUMBER' in df.columns:
        object_number = pd.to_numeric(df['OBJECT NUMBER'], errors='coerce')
        n_before = len(df)
        df = df[object_number != 0]
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"load_localisation_csv: dropped {n_dropped} empty-hologram placeholder row(s)")

    # Assume exact columns as provided by the user (Z present):
    # 'HOLOGRAM NUMBER','OBJECT NUMBER','X POSITION (m)','Y POSITION (m)','Z POSITION (m)','NUMBER OF VOXEL'
    # Map these to a minimal dataframe with columns: frame,x,y,z (units: meters)
    cols = ['HOLOGRAM NUMBER', 'X POSITION (m)', 'Y POSITION (m)', 'Z POSITION (m)']
    df2 = df[cols].copy()
    df2.columns = ['frame', 'x', 'y', 'z']

    df2['frame'] = pd.to_numeric(df2['frame'], errors='coerce').fillna(0).astype(int)
    df2['x'] = pd.to_numeric(df2['x'], errors='coerce')
    df2['y'] = pd.to_numeric(df2['y'], errors='coerce')
    df2['z'] = pd.to_numeric(df2['z'], errors='coerce')

    return df2


from typing import Union, Tuple

# Journal: voir holo_log.py. Les exceptions ignorées y laissent une trace,
# avec fichier, fonction et ligne, sans interrompre l'application.
import logging
log = logging.getLogger(__name__)


def link_df(df: pd.DataFrame, search_range: Union[float, Tuple[float, float, float]], memory: int, minlength: int = 0) -> pd.DataFrame:
    """Thin wrapper around trackpy.link_df.

    df must contain columns 'frame','x','y' (units: meters).
    search_range: search radius in same units as x,y (meters)
    memory: trackpy memory
    """
    if not TRACKPY_AVAILABLE:
        raise RuntimeError('trackpy not available in environment')

    # trackpy expects frame and x,y; preserve z if present
    cols = ['frame', 'x', 'y', 'z']

    # select only expected columns (if z missing, ensure it's present with zeros)
    cols_present = ['frame', 'x', 'y']
    if 'z' in df.columns:
        cols_present.append('z')
    else:
        # create a z column of zeros for trackpy APIs that expect it
        df = df.copy()
        df['z'] = 0.0
        cols_present.append('z')

    df_tp = df[cols_present].copy()

    # Carry the original CSV row identity along as a column. trackpy's filter_stubs
    # rebuilds the index (reset_index then set_index('frame')), so the DataFrame index
    # cannot be relied upon downstream; '_row' survives every step and is restored as
    # the index just before returning, so the caller can map each linked point back to
    # the exact line it came from.
    df_tp['_row'] = df.index

    # If search_range is a tuple (per-axis), use the older tp.link API directly
    try:
        if isinstance(search_range, tuple) or isinstance(search_range, list):
            tuple_range = tuple(float(v) for v in search_range)
            trajectories = tp.link(f=df_tp, search_range=tuple_range, memory=memory, t_column='frame', pos_columns=['x', 'y', 'z'] if 'z' in df_tp.columns else ['x', 'y'])
        else:
            # First attempt: use link_df (vectorized) with scalar radius
            trajectories = tp.link_df(df_tp, float(search_range), memory=memory)
    except Exception as e:
        # If trackpy reports a subnetwork error or other issues, attempt fallback
        msg = str(e).lower()
        if 'subnetwork contains' in msg or 'subnetwork' in msg:
            # Try fallback using tuple search_range (same value on all axes)
            tuple_range = (float(search_range), float(search_range), float(search_range))
            trajectories = tp.link(f=df_tp, search_range=tuple_range, memory=memory, t_column='frame', pos_columns=['x', 'y', 'z'] if 'z' in df_tp.columns else ['x', 'y'])
        else:
            # Re-raise unexpected exceptions so GUI can display them
            raise

    # Min length: drop the trajectories shorter than `minlength` points.
    #
    # This step used to re-run tp.link on the filtered set. That was wrong twice over:
    #   - re-linking reassigns the particle ids and can regroup the remaining points,
    #     so `minlength` silently changed the tracking result instead of only filtering it;
    #   - the whole block sat in a bare `except Exception: pass`. When `minlength` was
    #     larger than the number of frames in the movie, filter_stubs legitimately
    #     removed everything, the re-link then raised IndexError on the empty frame, and
    #     the exception was swallowed — leaving the UNFILTERED trajectories in place.
    #     The parameter then looked like it had no effect at all.
    # Filtering alone, with errors left visible, is both correct and predictable.
    if minlength and minlength > 0:
        n_before = int(trajectories['particle'].nunique())
        trajectories = tp.filtering.filter_stubs(tracks=trajectories, threshold=minlength)
        n_after = int(trajectories['particle'].nunique()) if len(trajectories) else 0
        print(f"link_df: min length = {minlength} -> {n_after}/{n_before} trajectories kept")

    # Restore the original CSV row identity as the index (see '_row' above).
    if '_row' in trajectories.columns:
        trajectories = trajectories.set_index('_row')
        trajectories.index.name = None

    # Normalize output columns similar to wrapper
    try:
        # Ensure particle column exists
        if 'particle' not in trajectories.columns:
            # trackpy older API may use 'particle' or 'particle' will be added; if missing, add a placeholder
            trajectories = trajectories.reset_index().rename(columns={'index': 'particle'})
        # Ensure nb_pix column exists (wrapper expects it)
        if 'nb_pix' not in trajectories.columns:
            trajectories['nb_pix'] = 0
        # Standardize column order
        # Some trackpy versions return different column sets; try to set to expected columns
        cols_out = ['frame', 'x', 'y']
        if 'z' in trajectories.columns:
            cols_out.append('z')
        cols_out += ['nb_pix', 'particle']
        # Reindex to include expected columns (missing columns will be filled with NaN)
        trajectories = trajectories.reindex(columns=cols_out)
        trajectories = trajectories.sort_values(by=['particle', 'frame'])
    except Exception:
        # best-effort normalization; ignore failures
        log.debug("exception ignorée", exc_info=True)

    return trajectories


def build_trajectory_table(orig: pd.DataFrame, traj: pd.DataFrame):
    """Construit la table de sortie de HoloTracker Link.

    C'est le CSV de localisation d'origine, dans lequel la colonne 'OBJECT NUMBER' est
    REMPLACEE par 'TRAJECTORY NUMBER': l'identifiant de trajectoire attribue par trackpy,
    ou -1 pour les detections qui n'ont ete retenues dans aucune trajectoire.

    Parameters
    ----------
    orig : le CSV de localisation lu tel quel (colonnes de HoloTracker Locate)
    traj : la sortie de link_df, indexee par le numero de ligne d'origine

    Returns
    -------
    (table de sortie, nombre de lignes auxquelles une trajectoire a ete attribuee)

    Cette fonction etait auparavant enfouie dans la methode _save de l'interface, melee a
    la lecture des widgets et a l'ecriture du fichier. Isolee ici, elle est testable: c'est
    elle qui decide quelle trajectoire va sur quelle ligne, l'endroit ou une erreur passe
    le plus facilement inapercue.
    """
    if 'particle' not in traj.columns:
        raise ValueError("Les trajectoires ne contiennent pas de colonne 'particle'")

    out = orig.copy()
    nrows = len(out)
    numeros = [-1] * nrows

    # Chemin normal: link_df restitue le numero de ligne d'origine comme index (colonne
    # '_row' portee tout au long du traitement). L'appariement est alors exact.
    for idx, pid in zip(traj.index, traj['particle']):
        try:
            pos = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < nrows:
            try:
                numeros[pos] = int(pid)
            except (TypeError, ValueError):
                numeros[pos] = -1

    out['TRAJECTORY NUMBER'] = numeros
    attribuees = sum(1 for v in numeros if v != -1)

    # Filet de securite: si l'index ne correspondait pas aux lignes d'origine, on reapparie
    # par la cle (image, x, y, z) arrondie. Ne devrait plus se declencher depuis que
    # link_df conserve l'identite des lignes; conserve pour les fichiers anciens.
    if attribuees < max(10, int(0.5 * len(traj))):
        secours, utilisees = _apparier_par_coordonnees(orig, traj, nrows)
        if utilisees > attribuees:
            log.warning("appariement par index insuffisant (%d/%d), repli sur les "
                        "coordonnees (%d appariees)", attribuees, len(traj), utilisees)
            out['TRAJECTORY NUMBER'] = secours
            attribuees = utilisees

    # 'OBJECT NUMBER' est remplacee, pas conservee: son role est repris par la colonne
    # 'TRAJECTORY NUMBER'.
    if 'OBJECT NUMBER' in out.columns:
        out = out.drop(columns=['OBJECT NUMBER'])

    return out, attribuees


def _apparier_par_coordonnees(orig, traj, nrows):
    """Appariement de secours par la cle (image, x, y, z) arrondie au nanometre."""
    def colonne(df, candidats):
        for c in candidats:
            if c in df.columns:
                return c
        return None

    col_image = colonne(orig, ['HOLOGRAM NUMBER', 'frame', 'Frame', 'HOLOGRAM_NUMBER'])
    col_x = colonne(orig, ['X POSITION (m)', 'x', 'X_POSITION_(m)', 'X'])
    col_y = colonne(orig, ['Y POSITION (m)', 'y', 'Y_POSITION_(m)', 'Y'])
    col_z = colonne(orig, ['Z POSITION (m)', 'z', 'Z_POSITION_(m)', 'Z'])
    if not (col_image and col_x and col_y):
        return [-1] * nrows, 0

    positions = {}
    for pos, row in orig.iterrows():
        try:
            image = int(row[col_image])
        except (TypeError, ValueError):
            continue
        cle = (image,
               round(float(row[col_x]), 9),
               round(float(row[col_y]), 9),
               round(float(row[col_z]), 9) if col_z else 0.0)
        positions.setdefault(cle, []).append(pos)

    numeros = [-1] * nrows
    utilisees = 0
    for _, r in traj.iterrows():
        try:
            image = int(r['frame'])
        except (TypeError, KeyError, ValueError):
            continue
        cle = (image, round(float(r.get('x', 0.0)), 9),
               round(float(r.get('y', 0.0)), 9), round(float(r.get('z', 0.0)), 9))
        libres = positions.get(cle)
        if libres:
            numeros[libres.pop(0)] = int(r['particle'])
            utilisees += 1
    return numeros, utilisees
