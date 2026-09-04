import os
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from scipy import fft
from traitement_holo import *
try:
    import cupy as cp
    from traitement_holo import calc_holo_moyen, lister_images, read_image, projection_bool
    import propagation as propag
    import focus 
    from focus import Focus_type
    from CCL3D import CCL3D, calc_threshold, CCA_CUDA_float, CCL_filter, type_threshold, dobjet
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    CUPY_AVAILABLE = True
except ImportError as e:
    # Warning: CuPy or other dependencies not available
    cp = None
    CUPY_AVAILABLE = False

# Journal: voir holo_log.py. Les exceptions ignorées y laissent une trace,
# avec fichier, fonction et ligne, sans interrompre l'application.
import logging
log = logging.getLogger(__name__)

# Maximum number of objects that can be detected and displayed
MAX_OBJECT_DETECTION = 500


class HoloTrackerCore:
    def __init__(self):
        self.results = {}
        
        # Memory allocation status
        self.memory_allocated = False
        self.test_mode_params = {}  # To track parameters that require reallocation
        
        # Mode tracking
        self.mode = 'IDLE'  # Possible values: 'IDLE', 'TEST', 'BATCH'
        self.batch_first_hologram_done = False
        
        # Allocated GPU variables (as in test_HoloTracker_locate.py)
        self.h_raw_holo = None
        self.h_mean_holo =None
        self.h_cleaned_holo = None
        self.d_holo = None
        self.h_filtered_holo = None
        self.d_filtered_holo = None
        self.d_fft_holo = None
        self.d_fft_holo_filtered = None
        self.d_fft_holo_propag = None
        self.d_holo_propag = None
        self.d_KERNEL = None
        self.d_FFT_KERNEL = None
        self.d_volume_module = None
        self.d_bin_volume_focus = None
        self.d_mean_holo = None
        
        # Variables for results
        self.current_features = None
        
        # Hologram and system parameters
        self.mean_hologram_image_path = ""
        self.holograms_directory = ""
        self.image_type = "BMP"
        self.wavelength = 660e-9
        self.medium_optical_index = 1.33
        self.objective_magnification = 40.0
        self.pixel_size = 5.5e-6
        self.holo_size_x = 1024
        self.holo_size_y = 1024
        self.distance_ini = 20e-6
        self.step = 0.5e-6
        self.number_of_planes = 200
        
        # Cleaning parameters
        self.remove_mean = True
        self.cleaning_type = "subtraction"
        
        # Filtering parameters
        self.high_pass = 15
        self.low_pass = 125
        
        # Focus parameters
        self.focus_type = "TENEGRAD"
        self.sum_size = 15
        
        # Thresholding and detection parameters
        self.nb_StdVar_threshold = 14.0
        self.connectivity = 26
        self.min_voxel = 0
        self.max_voxel = 0
        self.batch_threshold = "compute on 1st hologram"
        
        # Display parameters
        self.additional_display = "Centroid positions"

    # Type attendu de chaque paramètre. L'interface graphique lit tous ses champs sous
    # forme de texte; sans conversion ici, le cœur stockait des chaînes et chaque point
    # d'utilisation devait faire son propre float()/int(). Effet de bord notable:
    # check_reallocation_needed comparait "1024" à 1024 et concluait toujours à un
    # changement. La conversion est faite une seule fois, à l'entrée.
    PARAMETER_TYPES = {
        # optique et géométrie
        "wavelength": float,
        "medium_optical_index": float,
        "objective_magnification": float,
        "pixel_size": float,
        "distance_ini": float,
        "step": float,
        "holo_size_x": int,
        "holo_size_y": int,
        "number_of_planes": int,
        # filtrage et focus
        "high_pass": int,
        "low_pass": int,
        "sum_size": int,
        "remove_mean": bool,
        # seuillage et détection
        "nb_StdVar_threshold": float,
        "connectivity": int,
        "min_voxel": int,
        "max_voxel": int,
        # chaînes de caractères (aucune conversion)
        "mean_hologram_image_path": str,
        "holograms_directory": str,
        "image_type": str,
        "focus_type": str,
        "cleaning_type": str,
        "batch_threshold": str,
        "additional_display": str,
    }

    @classmethod
    def _coerce_parameter(cls, name, value):
        """Convertit une valeur au type attendu du paramètre.

        Lève ValueError avec un message nommant le paramètre fautif: une case vidée dans
        l'interface doit produire un message lisible dans la barre d'état, pas une
        exception opaque au milieu du pipeline.
        """
        expected = cls.PARAMETER_TYPES.get(name)
        if expected is None or isinstance(value, expected) and expected is not bool:
            return value

        if expected is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "oui")
            return bool(value)

        if expected is str:
            return str(value)

        text = str(value).strip()
        if text == "":
            raise ValueError(f"Parameter '{name}' is empty")
        try:
            if expected is int:
                # int(float(...)) accepte "1024" comme "1024.0"
                return int(float(text))
            return float(text)
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{name}': cannot read '{value}' as {expected.__name__}")

    def set_parameters(self, **kwargs):
        """Set multiple parameters at once, converted to their expected type.

        L'affectation est atomique: si une valeur est illisible, aucune n'est appliquée
        et l'état courant du cœur reste cohérent.
        """
        converted = {}
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise AttributeError(f"Parameter '{name}' does not exist")
            converted[name] = self._coerce_parameter(name, value)

        for name, value in converted.items():
            setattr(self, name, value)
        return f"Updated {len(converted)} parameters"

    def get_parameters_dict(self):
        """Get all parameters as a dictionary"""
        param_attrs = [attr for attr in dir(self) if not attr.startswith('_') and 
                      not callable(getattr(self, attr)) and 
                      not attr.isupper() and
                      attr not in ['results', 'memory_allocated', 'test_mode_params',
                                  'h_raw_holo', 'h_mean_holo', 'h_cleaned_holo', 'd_holo',
                                  'h_filtered_holo', 'd_filtered_holo', 'd_fft_holo', 
                                  'd_volume', 'd_focus', 'd_threshold', 'd_CCL', 'd_filtered_objects', 'd_slices']]
        return {attr: getattr(self, attr) for attr in param_attrs}

    def load_mean_hologram(self):
        """Load mean hologram from TIF or NPY file"""
        if not self.mean_hologram_image_path:
            raise ValueError(
                "Aucun hologramme moyen sélectionné: le choisir dans l'onglet PATH, "
                "ou décocher 'Remove mean Hologram'")
        if self.mean_hologram_image_path.lower().endswith('.npy'):
            # Legacy NPY format
            self.h_mean_holo = np.load(self.mean_hologram_image_path)
        else:
            # New TIF format
            mean_pil = Image.open(self.mean_hologram_image_path)
            self.h_mean_holo = np.array(mean_pil)

    def print_parameters(self):
        """Print all parameters, one parameter per line"""
        print("=== HOLOTRACKER PARAMETERS ===")
        # Print all parameter attributes (skip internal variables)
        param_attrs = [attr for attr in dir(self) if not attr.startswith('_') and 
                      not callable(getattr(self, attr)) and 
                      not attr.isupper() and
                      attr not in ['results', 'memory_allocated', 'test_mode_params',
                                  'h_raw_holo', 'h_mean_holo', 'h_cleaned_holo', 'd_holo',
                                  'h_filtered_holo', 'd_filtered_holo', 'd_fft_holo', 
                                  'd_fft_holo_filtered', 'd_fft_holo_propag', 'd_holo_propag',
                                  'd_KERNEL', 'd_FFT_KERNEL', 'd_volume_module', 'd_bin_volume_focus',
                                  'd_mean_holo', 'current_features', 'threshold']]
        
        for attr in sorted(param_attrs):
            value = getattr(self, attr)
            print(f"{attr}: {value}")
        print("==============================")


    def set_parameter(self, name, value):
        setattr(self, name, self._coerce_parameter(name, value))
        return f"{name} updated"
    
    def get_parameter(self, name, default=None):
        """Get a parameter value"""
        return getattr(self, name, default)


    def compute_mean_hologram(self, directory, image_type, progress_callback=None,
                              mean_type="arithmetic"):
        """Calcule l'hologramme moyen d'un répertoire et l'enregistre.

        Enveloppe autour de traitement_holo.calc_holo_moyen, qui fait le calcul pour tout
        le projet. Ne reste ici que ce qui est propre au bouton de l'interface: où écrire
        les fichiers et comment les nommer.

        Renvoie le chemin du .tif, qui est le fichier destiné aux calculs. Un aperçu .bmp
        est écrit à côté, pour l'œil seulement.
        """
        ext_map = {"BMP": "bmp", "TIF": "tif", "JPG": "jpg", "PNG": "png"}
        if not os.path.isdir(directory):
            raise ValueError("Invalid directory")

        chemins = lister_images(directory, ext_map[image_type])
        if not chemins:
            raise ValueError("No images found")

        # "logarithmic" est le nom historique de la moyenne géométrique, conservé ici
        # parce que c'est la valeur choisie dans l'interface.
        nom = "mean_arith" if mean_type == "arithmetic" else "mean_log"
        mean_dir = os.path.join(directory, "mean")
        chemin_tif = os.path.join(mean_dir, nom + ".tif")
        chemin_bmp = os.path.join(mean_dir, nom + ".bmp")

        calc_holo_moyen(chemins, type_moyenne=mean_type,
                        fichiers_sortie=(chemin_tif, chemin_bmp),
                        progress_callback=progress_callback)
        return chemin_tif

    def enter_test_mode(self):
        """Entre en mode test"""
        self.mode = 'TEST'
        return "Test mode activated"

    def exit_test_mode(self):
        """Sort du mode test"""
        self.mode = 'IDLE'
        self.cleanup_test_mode()
        return "Test mode deactivated"
    
    def enter_batch_mode(self):
        """Entre en mode batch"""
        self.mode = 'BATCH'
        self.batch_first_hologram_done = False
        return "Batch mode activated"
    
    def exit_batch_mode(self):
        """Sort du mode batch"""
        self.mode = 'IDLE'
        self.batch_first_hologram_done = False
        return "Batch mode deactivated"
    
    def allocate(self):
        """Allocation mémoire pour le traitement des hologrammes"""
        import cupy as cp
        
        # Basic parameters
        cam_nb_pix_X = int(self.holo_size_x)
        cam_nb_pix_Y = int(self.holo_size_y)
        nb_plane = int(self.number_of_planes)
        
        # Array allocation according to new architecture
        self.h_raw_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.h_mean_holo = None
        self.h_cleaned_holo = None
        self.h_filtered_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.d_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_filtered_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_fft_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_fft_holo_filtered = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_fft_holo_propag = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_holo_propag = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_FFT_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_volume_module = cp.zeros(shape=(nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_bin_volume_focus = cp.zeros(shape=(nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.dtype(bool))
        
        # Charger l'hologramme moyen en GPU si disponible
        if self.h_mean_holo is not None:
            self.d_mean_holo = cp.asarray(self.h_mean_holo)
        else:
            self.d_mean_holo = None
    
        self.memory_allocated = True
        
        # Print parameters for debugging
        self.print_parameters()

    def check_reallocation_needed(self, new_params):
        """Check if GPU memory reallocation is needed based on parameter changes"""
        if not self.memory_allocated:
            return True
            
        # Parameters that require reallocation
        size_changing_params = ["holo_size_x", "holo_size_y", "number_of_planes"]
        
        for param in size_changing_params:
            if param in new_params:
                old_value = getattr(self, param)
                # Convertir avant de comparer: la valeur entrante vient de l'interface
                # sous forme de texte, et "1024" != 1024 concluait toujours au changement.
                new_value = self._coerce_parameter(param, new_params[param])
                if old_value != new_value:
                    return True
        return False

    def update_parameters_and_reallocate_if_needed(self, new_params):
        """Update parameters and reallocate GPU memory if needed"""
        if self.check_reallocation_needed(new_params):
            # Cleanup existing allocation
            if self.memory_allocated:
                self.cleanup_test_mode()
            
            # Update parameters
            self.set_parameters(**new_params)
            
            # Reallocate with new parameters
            result = self.allocate()
            return result
        else:
            # Just update parameters, no reallocation needed
            self.set_parameters(**new_params)
            return "Parameters updated without reallocation"

    def process_hologram_complete_pipeline(self, directory, filename):
        """
        Complete hologram processing pipeline following test_HoloTracker_locate.py
        Pipeline: Load -> Remove Mean -> Volume Propagation -> Focus -> CCL3D -> Label Analysis
        """

        # Conditions d'exécution: on lève, on ne renvoie pas une chaîne. Le message
        # remonte ainsi jusqu'à la barre d'état au lieu de passer pour un succès.
        if not CUPY_AVAILABLE:
            self.results = {
                'number_of_objects': 0,
                'features': np.array([]),
                'processing_times': {'total_processing': 0.0},
            }
            raise RuntimeError('CuPy not available for GPU processing')

        if not self.memory_allocated:
            self.results = {
                'number_of_objects': 0,
                'features': np.array([]),
                'processing_times': {'total_processing': 0.0},
            }
            raise RuntimeError('GPU memory not allocated (allocate must be called before processing)')
        
        # Clear any previous results/errors at the start of each pipeline run
        # This ensures we start with a clean state
        self.results = {}
            
        # Print parameters for debugging
        self.print_parameters()
            
        try:
            import time
            
            # 1. Load raw hologram and preprocessing
            start_processing = time.perf_counter()
            filepath = os.path.join(directory, filename)
            cam_nb_pix_X = int(self.holo_size_x)
            cam_nb_pix_Y = int(self.holo_size_y)
            
            # Load raw hologram into h_raw_holo
            self.h_raw_holo[:] = read_image(filepath, cam_nb_pix_X, cam_nb_pix_Y)
            
            # 2. Clean hologram (remove mean according to parameter and type)
            remove_mean = self.remove_mean
            if remove_mean:
                cleaning_type = self.cleaning_type
                # Load mean hologram if needed
                if self.h_mean_holo is None:
                    self.load_mean_hologram()
                
                # Debug info about mean hologram
                if self.h_mean_holo is not None:
                    pass

                # Apply cleaning based on type
                if cleaning_type == "subtraction":
                    
                    self.h_cleaned_holo = self.h_raw_holo - self.h_mean_holo
                    self.h_cleaned_holo = self.h_cleaned_holo - self.h_cleaned_holo.min()  # Ensure non-negative

                else:  # division (default)
                    
                    self.h_cleaned_holo = self.h_raw_holo.astype(np.float64) / (self.h_mean_holo + 1e-10)
                    self.h_cleaned_holo = np.power(self.h_cleaned_holo, 0.8).astype(np.float32)  # Limit extreme values

            else:
                self.h_cleaned_holo = self.h_raw_holo.copy()
            
            # Transfer cleaned hologram to GPU - reuse pre-allocated array
            self.d_holo[:] = cp.asarray(self.h_cleaned_holo.astype(cp.complex64))
            
            t1 = time.perf_counter()
            t_preprocess = t1 - start_processing
            
            # 3. Spectral filtering before propagation
            
            # Get filtering parameters from UI
            f_pix_min = int(self.high_pass)  # high_pass = f_pix_min
            f_pix_max = int(self.low_pass)  # low_pass = f_pix_max
                    
            # 4. Volume propagation by angular spectrum method (without filtering since already done)
            
            # Get parameters for propagation
            wavelength = float(self.wavelength)
            medium_optical_index = float(self.medium_optical_index)
            medium_wavelength = wavelength / medium_optical_index
            objective_magnification = float(self.objective_magnification)
            pixel_size = float(self.pixel_size)
            nb_plane = int(self.number_of_planes)
            dx = pixel_size / objective_magnification # in meters
            dy = pixel_size / objective_magnification # in meters
            dz = float(self.step)  # in meters
            distance_ini = float(self.distance_ini)

            propag_start = time.perf_counter()

            propag.volume_propag_angular_spectrum_to_module(
                self.d_holo, self.d_fft_holo, self.d_fft_holo_filtered,self.d_KERNEL, 
                self.d_filtered_holo, self.d_fft_holo_propag, self.d_holo_propag, 
                self.d_volume_module, medium_wavelength, objective_magnification, 
                pixel_size, cam_nb_pix_X, cam_nb_pix_Y, distance_ini, dz, nb_plane, f_pix_min, f_pix_max)
            
          
            # Copy filtered hologram to CPU for display purposes
            if self.d_filtered_holo is not None:
                self.h_filtered_holo[:] = cp.asnumpy(self.d_filtered_holo)
                   
            t2 = time.perf_counter()
            t_propag = t2 - propag_start
            
            # 4. Focus on the volume (INPLACE)
            focus_start = time.perf_counter()
            sum_size = int(self.sum_size)
            
            # Get focus type from parameters
            focus_type_str = self.focus_type
            focus_type_map = {
                "SUM_OF_INTENSITY": Focus_type.SUM_OF_INTENSITY,
                "SUM_OF_LAPLACIAN": Focus_type.SUM_OF_LAPLACIAN,
                "SUM_OF_VARIANCE": Focus_type.SUM_OF_VARIANCE,
                "TENEGRAD": Focus_type.TENEGRAD,
                "SUM_OF_GRADIENT": Focus_type.SUM_OF_GRADIENT,
                "MEAN_ARITH_ALL": Focus_type.MEAN_ARITH_ALL,
                "MEAN_GEO_ALL": Focus_type.MEAN_GEO_ALL
            }
            focus_type_enum = focus_type_map[focus_type_str]
            
            # print(f" Core: Applying focus type: {focus_type_str} (enum: {focus_type_enum})")
            focus.focus(self.d_volume_module, self.d_volume_module, sum_size, focus_type_enum)
            
            t3 = time.perf_counter()
            t_focus = t3 - focus_start
            
            # 5. Compute threshold and CCL3D
            ccl_start = time.perf_counter()
            nb_StdVar_threshold = float(self.nb_StdVar_threshold)
            n_connectivity = int(self.connectivity)
            
            # Determine if threshold needs to be recalculated
            if self.mode == 'TEST':
                # TEST mode: always recalculate threshold
                need_recalc = True
            elif self.mode == 'BATCH':
                # BATCH mode: honour the 'batch_threshold' parameter set in the UI
                #   "compute on each hologram" -> recalculate for every hologram
                #   "compute on 1st hologram"  -> compute once, reuse for the whole batch
                recompute_each = str(self.batch_threshold).strip().lower() == "compute on each hologram"
                need_recalc = recompute_each or not self.batch_first_hologram_done
            else:
                # IDLE mode: never recalculate (not used in practice)
                need_recalc = False
            
            if need_recalc:
                self.threshold = calc_threshold(self.d_volume_module, nb_StdVar_threshold)
                # Mark first hologram as done in batch mode
                if self.mode == 'BATCH':
                    self.batch_first_hologram_done = True

            # CCL3D
            d_labels, number_of_labels = CCL3D(
                self.d_bin_volume_focus, self.d_volume_module, 
                type_threshold.THRESHOLD, self.threshold, n_connectivity
            )

            t4 = time.perf_counter()
            t_ccl = t4 - ccl_start
            
            # 6. Label analysis (centroid computation)
            cca_start = time.perf_counter()
            if number_of_labels > 0:
                features = np.ndarray(shape=(number_of_labels,), dtype=dobjet)

                features = CCA_CUDA_float(
                    d_labels, self.d_volume_module, number_of_labels, 
                    1, cam_nb_pix_X, cam_nb_pix_Y, nb_plane, dx, dy, dz,
                    z_offset=distance_ini
                )

                # Filtrage par nombre de voxels
                min_voxel = int(self.min_voxel)
                max_voxel = int(self.max_voxel)

                if min_voxel != 0 or max_voxel != 0:
                    features = CCL_filter(features, min_voxel, max_voxel)

            else:
                features = np.array([])

            t5 = time.perf_counter()
            t_cca = t5 - cca_start
            t_total = t5 - start_processing
            
            # TIME DISPLAY LIKE test_HoloTracker_locate.py
            # print(f'number of objects located: {number_of_labels}')
            # print(f't preprocess: {t_preprocess:.6f}')
            # print(f't propagation: {t_propag:.6f}')
            # print(f't focus: {t_focus:.6f}') 
            # print(f't ccl: {t_ccl:.6f}')
            # print(f't cca: {t_cca:.6f}')
            # print(f'total iteration time: {t_total:.6f}')
            # print(f'---')
            
            # Check if number of objects exceeds maximum
            if number_of_labels > MAX_OBJECT_DETECTION:
                # Store error result - clear any previous results first
                self.results = {
                    'number_of_objects': number_of_labels,
                    'features': np.array([]),  # Empty features array - objects won't be displayed
                    'processing_times': {
                        'preprocessing': t_preprocess,
                        'propagation': t_propag,
                        'focus': t_focus,
                        'ccl': t_ccl,
                        'cca': t_cca,
                        'total_processing': t_total
                    },
                    'error': f'MAX OBJECTS DETECTION ({MAX_OBJECT_DETECTION}). Please increase threshold value'
                }
                return f"Error: Too many objects detected ({number_of_labels} > {MAX_OBJECT_DETECTION})"
            
            # Store results (normal case) - clear any previous results first to remove old 'error' key
            self.results = {
                'number_of_objects': number_of_labels,
                'features': features,
                'processing_times': {
                    'preprocessing': t_preprocess,
                    'propagation': t_propag,
                    'focus': t_focus,
                    'ccl': t_ccl,
                    'cca': t_cca,
                    'total_processing': t_total
                }
                # Note: 'error' key is explicitly NOT included here - this ensures old errors are cleared
            }

            if number_of_labels > 0:
                return f"Processing completed: {number_of_labels} objects found"
            else:
                return "Processing completed: No objects found"
                
        except Exception:
            # Ne PAS avaler l'exception. Auparavant elle était transformée en simple chaîne
            # de retour, que l'appelant ne lisait pas: le résultat était alors marqué comme
            # un succès sans détection, et un lot entier pouvait échouer sans le moindre
            # signal. CoreCommunicator._process_command construit un Result(success=False)
            # à partir de l'exception, que le contrôleur affiche.
            self.results = {
                'number_of_objects': 0,
                'features': np.array([]),
                'processing_times': {'total_processing': 0.0},
            }
            raise

    def get_3d_results_data(self):
        """Get 3D results data for display in UI tab (no pop-up)"""
        try:
            # Initialize result data structure
            result_data = {
                'localizations': [],
                'particle_sizes': [],
                'count': 0
            }

            # Add processing times only if available to avoid KeyError
            if self.results and isinstance(self.results, dict) and 'processing_times' in self.results:
                result_data['processing_times'] = self.results['processing_times']
            
            # Check if there's an error (e.g., MAX_OBJECT_DETECTION exceeded)
            if self.results and isinstance(self.results, dict) and 'error' in self.results:
                result_data['error'] = self.results['error']
                # Still include the count even if there's an error
                if 'number_of_objects' in self.results:
                    result_data['count'] = self.results['number_of_objects']
                return result_data
            
            # Add detection results if available
            if self.results and 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                if len(features) > 0:
                    # Extract coordinates from features (following test_HoloTracker_locate.py format)
                    positions = pd.DataFrame(features, columns=['i_image', 'baryX', 'baryY', 'baryZ', 'nb_pix'])
                    
                    # Return localizations in the format expected by UI (convert from meters to micrometers)
                    localizations = []
                    for _, row in positions.iterrows():
                        # Convert from meters to micrometers for UI display
                        x_um = row['baryX'] * 1e6
                        y_um = row['baryY'] * 1e6  
                        z_um = row['baryZ'] * 1e6
                        localizations.append((x_um, y_um, z_um))
                    
                    result_data.update({
                        'localizations': localizations,
                        'particle_sizes': positions['nb_pix'].tolist(),
                        'count': len(features)
                    })
                
            return result_data
            
        except Exception as e:
            print(f"Error getting 3D results data: {e}")
            # Return basic structure with timing if available (guard self.results)
            basic_data = {'localizations': [], 'particle_sizes': [], 'count': 0}
            if self.results and isinstance(self.results, dict) and 'processing_times' in self.results:
                basic_data['processing_times'] = self.results['processing_times']
            return basic_data

    def show_3d_results(self):
        """Legacy method - now returns data for UI display instead of pop-up"""
        data = self.get_3d_results_data()
        if data is None:
            return "No results to display"
        return f"3D data ready with {data['count']} particles"


        
    def cleanup_test_mode(self):
        """Nettoie la mémoire allouée pour le mode test"""
        # Free GPU and CPU memory
        gpu_vars = ['d_holo', 'd_filtered_holo', 'd_fft_holo', 'd_fft_holo_filtered', 'd_fft_holo_propag', 'd_holo_propag', 
                   'd_KERNEL', 'd_FFT_KERNEL', 'd_volume_module', 'd_bin_volume_focus', 'd_mean_holo']
        
        cpu_vars = ['h_raw_holo', 'h_cleaned_holo', 'h_mean_holo', 'h_filtered_holo']
        
        # Nettoyer les variables GPU
        for var_name in gpu_vars:
            var = getattr(self, var_name)
            if var is not None:
                del var
            setattr(self, var_name, None)
        
        # Nettoyer les variables CPU
        for var_name in cpu_vars:
            var = getattr(self, var_name)
            if var is not None:
                del var
            setattr(self, var_name, None)
        
        self.memory_allocated = False
        


    def analyze_focus_at_position(self, x_pos, y_pos, focus_type_str, sum_size):
        """Analyzes focus function at a given position through all Z layers
        
        Args:
            x_pos: X position of the pixel to analyze
            y_pos: Y position of the pixel to analyze
            focus_type_str: Focus type ("TENEGRAD", "SUM_OF_INTENSITY", etc.)
            sum_size: Size of the summation window
            
        Returns:
            List of focus values for each Z plane or None in case of error
        """
        try:
            # Check that we have a propagated volume
            if self.d_volume_module is None:

                return None
                
            # Check coordinates
            if (x_pos < 0 or x_pos >= self.d_volume_module.shape[2] or
                y_pos < 0 or y_pos >= self.d_volume_module.shape[1]):

                return None
                
            # Focus type mapping
            focus_type_map = {
                "SUM_OF_INTENSITY": Focus_type.SUM_OF_INTENSITY,
                "SUM_OF_LAPLACIAN": Focus_type.SUM_OF_LAPLACIAN,
                "SUM_OF_VARIANCE": Focus_type.SUM_OF_VARIANCE,
                "TENEGRAD": Focus_type.TENEGRAD,
                "SUM_OF_GRADIENT": Focus_type.SUM_OF_GRADIENT,
                "MEAN_ARITH_ALL": Focus_type.MEAN_ARITH_ALL,
                "MEAN_GEO_ALL": Focus_type.MEAN_GEO_ALL
            }
            
            if focus_type_str not in focus_type_map:

                return None
                
            focus_type_enum = focus_type_map[focus_type_str]

            # Create temporary volume for focus computation
            focus_volume = cp.copy(self.d_volume_module)
            
            # Apply focus function on entire volume
            focus.focus(focus_volume, focus_volume, sum_size, focus_type_enum)
            
            # Extract focus values for given position
            focus_values = []
            for z in range(focus_volume.shape[0]):
                # Extract value at specified pixel
                pixel_value = float(focus_volume[z, y_pos, x_pos])
                focus_values.append(pixel_value)

            # Clean up GPU memory
            del focus_volume
            cp.get_default_memory_pool().free_all_blocks()
            
            return focus_values
            
        except Exception as e:

            import traceback
            traceback.print_exc()
            return None
