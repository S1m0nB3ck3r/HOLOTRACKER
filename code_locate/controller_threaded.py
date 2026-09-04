import threading
import queue
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image
try:
    import cupy as cp
except ImportError:
    cp = None
from core_communicator import CoreCommunicator, CommandType
import display

# Journal: voir holo_log.py. Les exceptions ignorées y laissent une trace,
# avec fichier, fonction et ligne, sans interrompre l'application.
import logging
log = logging.getLogger(__name__)

class HoloTrackerController:
    def __init__(self, app_ui, core):
        self.ui = app_ui
        self.core = core
        self.queue = queue.Queue()
        self.running = False
        self.state = "WAIT"
        
        # Variables for BATCH processing
        self.last_batch_result = None
        self.batch_total_files = 0
        self.batch_iteration_start_time = None
        self.last_iteration_time = None
        
        # Variables for pending batch setup
        self.pending_batch_directory = None
        self.pending_batch_files = None
        self.pending_batch_display_results = None
        
        # Error state tracking
        self.has_error = False
        self.error_message = ""
        
        # Variables for current image display
        self.current_display_info = {
            'directory': None,
            'filename': None,
            'display_type': None,
            'plane_number': None,
            'additional_display': None
        }
        
        # New threaded architecture
        self.core_comm = CoreCommunicator(core)
        self.core_comm.start()
        
        # Timer to check results
        self.ui.root.after(100, self.check_core_results)
        
        self.ui.update_buttons_state(self.state)

    def update_status(self, text, is_error=False):
        """Update status bar. Error messages have priority and won't be overwritten by normal messages."""
        if is_error:
            # Set error state
            self.has_error = True
            self.error_message = text
            self.ui.status_var.set(f"[{self.state}] {text}")
        elif not self.has_error:
            # Only update if there's no error currently displayed
            self.ui.status_var.set(f"[{self.state}] {text}")
        # else: error is displayed, don't overwrite it with normal message
    
    def clear_error(self):
        """Clear the error state"""
        self.has_error = False
        self.error_message = ""

    def set_state(self, new_state):
        self.state = new_state
        self.ui.update_buttons_state(self.state)
    
    def check_core_results(self):
        """Periodically checks Core results (called every 100ms)"""
        try:
            result = self.core_comm.get_result()
            if result:
                self.handle_core_result(result)
        except Exception as e:
            # print(f" Error checking core results: {e}")
            log.debug("exception ignorée", exc_info=True)
        finally:
            # Schedule next check
            if self.running or self.core_comm.running:
                self.ui.root.after(100, self.check_core_results)
    
    def handle_core_result(self, result):
        """Handles results received from Core"""
        # print(f"📥 Received result: {result.command_type.value} - Success: {result.success}")
        
        if not result.success:
            # is_error=True: le message reste affiché et n'est pas écrasé par le
            # commentaire de progression suivant.
            self.update_status(f"Error: {result.error}", is_error=True)
            if result.data.get('batch_aborted'):
                # Le Core a vidé la file et quitté le mode batch: remettre l'IHM
                # dans un état cohérent, sinon seul STOP BATCH reste actif.
                self.set_state("WAIT")
                self.ui.on_batch_mode_exited()
            return
        
        if result.command_type == CommandType.ENTER_TEST_MODE:
            self.set_state("TEST_MODE")
            
            # Auto-set display according to Remove mean Hologram
            remove_mean = self.ui.parameters.get('remove_mean', False)
            if remove_mean:
                self.ui.display_combobox.set("CLEANED_HOLOGRAM")
                # print(" Auto-display: CLEANED_HOLOGRAM (Remove mean = True)")
            else:
                self.ui.display_combobox.set("RAW_HOLOGRAM")
                # print(" Auto-display: RAW_HOLOGRAM (Remove mean = False)")
            
            # Unified processing: ENTER_TEST_MODE now does Allocation + Pipeline
            if result.data.get('cluster_positions') is not None:
                # We have complete results (allocation + pipeline)
                num_objects = result.data.get('count', 0)
                self.last_object_count = num_objects  # Store for future parameter updates
                
                # Check if there's an error (MAX_OBJECT_DETECTION exceeded)
                if 'error' in result.data:
                    self.update_status(f"Error: {result.data['error']}", is_error=True)
                else:
                    self.clear_error()  # Clear any previous error
                    self.update_status(f"Test mode activated - Found {num_objects} objects")
                
                try:

                    self.ui.display_results(result.data)

                except Exception as e:

                    log.debug("exception ignorée", exc_info=True)
            else:
                # Just allocation (no hologram selected)
                self.update_status("Test mode activated - Allocation completed, select hologram")
                
                # Display image according to selected display type if hologram available
                directory = self.ui.dir_text.get("1.0", "end-1c").strip()
                filename = self.ui.hologram_combobox.get()
                if directory and filename:
                    try:

                        self.ui.on_display_changed()

                    except Exception as e:

                        log.debug("exception ignorée", exc_info=True)
            
        elif result.command_type == CommandType.PROCESS_HOLOGRAM:
            # Display results and timing with robust error handling
            try:

                self.ui.display_results(result.data)

            except Exception as e:

                log.debug("exception ignorée", exc_info=True)
            
            # processing_times are already in result.data, so no need for separate call
            # but we can make explicit call to ensure timing display
            try:
                if result.processing_times:
                    timing_data = {'processing_times': result.processing_times}
                    self.ui.update_timing_display(timing_data)

            except Exception as e:

                log.debug("exception ignorée", exc_info=True)
            
            # Afficher l'image
            directory = result.data.get('directory', '')
            filename = result.data.get('filename', '')
            if directory and filename:
                try:
                    display_type = self.ui.display_combobox.get()
                    plane_number = int(self.ui.plane_number_spinbox.get()) if self.ui.plane_number_spinbox.get() else 0
                    additional_display = self._get_current_additional_display()
                    result_image = display.get_display_image(self.core, directory, filename, display_type, plane_number, additional_display)
                    
                    # Store current display info
                    self.current_display_info = {
                        'directory': directory,
                        'filename': filename,
                        'display_type': display_type,
                        'plane_number': plane_number,
                        'additional_display': additional_display
                    }
                    
                    self.ui.display_hologram_image(result_image)

                except Exception as e:

                    log.debug("exception ignorée", exc_info=True)
            
            num_objects = result.data.get('count', 0)
            self.last_object_count = num_objects  # Store for future parameter updates
            
            # Check if there's an error (MAX_OBJECT_DETECTION exceeded)
            if 'error' in result.data:
                self.update_status(f"Error: {result.data['error']}", is_error=True)
            else:
                self.clear_error()  # Clear any previous error
                self.update_status(f"Hologram processed - Found {num_objects} objects")
            
        elif result.command_type == CommandType.CHANGE_PARAMETER:
            # Suppression des messages "Parameter updated" - pas utiles pour l'utilisateur
            # Le retraitement automatique affichera le résultat
            pass
            
            # Auto-reprocess current hologram after any parameter change
            directory = self.ui.dir_text.get("1.0", "end-1c").strip()
            filename = self.ui.hologram_combobox.get()
            if directory and filename:
                self.core_comm.send_command(
                    CommandType.PROCESS_HOLOGRAM,
                    {'directory': directory, 'filename': filename}
                )
        
        elif result.command_type == CommandType.REALLOCATE:
            allocation_time = result.processing_times.get('allocation', 0) if result.processing_times else 0
            self.update_status(f"Reallocation completed: {allocation_time:.3f}s")
        
        elif result.command_type == CommandType.UPDATE_ALL_PARAMETERS:
            # Parameters updated, check if we have results to display
            print(f"[DEBUG] UPDATE_ALL_PARAMETERS - has cluster_positions: {result.data.get('cluster_positions') is not None}")
            print(f"[DEBUG] UPDATE_ALL_PARAMETERS - count: {result.data.get('count', 'N/A')}")
            print(f"[DEBUG] UPDATE_ALL_PARAMETERS - data keys: {list(result.data.keys())}")
            
            if result.data.get('cluster_positions') is not None:
                # We have processed results - display them
                try:
                    self.ui.display_results(result.data)
                except Exception as e:
                    log.debug("exception ignorée", exc_info=True)
                
                # Update timing display
                try:
                    if result.processing_times:
                        timing_data = {'processing_times': result.processing_times}
                        self.ui.update_timing_display(timing_data)
                except Exception as e:
                    log.debug("exception ignorée", exc_info=True)
                
                num_objects = result.data.get('count', 0)
                self.last_object_count = num_objects  # Store for future parameter updates
                print(f"[DEBUG] Full reprocessing - Found {num_objects} objects")
                
                # Check if there's an error (MAX_OBJECT_DETECTION exceeded)
                if 'error' in result.data:
                    self.update_status(f"Error: {result.data['error']}", is_error=True)
                else:
                    self.clear_error()  # Clear any previous error
                    # Suppression du message verbeux "Parameters updated - Hologram reprocessed - Found N objects"
                    # Le comptage sera visible dans les autres onglets
                    pass
            else:
                # Just parameter update without reprocessing - no status message needed
                pass
            
            # ALWAYS refresh image display after parameter update (whether reprocessed or not)
            # This handles changes to display parameters (display_type, plane_number, use_log, etc.)
            directory = result.data.get('directory', '') or self.ui.dir_text.get("1.0", "end-1c").strip()
            filename = result.data.get('filename', '') or self.ui.hologram_combobox.get()
            
            if directory and filename:
                try:
                    display_type = self.ui.display_combobox.get()
                    plane_number = int(self.ui.plane_number_spinbox.get()) if self.ui.plane_number_spinbox.get() else 0
                    additional_display = self._get_current_additional_display()
                    use_log = self.ui.display_log_var.get()
                    result_image = display.get_display_image(self.core, directory, filename, display_type, plane_number, additional_display, use_log=use_log)
                    
                    # Store current display info
                    self.current_display_info = {
                        'directory': directory,
                        'filename': filename,
                        'display_type': display_type,
                        'plane_number': plane_number,
                        'additional_display': additional_display,
                        'use_log': use_log
                    }
                    
                    self.ui.display_hologram_image(result_image)
                except Exception as e:
                    log.debug("exception ignorée", exc_info=True)
                
        elif result.command_type == CommandType.EXIT_TEST_MODE:
            self.set_state("WAIT")
            self.update_status("Test mode exited")
        
        # ==================== BATCH MODE RESULT HANDLERS ====================
        elif result.command_type == CommandType.ALLOCATE:
            # Allocation for TEST_MODE or BATCH_MODE
            allocation_time = result.processing_times.get('allocation', 0) if result.processing_times else 0
            self.update_status(f"Allocation completed: {allocation_time:.3f}s")
            
        elif result.command_type == CommandType.ENTER_BATCH_MODE:
            if result.success:
                self.set_state("BATCH_MODE")
                csv_path = result.data.get('csv_path', '')
                if csv_path:
                    csv_filename = os.path.basename(csv_path)
                    self.update_status(f"Batch mode activated - CSV: {csv_filename}")
                    # UI update: enable batch controls, gray out parameters
                    self.ui.on_batch_mode_entered(csv_filename)
                else:
                    self.update_status("Batch mode activated")
                
                # Continuer avec le reste des commandes batch
                self._continue_batch_setup()
            else:
                # Erreur lors de l'activation du mode batch
                self.update_status(f"Failed to start batch mode: {result.error_message}")
                # S'assurer que l'état reste en WAIT
                self.set_state("WAIT")
                
        elif result.command_type == CommandType.PROCESS_HOLOGRAM_BATCH:
            # Calculate time between iterations (complete pipeline)
            import time
            current_time = time.perf_counter()
            if self.batch_iteration_start_time is not None:
                self.last_iteration_time = current_time - self.batch_iteration_start_time
                # print(f"⏱  CONTROLLER: Iteration time calculated: {self.last_iteration_time*1000:.1f}ms")
            else:
                # print("⏱  CONTROLLER: First iteration, no previous time")
                pass
            self.batch_iteration_start_time = current_time
            
            # Stocker les résultats du dernier hologramme traité
            self.last_batch_result = result.data
            
            # Essayer de trouver la checkbox DISPLAY RESULTS
            display_checked = False
            
            # Chercher spécifiquement la checkbox
            if hasattr(self.ui, 'display_results_checkbox'):
                # Essayer d'accéder à la variable via la checkbox
                try:
                    checkbox = self.ui.display_results_checkbox
                    if hasattr(checkbox, 'cget'):
                        var_name = checkbox.cget('variable')
                        if var_name:
                            display_checked = var_name.get()

                        else:
                            pass

                except Exception as e:
                    log.debug("exception ignorée", exc_info=True)

            # Si pas trouvé, essayer les noms de variables directement
            if not display_checked:
                for attr_name in dir(self.ui):
                    if 'display' in attr_name.lower() and 'var' in attr_name.lower():
                        try:
                            attr_obj = getattr(self.ui, attr_name)
                            if hasattr(attr_obj, 'get'):
                                value = attr_obj.get()

                                if attr_name in ['display_results_var', 'display_batch_results_var']:
                                    display_checked = value

                                    break
                        except:
                            log.debug("exception ignorée", exc_info=True)

            # TOUJOURS mettre à jour les temps d'exécution en mode BATCH
            if 'processing_times' in result.data and self.last_iteration_time:
                # Créer une copie des processing_times avec iteration_time
                timing_with_iteration = result.data['processing_times'].copy()
                timing_with_iteration['iteration_time'] = self.last_iteration_time
                self.ui.update_timing_display({'processing_times': timing_with_iteration})
            
            # Afficher si checkbox DISPLAY RESULTS est cochée
            if display_checked:
                self._display_batch_results(result.data)
            
            # Afficher directement le temps d'itération dans le status
            if self.last_iteration_time:
                iteration_ms = self.last_iteration_time * 1000
                processing_ms = result.data.get('processing_times', {}).get('total_processing', 0) * 1000
                
                # Afficher les timing avec iteration dans le status
                batch_info = result.data.get('batch_info', {})
                if batch_info:
                    hologram_number = batch_info.get('hologram_number', 0)
                    self.update_status(f"Batch {hologram_number}/{self.batch_total_files} - Processing: {processing_ms:.0f}ms, Iteration: {iteration_ms:.1f}ms")
            
            # Progression batch
            batch_info = result.data.get('batch_info', {})
            if batch_info:
                hologram_number = batch_info.get('hologram_number', 0)
                total_files = getattr(self, 'batch_total_files', '?')
                self.update_status(f"Batch processing: {hologram_number}/{total_files}")
                
                # Auto-exit batch mode when all holograms are processed
                if isinstance(total_files, int) and hologram_number >= total_files:
                    self.update_status(f"Batch complete: {hologram_number}/{total_files} processed - Exiting batch mode")
                    # Exit batch mode automatically
                    self.on_exit_batch_mode()
            
            num_objects = result.data.get('count', 0)
            filename = result.data.get('filename', '')
            
        elif result.command_type == CommandType.EXIT_BATCH_MODE:
            self.set_state("WAIT")
            self.update_status("Batch mode exited")
            # Mise à jour UI: réactiver les contrôles normaux
            self.ui.on_batch_mode_exited()
            
        elif result.command_type == CommandType.EXTRACT_OBJECT_SLICES:
            if result.success:
                object_num = result.data['object_num']
                slices_data = result.data['slices']
                display_in_dialog = result.data.get('display_in_dialog', False)

                # Send to UI for display - different methods based on display mode
                if display_in_dialog and hasattr(self, 'dialog_for_slices'):
                    # Display in existing dialog
                    self.ui.display_object_slices_in_dialog(object_num, slices_data, self.dialog_for_slices)
                    # Clean up dialog reference
                    delattr(self, 'dialog_for_slices')
                else:
                    # Display in separate window (original behavior)
                    self.ui.display_object_slices(object_num, slices_data)
            else:

                self.update_status(f"Error extracting slices: {result.error}")
                # If there's a dialog waiting, show error there too
                if hasattr(self, 'dialog_for_slices'):
                    messagebox.showerror("Error", f"Error extracting slices: {result.error}")
                    delattr(self, 'dialog_for_slices')

    def on_parameter_changed(self, name, value):
        try:
            if self.state == "TEST_MODE":
                # First, update ALL parameters from UI
                self.ui.update_parameters_from_ui()
                
                # IN TEST_MODE: ANY PARAMETER CHANGE = FULL REPROCESSING
                # (UPDATE → ALLOCATION → PROCESSING → DISPLAY)
                
                # Check if this parameter requires reallocation (dimensions changed)
                reallocation_params = ['holo_size_x', 'holo_size_y', 'number_of_planes']
                needs_reallocation = name in reallocation_params
                
                print(f"Parameter '{name}' changed to '{value}' - full reprocessing")
                
                # If reallocation needed, send REALLOCATE command first
                if needs_reallocation:
                    print(f"Reallocation needed (dimension changed: {name})")
                    command_data = {
                        'parameters': self.ui.parameters,
                        'changed_param': name
                    }
                    self.core_comm.send_command(CommandType.REALLOCATE, command_data)
                
                # ALWAYS restart pipeline with current hologram (if available)
                if (self.current_display_info['directory'] and 
                    self.current_display_info['filename']):
                    print(f"Restarting full pipeline with updated parameters")
                    command_data = {
                        'parameters': self.ui.parameters,
                        'directory': self.current_display_info['directory'],
                        'filename': self.current_display_info['filename']
                    }
                    # Clear previous error before restarting pipeline
                    self.clear_error()
                    self.core_comm.send_command(CommandType.UPDATE_ALL_PARAMETERS, command_data)
                else:
                    # No hologram loaded yet - just update parameters
                    command_data = {'parameters': self.ui.parameters}
                    # Clear previous error before updating parameters (may trigger processing)
                    self.clear_error()
                    self.core_comm.send_command(CommandType.UPDATE_ALL_PARAMETERS, command_data)
                
            else:
                # En mode WAIT, mise à jour directe
                result = self.core.set_parameter(name, value)
                self.update_status(f"{name} : {value}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
    
    def sync_parameters_to_core(self):
        """Synchronize parameters one by one with explicit validation"""
        if not hasattr(self.ui, 'parameters') or not self.ui.parameters:
            return
            
        params = self.ui.parameters
        synced_count = 0
        
        try:
            # === HOLOGRAM SETUP ===
            if "mean_hologram_image_path" in params:
                self.core.set_parameter("mean_hologram_image_path", str(params["mean_hologram_image_path"]))
                synced_count += 1
                
            if "holograms_directory" in params:
                self.core.set_parameter("holograms_directory", str(params["holograms_directory"]))
                synced_count += 1
                
            if "image_type" in params:
                self.core.set_parameter("image_type", str(params["image_type"]))
                synced_count += 1
            
            # === OPTICAL PARAMETERS ===
            if "wavelength" in params and params["wavelength"]:
                self.core.set_parameter("wavelength", float(params["wavelength"]))
                synced_count += 1
                
            if "medium_optical_index" in params and params["medium_optical_index"]:
                self.core.set_parameter("medium_optical_index", float(params["medium_optical_index"]))
                synced_count += 1
                
            if "objective_magnification" in params and params["objective_magnification"]:
                self.core.set_parameter("objective_magnification", float(params["objective_magnification"]))
                synced_count += 1
                
            if "pixel_size" in params and params["pixel_size"]:
                self.core.set_parameter("pixel_size", float(params["pixel_size"]))
                synced_count += 1
            
            # === RECONSTRUCTION PARAMETERS ===
            if "holo_size_x" in params and params["holo_size_x"]:
                self.core.set_parameter("holo_size_x", int(params["holo_size_x"]))
                synced_count += 1
                
            if "holo_size_y" in params and params["holo_size_y"]:
                self.core.set_parameter("holo_size_y", int(params["holo_size_y"]))
                synced_count += 1
                
            if "distance_ini" in params and params["distance_ini"]:
                self.core.set_parameter("distance_ini", float(params["distance_ini"]))
                synced_count += 1
                
            if "step" in params and params["step"]:
                self.core.set_parameter("step", float(params["step"]))
                synced_count += 1
                
            if "number_of_planes" in params and params["number_of_planes"]:
                self.core.set_parameter("number_of_planes", int(params["number_of_planes"]))
                synced_count += 1
            
            # === FILTERING PARAMETERS ===
            if "remove_mean" in params:
                self.core.set_parameter("remove_mean", bool(params["remove_mean"]))
                synced_count += 1
                
            if "cleaning_type" in params:
                self.core.set_parameter("cleaning_type", str(params["cleaning_type"]))
                synced_count += 1
                
            if "high_pass" in params and params["high_pass"]:
                self.core.set_parameter("high_pass", int(params["high_pass"]))
                synced_count += 1
                
            if "low_pass" in params and params["low_pass"]:
                self.core.set_parameter("low_pass", int(params["low_pass"]))
                synced_count += 1
                
            # === FOCUS PARAMETERS ===
            if "focus_type" in params:
                self.core.set_parameter("focus_type", str(params["focus_type"]))
                synced_count += 1
                
            if "sum_size" in params and params["sum_size"]:
                self.core.set_parameter("sum_size", int(params["sum_size"]))
                synced_count += 1
            
            # === LOCALIZATION PARAMETERS ===
            if "nb_StdVar_threshold" in params and params["nb_StdVar_threshold"]:
                self.core.set_parameter("nb_StdVar_threshold", float(params["nb_StdVar_threshold"]))
                synced_count += 1
                
            if "connectivity" in params and params["connectivity"]:
                self.core.set_parameter("connectivity", int(params["connectivity"]))
                synced_count += 1
                
            if "min_voxel" in params and params["min_voxel"]:
                self.core.set_parameter("min_voxel", int(params["min_voxel"]))
                synced_count += 1
                
            if "max_voxel" in params and params["max_voxel"]:
                self.core.set_parameter("max_voxel", int(params["max_voxel"]))
                synced_count += 1
                
            if "batch_threshold" in params:
                self.core.set_parameter("batch_threshold", params["batch_threshold"])
                synced_count += 1
                
            if "additional_display" in params:
                self.core.set_parameter("additional_display", str(params["additional_display"]))
                synced_count += 1
            
            self.update_status(f"Parameters synchronized ({synced_count} params)")
            
        except Exception as e:
            self.update_status(f"Error syncing parameters: {str(e)}")

    def on_enter_test_mode(self):
        if self.state == "WAIT":
            self.clear_error()  # Clear any previous error when entering test mode
            self.update_status("Entering test mode...")
            
            # Préparer les paramètres depuis l'UI
            self.ui.update_parameters_from_ui()
            
            # Récupérer hologramme sélectionné
            directory = self.ui.dir_text.get("1.0", "end-1c").strip()
            filename = self.ui.hologram_combobox.get()
            
            # Nouveau workflow séparé: ENTER_TEST_MODE + ALLOCATE + optionnel PROCESS

            # 1. Changement d'état
            self.core_comm.send_command(CommandType.ENTER_TEST_MODE, {})
            
            # 2. Allocation
            self.core_comm.send_command(CommandType.ALLOCATE, {'parameters': self.ui.parameters})
            
            # 3. Chargement hologramme moyen
            # L'hologramme moyen n'est chargé que s'il sert: sinon son chemin peut être vide,
            # la commande échoue pour rien et laisse une erreur affichée alors que le
            # traitement se déroulera correctement.
            if self.ui.parameters.get('remove_mean'):
                self.core_comm.send_command(CommandType.LOAD_MEAN_HOLO, {'parameters': self.ui.parameters})
            
            # 4. Traitement (si hologramme disponible)
            if directory and filename:
                # Clear any previous error message before starting new pipeline
                self.clear_error()
                self.core_comm.send_command(CommandType.PROCESS_HOLOGRAM, {
                    'directory': directory, 
                    'filename': filename
                })

    def on_exit_test_mode(self):
        if self.state == "TEST_MODE":
            self.clear_error()  # Clear error when exiting test mode
            self.update_status("Exiting test mode...")
            
            # Envoyer commande de sortie du mode test au Core
            self.core_comm.send_command(
                CommandType.EXIT_TEST_MODE,
                {}
            )

    def on_hologram_selected(self, directory, filename):
        if self.state == "TEST_MODE":
            self.clear_error()  # Clear error when selecting a new hologram
            self.update_status(f"Processing {filename}...")
            
            # Envoyer commande de traitement d'hologramme au Core
            self.core_comm.send_command(
                CommandType.PROCESS_HOLOGRAM,
                {
                    'directory': directory,
                    'filename': filename
                }
            )
        else:
            self.update_status("Enter test mode to process holograms.")

    def _get_current_additional_display(self):
        """Get the current additional display setting from UI"""
        try:
            return self.ui.additional_display_combobox.get() if hasattr(self.ui, 'additional_display_combobox') else "None"
        except:
            return "None"

    def on_display_changed(self, display_type, plane_number, additional_display="None"):
        """Handle display type or plane number change - update image immediately"""
        if self.state == "TEST_MODE":
            filename = self.ui.hologram_combobox.get()
            directory = self.ui.dir_text.get("1.0", tk.END).strip()
            if filename and directory:
                try:
                    # Get LOG checkbox state
                    use_log = self.ui.display_log_var.get()
                    
                    # Store current display info
                    self.current_display_info = {
                        'directory': directory,
                        'filename': filename,
                        'display_type': display_type,
                        'plane_number': plane_number,
                        'additional_display': additional_display,
                        'use_log': use_log
                    }
                    
                    # Update image display immediately
                    result_image = display.get_display_image(self.core, directory, filename, display_type, plane_number, additional_display, use_log=use_log)
                    self.ui.display_hologram_image(result_image)
                    log_text = " (LOG)" if use_log else ""
                    self.update_status(f"Display updated to {display_type}{log_text} - Plane {plane_number} - Additional: {additional_display}")
                except Exception as e:
                    self.update_status(f"Error updating display: {str(e)}")
    
    def get_pixel_info(self, x, y):
        """Get pixel information (coordinates and value) for the current displayed image"""
        try:
            if not self.current_display_info['directory'] or not self.current_display_info['filename']:
                return f"Pixel ({x}, {y}) - No image data available"
                
            # Get current display parameters
            directory = self.current_display_info['directory']
            filename = self.current_display_info['filename']
            display_type = self.current_display_info['display_type']
            plane_number = self.current_display_info['plane_number']
            
            # Get pixel value from core
            try:
                pixel_value = display.get_pixel_value(self.core, directory, filename, display_type, plane_number, x, y)
                
                # Convert to physical coordinates
                physical_coords = self._get_physical_coordinates(x, y, display_type)
                
                # Format the information
                if physical_coords:
                    phys_x, phys_y = physical_coords
                    coord_info = f"Pixel ({x}, {y}) | Physical ({phys_x:.2f}, {phys_y:.2f}) µm"
                else:
                    coord_info = f"Pixel ({x}, {y})"
                    
                # Format value based on type
                if isinstance(pixel_value, complex):
                    value_info = f"Value: {pixel_value.real:.3f} + {pixel_value.imag:.3f}i"
                elif isinstance(pixel_value, (int, float)):
                    value_info = f"Value: {pixel_value:.3f}"
                else:
                    value_info = f"Value: {pixel_value}"
                    
                return f"{coord_info} | {value_info}"
                
            except Exception as e:
                return f"Pixel ({x}, {y}) - Error getting value: {str(e)}"
                
        except Exception as e:
            return f"Pixel ({x}, {y}) - Error: {str(e)}"
    
    def _get_physical_coordinates(self, x, y, display_type):
        """Convert pixel coordinates to physical coordinates in micrometers"""
        try:
            # Get parameters from core or UI
            cam_pix_size = float(self.core.get_parameter("pixel_size", 7e-6))  # meters - use same parameter name as core
            cam_magnification = float(self.core.get_parameter("objective_magnification", 40.0))  # use same parameter name as core
            step_um = float(self.core.get_parameter("step", 0.2e-6)) * 1e6  # Convert step from meters to micrometers
            
            # Calculate effective pixel size in micrometers for XY plane
            effective_pixel_size_um = (cam_pix_size * 1e6) / cam_magnification
            
            # Convert to physical coordinates based on projection type
            if display_type in ["XY_SUM_PROJECTION", "XY_MAX_PROJECTION", "RAW_HOLOGRAM", "CLEANED_HOLOGRAM", 
                              "FILTERED_HOLOGRAM", "FFT_HOLOGRAM", "FFT_FILTERED_HOLOGRAM", "VOLUME_PLANE_NUMBER"]:
                # Standard XY plane: both dimensions use camera pixel size
                phys_x = x * effective_pixel_size_um
                phys_y = y * effective_pixel_size_um
                
            elif display_type in ["XZ_SUM_PROJECTION", "XZ_MAX_PROJECTION"]:
                # XZ plane: X uses camera pixel size, Y (vertical) uses Z step
                phys_x = x * effective_pixel_size_um  # X dimension
                phys_y = y * step_um  # Z dimension (vertical axis)
                
            elif display_type in ["YZ_SUM_PROJECTION", "YZ_MAX_PROJECTION"]:
                # YZ plane: X (horizontal) uses Z step, Y uses camera pixel size  
                phys_x = x * step_um  # Z dimension (horizontal axis)
                phys_y = y * effective_pixel_size_um  # Y dimension
                
            else:
                # Default case
                phys_x = x * effective_pixel_size_um
                phys_y = y * effective_pixel_size_um
            
            return phys_x, phys_y
            
        except Exception as e:
            return None

    # ==================== BATCH MODE METHODS ====================
    
    def on_enter_batch_mode(self, batch_directory):
        """Start batch processing mode with FIFO approach"""
        if self.state == "WAIT":
            self.update_status("Preparing batch processing...")
            
            # Préparer les paramètres depuis l'UI
            self.ui.update_parameters_from_ui()
            
            # Lister les fichiers selon image_type
            image_type = self.ui.parameters.get('image_type', 'bmp')
            try:
                files = [f for f in os.listdir(batch_directory) 
                        if f.lower().endswith(f'.{image_type.lower()}')]
                files.sort()
            except Exception as e:
                self.update_status(f"Error reading directory: {str(e)}")
                return
            
            if not files:
                self.update_status(f"No {image_type} files found in directory")
                return
            
            display_results = getattr(self.ui, 'display_results_var', None)
            display_enabled = display_results.get() if display_results else False
            
            # Tracker le nombre total pour progression
            self.batch_total_files = len(files)
            
            # Réinitialiser le timing des itérations
            self.batch_iteration_start_time = None
            self.last_iteration_time = None
            
            # Auto-définir DISPLAY selon Remove mean Hologram (toujours au démarrage BATCH)
            remove_mean = self.ui.parameters.get('remove_mean', False)
            if remove_mean:
                self.ui.display_combobox.set("CLEANED_HOLOGRAM")
            else:
                self.ui.display_combobox.set("RAW_HOLOGRAM")

            # Envoyer TOUTES les commandes dans la FIFO
            # Cet appel était en dehors du bloc 'if': hors de l'état WAIT, 'files' et
            # 'display_enabled' n'étaient jamais créées et Python levait un
            # UnboundLocalError. Le bouton START BATCH est grisé hors WAIT, mais la
            # branche par défaut de update_buttons_state réactive tous les boutons pour
            # un état inattendu, ce qui rendait le chemin atteignable.
            self._send_batch_commands_to_fifo(batch_directory, files, display_enabled)

            self.update_status(f"Batch started: {len(files)} files queued")
        else:
            self.update_status(f"Cannot start batch while in state {self.state}", is_error=True)

    def _send_batch_commands_to_fifo(self, directory, files, display_results):
        """Send all batch commands to FIFO at once"""
        
        # 1. Création CSV + état BATCH (seulement cette commande au début)
        self.core_comm.send_command(CommandType.ENTER_BATCH_MODE, {'directory': directory})
        
        # Stocker les informations pour continuer après succès ENTER_BATCH_MODE
        self.pending_batch_directory = directory
        self.pending_batch_files = files
        self.pending_batch_display_results = display_results
    
    def _continue_batch_setup(self):
        """Continue batch setup after successful ENTER_BATCH_MODE"""
        directory = self.pending_batch_directory
        files = self.pending_batch_files
        
        # 2. Allocation GPU
        self.core_comm.send_command(CommandType.ALLOCATE, {'parameters': self.ui.parameters})
        
        # 3. Chargement hologramme moyen
        # L'hologramme moyen n'est chargé que s'il sert: sinon son chemin peut être vide,
        # la commande échoue pour rien et laisse une erreur affichée alors que le
        # traitement se déroulera correctement.
        if self.ui.parameters.get('remove_mean'):
            self.core_comm.send_command(CommandType.LOAD_MEAN_HOLO, {'parameters': self.ui.parameters})
        
        # 4. Traitement de chaque fichier (plus de display_results dans FIFO)
        for filename in files:
            command_data = {
                'directory': directory, 
                'filename': filename
            }
            self.core_comm.send_command(CommandType.PROCESS_HOLOGRAM_BATCH, command_data)
        
        # Nettoyer les variables temporaires
        self.pending_batch_directory = None
        self.pending_batch_files = None
        self.pending_batch_display_results = None

    def _display_batch_results(self, result_data):
        """Affiche les résultats batch selon le DISPLAY sélectionné dynamiquement"""
        try:

            # Afficher les résultats (graphique 3D, timing, etc.)

            self.ui.display_results(result_data)
            
            # Afficher l'image selon le DISPLAY actuel (peut changer pendant le batch)
            display_type = self.ui.display_combobox.get()
            plane_number_str = self.ui.plane_number_spinbox.get()
            plane_number = int(plane_number_str) if plane_number_str else 0

            directory = result_data.get('directory', '')
            filename = result_data.get('filename', '')

            if directory and filename:
                additional_display = self._get_current_additional_display()

                # Use the features from the batch result for overlay drawing.
                # Temporarily inject result.data into core.results so get_display_image
                # (which uses self.results) draws centroids consistent with the shown CSV/3D plot.
                prev_results = getattr(self.core, 'results', None)
                try:
                    # Build a minimal results dict expected by display.get_display_image overlays
                    temp_results = {
                        'number_of_objects': result_data.get('count', 0),
                        'features': result_data.get('features', None),
                        'processing_times': result_data.get('processing_times', {})
                    }
                    self.core.results = temp_results

                    result_image = display.get_display_image(self.core, directory, filename, display_type, plane_number, additional_display)

                    # Store current display info
                    self.current_display_info = {
                        'directory': directory,
                        'filename': filename,
                        'display_type': display_type,
                        'plane_number': plane_number,
                        'additional_display': additional_display
                    }

                    if result_image is not None:
                        self.ui.display_hologram_image(result_image)
                finally:
                    # Restore previous core.results to avoid side-effects
                    self.core.results = prev_results

        except Exception as e:
            import traceback
            traceback.print_exc()

    def on_exit_batch_mode(self):
        """Exit batch processing mode and clear FIFO"""
        if self.state == "BATCH_MODE":
            # VIDER LA FIFO
            cleared = self.core_comm.clear_command_queue()
            self.update_status(f"Batch stopped - {cleared} commands cancelled")
            
            # Envoyer commande de sortie du mode batch au Core
            self.core_comm.send_command(CommandType.EXIT_BATCH_MODE, {})

    def on_process_hologram_batch(self, directory, filename):
        """Process a single hologram in batch mode"""
        if self.state == "BATCH_MODE":
            self.update_status(f"Batch processing {filename}...")
            # Clear any previous error message before starting new pipeline
            self.clear_error()
            # Envoyer commande de traitement d'hologramme batch au Core
            self.core_comm.send_command(
                CommandType.PROCESS_HOLOGRAM_BATCH,
                {
                    'directory': directory,
                    'filename': filename
                }
            )
        else:
            self.update_status("Enter batch mode to process holograms in batch.")

    def on_holograms_directory_changed(self, directory):
        """Handle change in holograms directory"""
        try:
            self.core.set_parameter("holograms_directory", directory)
            
            # Update the hologram list in UI
            if os.path.exists(directory):
                supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
                files = [f for f in os.listdir(directory) 
                        if any(f.lower().endswith(ext) for ext in supported_extensions)]
                files.sort()
                
                # Update UI combobox
                self.ui.hologram_combobox['values'] = files
                if files:
                    self.ui.hologram_combobox.set(files[0])
                    self.update_status(f"Directory loaded: {len(files)} holograms found")
                else:
                    self.update_status("Directory loaded: No supported images found")
            else:
                self.update_status("Invalid directory path")
                
        except Exception as e:
            self.update_status(f"Error loading directory: {str(e)}")

    def handle_parameter_changed(self, data):
        """Handle parameter change event from message manager"""
        if isinstance(data, dict) and 'name' in data and 'value' in data:
            self.on_parameter_changed(data['name'], data['value'])

    def extract_object_slices(self, object_num, pos_x, pos_y, pos_z, vox_xy, vox_z):
        """Extract 3 slice views around an object from the reconstructed volume"""

        if self.state not in ["TEST", "TEST_MODE"]:

            self.update_status("Object viewer only available in TEST mode")
            messagebox.showerror("Error", "Please enter TEST mode first")
            return
            
        # Check if we have volume data
        if not hasattr(self.core, 'd_volume_module') or self.core.d_volume_module is None:

            self.update_status("No volume data available")
            messagebox.showerror("Error", "No volume data available. Please process a hologram first.")
            return
            
        # Send command to core to extract slices
        command_data = {
            'object_num': object_num,
            'position': [pos_x, pos_y, pos_z],
            'vox_xy': vox_xy,
            'vox_z': vox_z
        }

        print(f"📐 Slice sizes: XY={vox_xy}x{vox_xy}, Z={vox_z}")
        
        self.core_comm.send_command(CommandType.EXTRACT_OBJECT_SLICES, command_data)

    def extract_object_slices_for_dialog(self, object_num, pos_x, pos_y, pos_z, vox_xy, vox_z, dialog):
        """Extract 3 slice views and display them in the provided dialog"""

        if self.state not in ["TEST", "TEST_MODE"]:

            self.update_status("Object viewer only available in TEST mode")
            messagebox.showerror("Error", "Please enter TEST mode first")
            return
            
        # Check if we have volume data
        if not hasattr(self.core, 'd_volume_module') or self.core.d_volume_module is None:

            self.update_status("No volume data available")
            messagebox.showerror("Error", "No volume data available. Please process a hologram first.")
            return
        
        # Store dialog reference for the result handler
        self.dialog_for_slices = dialog
        
        # Send command to core to extract slices
        command_data = {
            'object_num': object_num,
            'position': [pos_x, pos_y, pos_z],
            'vox_xy': vox_xy,
            'vox_z': vox_z,
            'display_in_dialog': True  # Flag to indicate dialog display
        }

        print(f"📐 Slice sizes: XY={vox_xy}x{vox_xy}, Z={vox_z}")
        
        self.core_comm.send_command(CommandType.EXTRACT_OBJECT_SLICES, command_data)

    def start_mean_hologram_computation(self, directory, image_type, mean_type="arithmetic"):
        """Start mean hologram computation in a separate thread"""

        def compute_in_thread():
            try:
                self.update_status(f"Computing {mean_type} mean hologram...")
                
                # Define progress callback to update UI
                def progress_callback(current, total):
                    progress = int((current / total) * 100)
                    self.ui.root.after(0, lambda: self.update_status(f"Processing image {current}/{total} ({progress}%)"))
                
                # Call the core method with mean_type
                mean_path = self.core.compute_mean_hologram(directory, image_type, progress_callback, mean_type)
                
                # Update UI and parameters with the new mean hologram path
                self.ui.root.after(0, lambda: self._update_mean_hologram_path(mean_path))
                
                # Update UI on completion
                self.ui.root.after(0, lambda: self.update_status("Mean hologram computation completed"))
                self.ui.root.after(0, lambda: messagebox.showinfo("Success", f"Mean hologram saved to: {mean_path}"))

            except Exception as e:
                error_msg = f"Error computing mean hologram: {str(e)}"

                self.ui.root.after(0, lambda: self.update_status("Mean hologram computation failed"))
                self.ui.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        # Start computation in separate thread to avoid blocking UI
        import threading
        thread = threading.Thread(target=compute_in_thread, daemon=True)
        thread.start()

    def _update_mean_hologram_path(self, mean_path):
        """Update the mean hologram path in UI and parameters"""
        # Update the text field
        self.ui.mean_image_text.delete("1.0", tk.END)
        self.ui.mean_image_text.insert("1.0", mean_path)
        
        # Update parameters
        self.ui.parameters["mean_hologram_image_path"] = mean_path
        
        # Notify parameter change
        self.on_parameter_changed("mean_hologram_image_path", mean_path)
        
        self.update_status(f"Mean hologram path updated: {os.path.basename(mean_path)}")

    def run_focus_analysis(self, x_pos, y_pos, configs):
        """Launches focus function analysis for a given position
        
        Args:
            x_pos: X position of the pixel to analyze
            y_pos: Y position of the pixel to analyze  
            configs: List of focus configurations to analyze
                    Each config contains: {'focus_type': str, 'sum_size': int}
        
        Returns:
            List of normalized results for each configuration or None in case of error
        """
        try:
            # Store click position for dialog
            self.last_click_position = (x_pos, y_pos)
            
            # Check that we are in TEST mode
            if self.state != "TEST_MODE":

                return None
                
            # Get current parameters
            current_params = self.core.get_parameters_dict()
            if not current_params:

                return None
                
            # Restart pipeline in TEST mode up to focus step
            self.restart_test_mode_pipeline()
            
            # Wait for processing to complete
            import time
            timeout = 30  # 30 seconds maximum
            start_time = time.time()
            
            while self.state == "PROCESSING" and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if self.state != "TEST_MODE":

                return None
                
            # Launch analysis for each configuration
            results = []
            for config in configs:
                focus_type = config['focus_type']
                sum_size = config['sum_size']

                # Call core focus analysis method
                focus_values = self.core.analyze_focus_at_position(
                    x_pos, y_pos, focus_type, sum_size
                )
                
                if focus_values is not None:
                    # Normalize values between 0 and 1
                    focus_array = np.array(focus_values)
                    if len(focus_array) > 0:
                        min_val = np.min(focus_array)
                        max_val = np.max(focus_array)
                        if max_val > min_val:
                            normalized = (focus_array - min_val) / (max_val - min_val)
                        else:
                            normalized = np.ones_like(focus_array) * 0.5
                        results.append(normalized.tolist())
                    else:
                        results.append([])
                else:

                    results.append([])

            return results
            
        except Exception as e:

            return None

    def restart_test_mode_pipeline(self):
        """Restarts the pipeline in TEST mode up to the focus step"""
        try:
            if self.state != "TEST_MODE":

                return False
                
            print(" Restarting pipeline in TEST mode...")
            
            # Temporarily change state
            self.set_state("PROCESSING")
            self.update_status("Pipeline restarting...")
            
            # Relaunch complete processing
            self.core_comm.send_command(
                CommandType.RUN_TEST,
                self.core.get_parameters_dict()
            )
            
            return True
            
        except Exception as e:

            self.set_state("TEST_MODE")
            return False

    def cleanup(self):
        """Cleanup resources"""
        if self.core_comm:
            self.core_comm.stop()
        if self.running:
            self.running = False
