"""
Filename: main_holotracker_locate.py

Description:
Main Graphical Application for executing the holograms analysis (locating objects in 3d coordinates on holograms).
Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr
Date: 2026-06-01

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
import sys

# Le journal (holo_log.py) est à la racine du projet, un cran au-dessus de ce fichier.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_log import setup_logging

import ttkbootstrap as tb
from ui import HoloTrackerApp
from core import HoloTrackerCore
from controller_threaded import HoloTrackerController
from ui_styles import apply_custom_styles

# Journal: les exceptions ignorées y laissent une trace, sans interrompre l'application.
setup_logging("HoloTracker Locate")

root = tb.Window(themename="superhero")  # Modern dark theme

# Apply custom styles for rounded corners and enhanced appearance
apply_custom_styles(root)

core = HoloTrackerCore()
app_ui = HoloTrackerApp(root)
controller = HoloTrackerController(app_ui, core)
app_ui.controller = controller  # Inject the controller into the UI
# Load parameters and synchronize them properly
app_ui.load_parameters()
controller.sync_parameters_to_core()

try:
    root.mainloop()
finally:
    # Cleanup lors de la fermeture
    controller.cleanup()
