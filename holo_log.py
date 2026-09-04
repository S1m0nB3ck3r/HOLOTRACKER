# -*- coding: utf-8 -*-

"""
Filename: holo_log.py

Description:
Configuration du journal de HoloTracker. Ce fichier ne contient QUE la configuration:
les modules de l'application utilisent le module `logging` standard de Python, sans rien
importer d'ici.

Dans un module quelconque:

    import logging
    log = logging.getLogger(__name__)
    ...
    try:
        self.canvas.draw()
    except Exception:
        log.debug("exception ignorée", exc_info=True)

Le nom du fichier, la fonction et la ligne sont ajoutés automatiquement par le format du
journal: le message n'a pas besoin de les répéter.

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

import logging
import logging.handlers
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT_ROOT, "holotracker.log")

# Le fichier est plafonné et tourne sur deux sauvegardes: certaines exceptions ignorées
# se produisent dans des boucles appelées toutes les 100 ms, un journal non plafonné
# pourrait grossir sans limite.
MAX_BYTES = 2_000_000
BACKUP_COUNT = 2

# Bibliothèques tierces très bavardes en DEBUG: matplotlib, à lui seul, écrit des
# centaines de lignes sur le choix des polices à chaque tracé et rendrait le journal
# inutilisable. On ne garde que leurs avertissements.
BIBLIOTHEQUES_BAVARDES = ("matplotlib", "PIL", "fontTools", "asyncio", "numba")


def setup_logging(app_name, console_level=logging.WARNING, file_level=logging.DEBUG):
    """Installe le journal. À appeler UNE fois, au démarrage de l'application.

    Deux destinations:
      - holotracker.log  : tout, y compris les exceptions ignorées (niveau DEBUG)
      - la console       : seulement les avertissements et les erreurs

    Les exceptions ignorées sont donc tracées sans polluer l'affichage. Pour voir un
    problème d'affichage ou de sauvegarde de paramètres, ouvrir holotracker.log.
    """
    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))

    # Éviter d'empiler les handlers si la fonction est appelée deux fois.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fichier = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    fichier.setLevel(file_level)
    fichier.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s.%(funcName)s:%(lineno)d  %(message)s"))
    root.addHandler(fichier)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    for nom in BIBLIOTHEQUES_BAVARDES:
        logging.getLogger(nom).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("--- %s démarré, journal: %s ---", app_name, LOG_FILE)
    return LOG_FILE
