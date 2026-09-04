# -*- coding: utf-8 -*-
"""
Filename: link_reference.py

Description:
Exécution instrumentée de la chaîne complète Locate -> Link, utilisée à la fois pour
produire le fichier de référence et pour le test de non-régression.

Ce qui est couvert ici et nulle part ailleurs: la COUTURE entre les deux applications.
HoloTracker Locate écrit un CSV, HoloTracker Link le relit. C'est à cette jonction que se
trouvaient la plupart des défauts corrigés (lignes fantômes à l'origine, filtre
`min length` sans effet, identité des lignes perdue à l'export).

Author: Simon BECKER
mail: simon.becker@univ-lorraine.fr

License: GNU General Public License v3.0
"""

import glob
import json
import os
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code_locate"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code_link"))

import numpy as np
import pandas as pd

from core import HoloTrackerCore
from core_communicator import CoreCommunicator, CommandType
import processor

try:
    import trackpy
    trackpy.quiet()          # pas de progression ligne a ligne dans la sortie du test
except Exception:
    pass

EXPECTED_FILE = os.path.join(TESTS_DIR, "reference", "expected_link.json")
PARAMS_FILE = os.path.join(TESTS_DIR, "reference", "params.json")

# Colonnes que HoloTracker Locate doit écrire et que HoloTracker Link attend. Toute
# divergence entre les deux applications se voit ici.
COLONNES_LOCATE = ["HOLOGRAM NUMBER", "OBJECT NUMBER", "X POSITION (m)",
                   "Y POSITION (m)", "Z POSITION (m)", "NUMBER OF VOXEL"]

# Lot volontairement court: le but n'est pas de mesurer la détection (c'est le rôle de
# test_pipeline.py) mais de vérifier que le fichier produit traverse Link correctement.
NB_HOLOGRAMMES = 6
NB_PLANS = "60"

# Paramètres de liaison, figés.
SEARCH_RANGE = (5e-6, 5e-6, 5e-6)
MEMORY = 3
MIN_LENGTHS = [0, 2, 4, NB_HOLOGRAMMES + 10]   # le dernier dépasse le nombre d'images

# Seuil absurde: aucun objet ne peut le franchir. Sert au cas "hologramme sans detection".
SEUIL_SANS_DETECTION = 500


def produire_csv_locate(repertoire_travail, seuil=None):
    """Fait tourner HoloTracker Locate en mode lot et renvoie le CSV produit.

    seuil : valeur de nb_StdVar_threshold. Un seuil absurde (500) ne detecte rien, ce qui
    sert a verifier qu'un hologramme sans objet n'ecrit AUCUNE ligne. Sans ce cas, le test
    ne couvrirait jamais le defaut des lignes fantomes: toutes les images de reference ont
    des detections.
    """
    source = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Film REF", "cleaned", "*.tif")))
    if len(source) < NB_HOLOGRAMMES:
        raise FileNotFoundError("Pas assez d'images dans Film REF/cleaned")
    for chemin in source[:NB_HOLOGRAMMES]:
        shutil.copy(chemin, repertoire_travail)

    with open(PARAMS_FILE, encoding="utf-8") as f:
        params = json.load(f)
    params["number_of_planes"] = NB_PLANS
    params["holograms_directory"] = repertoire_travail
    if seuil is not None:
        params["nb_StdVar_threshold"] = str(seuil)

    core = HoloTrackerCore()
    comm = CoreCommunicator(core)
    comm.start()
    comm.send_command(CommandType.ENTER_BATCH_MODE, {"directory": repertoire_travail})
    comm.send_command(CommandType.ALLOCATE, {"parameters": params})
    # Comme le contrôleur: on ne charge l'hologramme moyen que s'il sert.
    if params.get("remove_mean"):
        comm.send_command(CommandType.LOAD_MEAN_HOLO, {"parameters": params})
    attendus = 2 + int(bool(params.get("remove_mean")))

    images = sorted(f for f in os.listdir(repertoire_travail) if f.lower().endswith(".tif"))
    for nom in images:
        comm.send_command(CommandType.PROCESS_HOLOGRAM_BATCH,
                          {"directory": repertoire_travail, "filename": nom})

    csv_path = None
    echecs = []
    recus = 0
    while recus < attendus + len(images):
        resultat = comm.get_result(timeout=300)
        if resultat is None:
            break
        recus += 1
        if resultat.command_type == CommandType.ENTER_BATCH_MODE:
            csv_path = resultat.data.get("csv_path")
        if not resultat.success:
            echecs.append(resultat.error)
    comm.stop()

    if echecs:
        raise RuntimeError("Le lot Locate a échoué: {}".format(echecs[0]))
    if not csv_path or not os.path.isfile(csv_path):
        raise RuntimeError("Aucun CSV produit par le lot Locate")
    return csv_path


def run_chaine():
    """Locate -> CSV -> Link -> table de trajectoires. Renvoie les grandeurs à comparer."""
    repertoire = tempfile.mkdtemp(prefix="holotracker_link_test_")
    try:
        csv_path = produire_csv_locate(repertoire)
        brut = pd.read_csv(csv_path)

        resultat = {
            "csv_colonnes": list(brut.columns),
            "csv_nb_lignes": int(len(brut)),
            "csv_nb_images": int(brut["HOLOGRAM NUMBER"].nunique()),
            "csv_lignes_fantomes": int((brut["OBJECT NUMBER"] == 0).sum()),
        }

        # Cas "aucun objet detecte": le CSV ne doit contenir que son en-tete.
        vide = tempfile.mkdtemp(prefix="holotracker_link_vide_")
        try:
            csv_vide = produire_csv_locate(vide, seuil=SEUIL_SANS_DETECTION)
            brut_vide = pd.read_csv(csv_vide)
            resultat["sans_detection_nb_lignes"] = int(len(brut_vide))
            resultat["sans_detection_colonnes"] = list(brut_vide.columns)
        finally:
            shutil.rmtree(vide, ignore_errors=True)

        # Chargement par Link, tel que l'application le fait
        df = processor.load_localisation_csv(csv_path)
        resultat["charge_nb_lignes"] = int(len(df))
        resultat["charge_index_conforme"] = bool(df.index.isin(brut.index).all())

        # Liaison, pour plusieurs valeurs de min length
        resultat["liaisons"] = {}
        for minlength in MIN_LENGTHS:
            traj = processor.link_df(df, SEARCH_RANGE, memory=MEMORY, minlength=minlength)
            if len(traj) == 0:
                resultat["liaisons"][str(minlength)] = {
                    "nb_trajectoires": 0, "nb_points": 0,
                    "longueur_min": None, "longueur_max": None,
                    "index_conforme": True, "nb_lignes_attribuees": 0,
                    "colonnes_sortie": None,
                }
                continue
            longueurs = traj.groupby("particle").size()
            table, attribuees = processor.build_trajectory_table(brut, traj)
            resultat["liaisons"][str(minlength)] = {
                "nb_trajectoires": int(traj["particle"].nunique()),
                "nb_points": int(len(traj)),
                "longueur_min": int(longueurs.min()),
                "longueur_max": int(longueurs.max()),
                # l'index doit rester le numéro de ligne du CSV d'origine
                "index_conforme": bool(traj.index.isin(brut.index).all()
                                       and traj.index.is_unique),
                "nb_lignes_attribuees": int(attribuees),
                "colonnes_sortie": list(table.columns),
            }
        return resultat
    finally:
        shutil.rmtree(repertoire, ignore_errors=True)
