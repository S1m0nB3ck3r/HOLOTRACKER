# -*- coding: utf-8 -*-
"""
Filename: make_reference.py

Description:
Régénère tests/reference/expected.json à partir du code courant.
À n'exécuter QUE lorsqu'un changement de résultat est voulu et compris: ce fichier est
l'étalon du test de non-régression, l'écraser sans raison revient à supprimer le test.

Usage:
    python tests/make_reference.py

Author: Simon BECKER
License: GNU General Public License v3.0
"""

import json
import os
import sys
import platform
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_reference import run_pipeline, load_params, EXPECTED_FILE, HIST_BINS
from link_reference import run_chaine, EXPECTED_FILE as EXPECTED_LINK_FILE

import cupy as cp


def _meta(extra=None):
    infos = {
        "genere_le": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "python": platform.python_version(),
    }
    infos.update(extra or {})
    return infos


def main():
    if os.path.exists(EXPECTED_FILE):
        print(f"ATTENTION: {EXPECTED_FILE} existe déjà et va être écrasé.")
        if input("Confirmer (oui/non) ? ").strip().lower() not in ("oui", "o", "yes", "y"):
            print("Abandon."); return 1

    params = load_params()
    print("Exécution du pipeline sur l'image de référence...")
    result = run_pipeline(params)

    result["_meta"] = {
        "genere_le": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "python": platform.python_version(),
        "hist_bins": HIST_BINS,
        "note": ("Étalon de non-régression. Les bornes d'histogramme (hist_range) sont "
                 "figées ici et réutilisées par le test, pour que les effectifs restent "
                 "comparables d'une exécution à l'autre."),
    }

    os.makedirs(os.path.dirname(EXPECTED_FILE), exist_ok=True)
    with open(EXPECTED_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print(f"\nRéférence écrite: {EXPECTED_FILE}")
    print(f"  image          : {result['image']}")
    print(f"  objets trouvés : {result['n_objects']}")
    print(f"  seuil          : {result['threshold']:.9f}")

    print("\nExécution de la chaîne Locate -> Link...")
    lien = run_chaine()
    lien["_meta"] = _meta({"note": "Étalon de la couture entre les deux applications."})
    with open(EXPECTED_LINK_FILE, "w", encoding="utf-8") as f:
        json.dump(lien, f, indent=1, ensure_ascii=False)

    print(f"\nRéférence écrite: {EXPECTED_LINK_FILE}")
    print(f"  détections     : {lien['csv_nb_lignes']} sur {lien['csv_nb_images']} images")
    for ml in sorted(lien["liaisons"], key=int):
        print(f"  min length {ml:>2} : {lien['liaisons'][ml]['nb_trajectoires']} trajectoires")
    return 0


if __name__ == "__main__":
    sys.exit(main())
