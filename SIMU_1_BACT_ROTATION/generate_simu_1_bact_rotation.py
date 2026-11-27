# -*- coding: utf-8 -*-
"""
Hologrammes d'une bactérie seule dont l'orientation varie de φ = 0° (axe optique) à 90° (couchée), pas de 5°.

Une simulation main_simu_hologram.py (mode bacteria_list) par angle, sortie dans phi_XX/<date>/ :
  bâtonnet 1,5 × 0,8 µm (bout à bout × épaisseur), incliné dans le plan XZ (θ = 0 : vers +X, axe horizontal
  de l'image), centre au pixel (512, 512) et à Z = 50 µm de l'hologramme ;
  onde plane 660 nm, intensité moyenne 10, bruit capteur 0,015 ; pixel 5,5 µm, grossissement 40 ; 1024 × 1024 ;
  BPM au pas de 0,1 µm dans une tranche de 10 µm autour de la bactérie (z_size = 100), puis propagation
  exacte (spectre angulaire) de 45 µm jusqu'au capteur : Z = distance_volume_camera + pos_z = 45 + 5 = 50 µm.
  Indices : milieu 1,33, bactérie 1,354 (défaut du script, mesure de Simon du 2026-10-23).

Lancer depuis la racine du dépôt :
  python SIMU_1_BACT_ROTATION/generate_simu_1_bact_rotation.py
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO, 'simu holo script', 'main_simu_hologram.py')

N = 1024
PIX, MAG = 5.5e-6, 40.0
VOX = PIX / MAG                       # 0,1375 µm au plan objet
Z_STEP, Z_SIZE = 0.1e-6, 100          # tranche BPM de 10 µm
POS_Z = 5.0e-6                        # centre de la bactérie dans la tranche
Z_HOLO = 50.0e-6                      # distance bactérie - hologramme
PHIS = list(range(0, 91, 5))

BASE = dict(
    mode='bacteria_list', nb_holo=1,
    holo_size_xy=N, border=512, upscale_factor=2, z_size=Z_SIZE, z_step=Z_STEP,
    pix_size=PIX, magnification=MAG, index_medium=1.33, index_object=1.354, wavelength=660e-9,
    illumination_mean=10.0, sensor_noise_std_min=0.015, sensor_noise_std_max=0.015,
    number_of_sources=1, sources_angle_azimuth=[0.0], sources_angle_polar=[0.0],
    distance_volume_camera=Z_HOLO - POS_Z, holo_ref_nb_planes=10,
    save_options=dict(hologram_bmp=True, hologram_tiff=True, hologram_npy=False, propagated_tiff=False,
                      propagated_npy=False, segmentation_tiff=False, segmentation_npy=False,
                      positions_csv=True, holo_ref=True),
)


def main():
    cfg_dir = os.path.join(HERE, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)
    rows = []
    for phi in PHIS:
        out = os.path.join(HERE, f'phi_{phi:02d}')
        cfg = dict(BASE, output_dir=out,
                   bacteria=[dict(pos_x=N / 2 * VOX, pos_y=N / 2 * VOX, pos_z=POS_Z,
                                  length=1.5e-6, thickness=0.8e-6, theta=0.0, phi=float(phi))])
        path = os.path.join(cfg_dir, f'config_phi_{phi:02d}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)
        print(f'φ = {phi:2d}° ...', flush=True)
        subprocess.run([sys.executable, '-W', 'ignore', SCRIPT, path], check=True, stdout=subprocess.DEVNULL,
                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        run = sorted(glob.glob(os.path.join(out, '*', '')))[-1]
        rows.append(f"{phi},{os.path.relpath(os.path.join(run, 'simulated_hologram', 'holo_0.tiff'), HERE)},"
                    f"{os.path.relpath(os.path.join(run, 'holo_ref', 'holo_ref_0.tiff'), HERE)}")
    with open(os.path.join(HERE, 'index.csv'), 'w', encoding='utf-8') as f:
        f.write('phi_deg,hologram_tiff,reference_tiff\n' + '\n'.join(rows) + '\n')
    print('terminé :', len(rows), 'hologrammes')


if __name__ == '__main__':
    main()
