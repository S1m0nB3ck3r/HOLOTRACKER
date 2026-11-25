# Erreur CuPy `nvrtc-builtins64_118.dll` — diagnostic (2026-11-25)

## Symptôme

Le premier appel CuPy qui compile un noyau (même `cp.ones(3)`) échoue :

```
cupy.cuda.compiler.CompileException: nvrtc: error: failed to open nvrtc-builtins64_118.dll.
  Make sure that nvrtc-builtins64_118.dll is installed correctly.
cupy_backends.cuda.libs.nvrtc.NVRTCError: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE (7)
```

Vu dans Simu-Bacteria-Holograms (`simulation_gui.py` → `processor_simu_bact.py`), mais la
cause tient à la machine, pas au code.

## Cause

Le dossier `bin` de CUDA n'est plus dans le PATH.

- Le PATH système contient `%CUDA_PATH_V11_8%\bin` (idem 11.0 et 11.2).
- Les variables système correspondantes portent maintenant les noms `CUDA_PATH_V11_8__`,
  `CUDA_PATH_V11_2__` et `CUDA_PATH_V11_0__`, avec deux `_` à la fin. `%CUDA_PATH_V11_8%`
  ne correspond donc plus à rien : Windows laisse le texte tel quel, et
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin` n'est plus dans le PATH.
- CuPy charge quand même `nvrtc64_112_0.dll` grâce à `CUDA_PATH` (qui, lui, est correct).
  Mais NVRTC cherche ensuite `nvrtc-builtins64_118.dll` dans le PATH, et ne la trouve pas.
  Le fichier existe bien : `CUDA\v11.8\bin\nvrtc-builtins64_118.dll`.

Ce n'est **pas** le problème connu `nvcc` / MSVC 14.42 (qui ne touche que `cupy.median` et
`cupy.bincount`). Ici, aucune compilation NVRTC ne passe.

## Vérification (Python 3.10, CuPy pour CUDA 11.8)

| Cas | Résultat |
|---|---|
| `import cupy; cp.ones(3)` | échec, même erreur |
| `import torch` avant cupy | OK — torch ajoute `torch\lib` au PATH, et ce dossier contient sa propre copie de la DLL |
| `CUDA\v11.8\bin` ajouté au PATH, puis cupy | OK |

Conséquence : les scripts qui importent torch marchent, ceux qui n'importent que cupy plantent.

## HOLOTRACKER est probablement touché

Le code de HOLOTRACKER importe cupy dans 20 fichiers (`code_locate/`, `tests/`, `utils/`) et
n'importe torch nulle part. Il devrait donc rencontrer la même erreur sur cette machine.
Non testé directement sur ses scripts.

## Corrections possibles (aucune appliquée)

1. **Réparer l'environnement Windows** (recommandé, règle le problème pour tous les projets) :
   - soit redonner aux variables leur nom d'origine (`CUDA_PATH_V11_8` sans `__`, etc.) ;
   - soit remplacer dans le PATH système `%CUDA_PATH_V11_8%\bin` par
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`.

   Puis rouvrir VS Code / les consoles. On ne sait pas qui a renommé les variables ni
   pourquoi : si c'était voulu (par exemple pour écarter les anciennes versions de CUDA),
   préférer la deuxième option.
2. **Contournement dans le code**, à placer avant `import cupy` :

   ```python
   import os
   os.environ["PATH"] = os.path.join(os.environ["CUDA_PATH"], "bin") + os.pathsep + os.environ["PATH"]
   ```

## Test rapide après correction

```
python -c "import cupy as cp; cp.ones(3); print('OK')"
```
