# M3GSS

Prototype de moteur de reconstruction d'image (C++17, CMake, MSVC 2022, Windows x64).
Aucune IA, aucune CUDA, aucune dépendance système : uniquement **stb_image** /
**stb_image_write** (headers locaux dans `third_party/stb/`).

## Build

```powershell
cmake --build build --config Release
cmake --build build --config Debug
ctest --test-dir build -C Release --output-on-failure
```

## Exécutables

| Cible | Rôle |
|---|---|
| `M3GSS` | Application principale : bannière, conversion PNG, `upscale` (baseline non-IA) + métriques MSE/PSNR/SSIM |
| `M3GSS_dataset_generator` | Génération de paires HR/LR pour l'entraînement futur |
| `M3GSS_compare` | **Outil temporaire de comparaison visuelle** (validation de la baseline, test RE9) |
| `M3GSS_*_tests` | Tests déterministes (CTest) |

## M3GSS_compare — comparaison visuelle de la baseline

```
.\build\Release\M3GSS_compare <original> <reconstructed> <output_directory>
```

Exemple (test réel RE9, images 1920×1080) :

```powershell
.\build\Release\M3GSS_compare C:\M3GSS\test_re9.jpg C:\M3GSS\test_re9_baseline.png C:\M3GSS\comparison
```

Génère dans le dossier de sortie :

| Fichier | Contenu |
|---|---|
| `comparison.png` | Planche verticale : **ORIGINAL** / **LOW RESOLUTION** / **BASELINE BICUBIC**, panneaux de même taille avec label |
| `comparison_face.png` | Crops alignés (zone visage, position proportionnelle fixe 0.50 ; 0.40) |
| `comparison_texture.png` | Crops alignés (zone texture, 0.75 ; 0.70) |
| `comparison_edges.png` | Crops alignés (zone contours, 0.25 ; 0.75) |

Chaque planche présente `Original | Low Resolution agrandie | Baseline Bicubic`.

Notes :

- La **LR à 50 %** est recalculée avec le **même code de réduction que la baseline**
  (`src/upscale/Resampler.cpp`) : aucune incohérence possible.
- Le panneau LOW RESOLUTION est agrandi en **plus-proche-voisin uniquement pour
  l'affichage** : ce n'est pas un second algorithme de reconstruction.
- Sortie console : dimensions original / LR / reconstruction, dimensions des
  planches, dossier de sortie, temps de génération.
- Erreurs gérées : fichier absent, image invalide, dimensions incompatibles,
  dossier inaccessibile, échec d'écriture PNG.

## Commandes utiles

```powershell
.\build\Release\M3GSS                                             # bannière + usage
.\build\Release\M3GSS image.png sortie.png                        # conversion PNG
.\build\Release\M3GSS upscale entree.png sortie.png 0.5           # baseline non-IA + métriques
.\build\Release\M3GSS_dataset_generator data\raw data\hr data\lr  # paires HR/LR (facteur x2)
```
