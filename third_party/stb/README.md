# stb_image / stb_image_write (dépendances locales)

Headers **header-only** ajoutés tels quels au projet, sans gestionnaire de
dépendances ni bibliothèque système.

| Fichier | Version | Source | Licence |
|---|---|---|---|
| `stb_image.h` | v2.30 | https://github.com/nothings/stb | Domaine public / MIT |
| `stb_image_write.h` | v1.16 | https://github.com/nothings/stb | Domaine public / MIT |

## Intégration

- `stb_impl.cpp` contient les `#define STB_IMAGE_IMPLEMENTATION` et
  `STB_IMAGE_WRITE_IMPLEMENTATION` : c'est la **seule** unité de compilation
  du projet qui inclut l'implémentation.
- `CMakeLists.txt` ajoute ce répertoire en `target_include_directories(... PRIVATE)`,
  donc les autres fichiers du projet peuvent faire `#include "stb_image.h"` /
  `#include "stb_image_write.h"` sans rien d'autre.

## Usage

```cpp
// Lecture PNG/JPEG (et autres formats supportés par stb)
int w, h, n;
unsigned char* pixels = stbi_load("image.png", &w, &h, &n, 0);
// ...
stbi_image_free(pixels);

// Écriture PNG
stbi_write_png("sortie.png", w, h, n, pixels, w * n);
```

Mise à jour : remplacer les deux headers par ceux du dépôt stb en conservant
les mêmes noms ; reconfigurer CMake si nécessaire (aucune autre modification).

