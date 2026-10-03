"""Dataset synthetique deterministe pour le smoke test M3GSS v0.

AUCUN telechargement, AUCUN fichier externe : toutes les images sont
construites avec NumPy a partir de motifs mathematiques. Le reproductibilite
est garantie par un `seed` fixe passe a `numpy.random.default_rng`.

Motifs disponibles : gradients horizontaux, gradients verticaux, damiers,
rectangles a fort contraste, diagonales, cercles, et combinaisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from m3gss_v0.patches import PatchCoords, grid_coords, make_patch, to_tensor

__all__ = [
    "TrainingSample",
    "make_synthetic_image",
    "build_smoke_dataset",
    "PATTERNS",
]

PATTERNS = (
    "grad_h",
    "grad_v",
    "checker",
    "rects",
    "diagonal",
    "circles",
    "combo",
)


@dataclass
class TrainingSample:
    """Un echantillon pret pour l'entrainement.

    bicubic : [3, 96, 96] float32 dans [0,1]  (entree du modele)
    target  : [3, 96, 96] float32 dans [0,1]  (HR original, cible)
    """

    name: str
    bicubic: torch.Tensor
    target: torch.Tensor
    coords: PatchCoords


def _grid(h: int, w: int):
    return np.mgrid[0:h, 0:w]


def _grad_h(h: int, w: int, rng) -> np.ndarray:
    base = np.linspace(0.0, 1.0, w, dtype=np.float64)
    return np.tile(base, (h, 1))


def _grad_v(h: int, w: int, rng) -> np.ndarray:
    base = np.linspace(0.0, 1.0, h, dtype=np.float64)
    return np.tile(base[:, None], (1, w))


def _checker(h: int, w: int, rng) -> np.ndarray:
    period = int(rng.integers(4, 17))
    ys, xs = _grid(h, w)
    return ((xs // period + ys // period) % 2).astype(np.float64)


def _rects(h: int, w: int, rng) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float64)
    for _ in range(4):
        rw = int(rng.integers(max(2, w // 12), max(3, w // 4)))
        rh = int(rng.integers(max(2, h // 12), max(3, h // 4)))
        x0 = int(rng.integers(0, max(1, w - rw)))
        y0 = int(rng.integers(0, max(1, h - rh)))
        mask[y0 : y0 + rh, x0 : x0 + rw] = float(rng.integers(0, 2))
    return mask


def _diagonal(h: int, w: int, rng) -> np.ndarray:
    period = int(rng.integers(6, 25))
    ys, xs = _grid(h, w)
    return (((xs + ys) % period) < period / 2.0).astype(np.float64)


def _circles(h: int, w: int, rng) -> np.ndarray:
    cx = float(rng.integers(w // 4, 3 * w // 4))
    cy = float(rng.integers(h // 4, 3 * h // 4))
    freq = float(rng.uniform(0.15, 0.45))
    ys, xs = _grid(h, w)
    radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return 0.5 + 0.5 * np.sin(radius * freq)


def _combo(h: int, w: int, rng) -> np.ndarray:
    return 0.5 * _grad_h(h, w, rng) + 0.5 * _checker(h, w, rng)


_BUILDERS = {
    "grad_h": _grad_h,
    "grad_v": _grad_v,
    "checker": _checker,
    "rects": _rects,
    "diagonal": _diagonal,
    "circles": _circles,
    "combo": _combo,
}


def make_synthetic_image(index: int, size: int = 192) -> np.ndarray:
    """Image RGB [size, size, 3] uint8, entierement deterministe.

    Le canal c utilise le motif PATTERNS[(index + c) % 7] : selon `index`,
    l'image est un degrade, un damier, des rectangles, des diagonales, des
    cercles ou une combinaison.
    """
    rng = np.random.default_rng(seed=1000 + index)
    h = w = size
    channels = []
    for c in range(3):
        name = PATTERNS[(index + c) % len(PATTERNS)]
        plane = _BUILDERS[name](h, w, rng)
        channels.append(np.clip(plane, 0.0, 1.0))
    img = np.stack(channels, axis=-1)
    # quantification uint8 : mêmes valeurs que le pipeline d'images réel
    return np.round(img * 255.0).astype(np.uint8)


def build_smoke_dataset(
    num_images: int = 4,
    samples_per_image: int = 2,
    patch_size: int = 96,
    base_size: int = 192,
    seed: int = 0,
    image_offset: int = 0,
) -> list[TrainingSample]:
    """Construit une liste d'échantillons HR/LR/bicubique deterministes.

    Chaque sample suit exactement :
        HR 96x96 -> area_downscale x2 -> LR 48x48 -> Catmull-Rom -> bicubique 96x96
    Les coordonnees de patch sont alignees sur la grille x2 (voir patches.py).
    """
    rng = np.random.default_rng(seed=seed)
    samples: list[TrainingSample] = []
    for i in range(num_images):
        img = make_synthetic_image(image_offset + i, size=base_size)
        positions = grid_coords(base_size, base_size, patch_size, 2)
        for j in range(samples_per_image):
            pick = int(rng.integers(0, len(positions)))
            coords = positions[pick]
            patch = make_patch(img, coords, patch_size=patch_size)
            samples.append(
                TrainingSample(
                    name=f"img{i:02d}_x{coords.x}_y{coords.y}",
                    bicubic=to_tensor(patch.bicubic),
                    target=to_tensor(patch.hr),
                    coords=coords,
                )
            )
    return samples
