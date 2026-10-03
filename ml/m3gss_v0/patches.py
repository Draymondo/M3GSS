"""Pipeline de patches M3GSS v0 — HR 96x96 / LR 48x48, facteur 2.

Le pipeline s'appuie EXCLUSIVEMENT sur `bicubic.py`, dont la parite avec le
moteur C++ a ete verifiee bit a bit (max_abs = 0) :
  - LR      = area_downscale(patch_HR, 48, 48)
  - bicubique= catmull_rom_upscale(LR, 96, 96)

CONVENTIONS DE COORDONNEES (verrouillees)
------------------------------------------
- Origine : coin SUPERIEUR GAUCHE. (0,0) designe le pixel (0,0) de l'image.
- Taille  : largeur = hauteur = 96 px (cote HR).
- x et y doivent etre des MULTIPLES DU FACTEUR (2). Cela garantit que la zone
  HR tombe exactement sur la grille de la reduction x2 : la zone LR 48x48
  correspondante commence en (x/2, y/2) sans decalage d'un pixel.
- Bornes  : 0 <= x  et  x + 96 <= largeur de l'image
            0 <= y  et  y + 96 <= hauteur de l'image
- AUCUN PADDING artificiel a ce stade : un patch doit tenir entierement dans
  l'image. Si l'image est trop petite, elle est REJETEE avec un motif explicite.
- Convention d'origine identique a celle du C++ (Image::row / Image::pixel) :
  ligne 0 en haut, colonne 0 a gauche.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale

__all__ = [
    "PATCH_SIZE_HR",
    "PATCH_SIZE_LR",
    "SCALE_FACTOR",
    "PatchValidationError",
    "PatchCoords",
    "PatchSample",
    "validate_coords",
    "grid_coords",
    "sample_coords",
    "make_patch",
    "to_tensor",
    "from_tensor",
]

PATCH_SIZE_HR = 96
PATCH_SIZE_LR = 48
SCALE_FACTOR = 2


class PatchValidationError(ValueError):
    """Rejet motivé d'un patch (image trop petite, coordonnées invalides)."""


@dataclass(frozen=True)
class PatchCoords:
    """Coin supérieur gauche d'un patch HR. x et y multiples du facteur."""

    x: int
    y: int


@dataclass
class PatchSample:
    """Echantillon complet : cible HR, LR, bicubique et coordonnées."""

    hr: np.ndarray
    lr: np.ndarray
    bicubic: np.ndarray
    coords: PatchCoords


def validate_coords(
    coords: PatchCoords,
    width: int,
    height: int,
    patch_size: int = PATCH_SIZE_HR,
    factor: int = SCALE_FACTOR,
) -> None:
    """Verifie qu'un patch tient dans l'image et respecte la grille x2.

    Leve PatchValidationError avec un motif explicite.
    """
    if width < patch_size or height < patch_size:
        raise PatchValidationError(
            f"image trop petite : {width}x{height} < {patch_size}x{patch_size} requis"
        )
    if coords.x < 0 or coords.y < 0:
        raise PatchValidationError(
            f"coordonnees negatives : ({coords.x}, {coords.y})"
        )
    if coords.x % factor != 0 or coords.y % factor != 0:
        raise PatchValidationError(
            f"coordonnees non alignees sur le facteur {factor} : "
            f"({coords.x}, {coords.y}) — un des deux est impair"
        )
    if coords.x + patch_size > width:
        raise PatchValidationError(
            f"patch hors borne horizontale : x={coords.x} + {patch_size} > {width}"
        )
    if coords.y + patch_size > height:
        raise PatchValidationError(
            f"patch hors borne verticale : y={coords.y} + {patch_size} > {height}"
        )


def grid_coords(
    width: int,
    height: int,
    patch_size: int = PATCH_SIZE_HR,
    factor: int = SCALE_FACTOR,
) -> list[PatchCoords]:
    """Toutes les positions valides, en ordre deterministe (lecture ligne)."""
    if width < patch_size or height < patch_size:
        raise PatchValidationError(
            f"image trop petite : {width}x{height} < {patch_size}x{patch_size} requis"
        )
    xs = list(range(0, width - patch_size + 1, factor))
    ys = list(range(0, height - patch_size + 1, factor))
    return [PatchCoords(x, y) for y in ys for x in xs]


def sample_coords(
    index: int,
    width: int,
    height: int,
    patch_size: int = PATCH_SIZE_HR,
    factor: int = SCALE_FACTOR,
) -> PatchCoords:
    """Position deterministe fonction de `index` (aucun etat global, reproductible).

    Le meme index sur la meme image redonne toujours le meme patch.
    """
    if index < 0:
        raise PatchValidationError(f"index negatif : {index}")
    if width < patch_size or height < patch_size:
        raise PatchValidationError(
            f"image trop petite : {width}x{height} < {patch_size}x{patch_size} requis"
        )
    xs = np.arange(0, width - patch_size + 1, factor, dtype=np.int64)
    ys = np.arange(0, height - patch_size + 1, factor, dtype=np.int64)
    rng = np.random.default_rng(seed=index)
    xi = int(rng.integers(0, xs.size))
    yi = int(rng.integers(0, ys.size))
    return PatchCoords(int(xs[xi]), int(ys[yi]))


def make_patch(
    hr_image: np.ndarray,
    coords: PatchCoords,
    patch_size: int = PATCH_SIZE_HR,
    factor: int = SCALE_FACTOR,
) -> PatchSample:
    """Extrait un patch HR et construit LR + bicubique avec le code valide.

    hr_image : [H, W, C] uint8, C = 3 (RGB).
    """
    if hr_image.ndim != 3:
        raise PatchValidationError(
            f"image attendue [H,W,C], recu {hr_image.shape}"
        )
    height, width = hr_image.shape[0], hr_image.shape[1]
    validate_coords(coords, width, height, patch_size, factor)

    lr_size = patch_size // factor
    if patch_size % factor != 0 or lr_size < 1:
        raise PatchValidationError(
            f"taille {patch_size} incompatible avec le facteur {factor}"
        )

    patch = np.ascontiguousarray(
        hr_image[
            coords.y : coords.y + patch_size,
            coords.x : coords.x + patch_size,
        ]
    )
    lr = area_downscale(patch, lr_size, lr_size)
    bicubic = catmull_rom_upscale(lr, patch_size, patch_size)
    return PatchSample(hr=patch, lr=lr, bicubic=bicubic, coords=coords)


# ---------------------------------------------------------------------------
# Conversions tenseur
# ---------------------------------------------------------------------------
def to_tensor(image: np.ndarray) -> torch.Tensor:
    """[H,W,C] uint8 -> [C,H,W] float32 dans [0,1].

    Regle de conversion : division exacte par 255.0 en float32, AUCUN autre
    arrondi. La valeur uint8 est entierement recuperable.
    """
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"image attendue [H,W,C], recu {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"image attendue uint8, recu {arr.dtype}")
    scaled = arr.astype(np.float32) / np.float32(255.0)
    chw = np.ascontiguousarray(scaled.transpose(2, 0, 1))
    return torch.from_numpy(chw)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """[C,H,W] float32 dans [0,1] -> [H,W,C] uint8 (inverse de to_tensor).

    Regle d'arrondi : saturation [0,1] puis * 255 puis arrondi AU PLUS PROCHE
    (numpy round, demi-pair). La valeur exacte .5 n'est jamais atteinte car
    x/255*255 redonne x a ~1e-5 pres : le demi-pair n'a donc aucun effet.
    """
    arr = tensor.detach().to("cpu").numpy()
    if arr.ndim != 3:
        raise ValueError(f"tenseur attendu [C,H,W], recu {arr.shape}")
    clipped = np.clip(arr.astype(np.float32), np.float32(0.0), np.float32(1.0))
    scaled = clipped * np.float32(255.0)
    return np.round(scaled).astype(np.uint8).transpose(1, 2, 0)
