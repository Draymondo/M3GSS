"""Reproduction Python du bicubique Catmull-Rom du baseline C++.

Reproduit fidelement `src/upscale/BaselineBicubicUpscaler.cpp` (upscaleBicubic)
et `src/upscale/Resampler.cpp` (downscaleArea + clampToByte).

Conventions C++ reproduites a l'identique :
  * ratio     = float(src_len) / float(dst_len)
  * center    = (float(i) + 0.5f) * ratio - 0.5f
  * base      = int(floor(center))
  * taps      = base + k - 1  (k = 0..3)
  * indices   = clamp(tap, 0, src_len - 1)   (bords : replication)
  * noyau     = Catmull-Rom a = -0.5, evalue en float32 (Horner identique)
  * normalisation = 1 / somme des 4 poids (float32)
  * ordre     = passe HORIZONTALE puis VERTICALE
  * conversion= clampToByte(v) : arrondi (v + 0.5) puis saturation [0,255]
  * precision = float32 (le float du C++), accumulation taps dans l'ordre k=0..3

`area_downscale` est reproduit aussi (meme ordre de boucles) car il constitue
l'entree de la baseline et doit etre identique pour la parite C++/Python.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "clamp_to_byte",
    "catmull_rom_upscale",
    "area_downscale",
    "baseline_upscale",
    "make_test_image",
]

F32 = np.float32


def clamp_to_byte(values: np.ndarray) -> np.ndarray:
    """Reproduit m3gss::resampler::clampToByte(float) sur float32."""
    v = np.asarray(values, dtype=F32)
    rounded = v + F32(0.5)
    truncated = np.clip(rounded, F32(0.0), F32(255.0)).astype(np.uint8)
    return np.where(
        rounded <= F32(0.0),
        np.uint8(0),
        np.where(rounded >= F32(255.0), np.uint8(255), truncated),
    ).astype(np.uint8)


def _catmull_rom(x: np.ndarray) -> np.ndarray:
    """Noyau Catmull-Rom (a=-0.5) - evaluation Horner identique au C++."""
    x = np.abs(np.asarray(x, dtype=F32))
    out = np.zeros_like(x, dtype=F32)

    m1 = x < F32(1.0)
    m2 = (x >= F32(1.0)) & (x < F32(2.0))

    a = x[m1]
    out[m1] = ((F32(1.5) * a - F32(2.5)) * a) * a + F32(1.0)

    b = x[m2]
    out[m2] = ((F32(-0.5) * b + F32(2.5)) * b - F32(4.0)) * b + F32(2.0)

    return out  # x >= 2 -> 0


def _coords(src_len: int, dst_len: int):
    """Centres et bases pour une dimension (meme formule que le C++)."""
    ratio = F32(src_len) / F32(dst_len)
    idx = np.arange(dst_len, dtype=F32)
    centers = (idx + F32(0.5)) * ratio - F32(0.5)
    bases = np.floor(centers).astype(np.int64)
    return centers, bases


def _weights(centers: np.ndarray, bases: np.ndarray, src_len: int):
    """Poids (dst,4) et indices (dst,4) des 4 taps par pixel de sortie."""
    dst = centers.shape[0]
    weights = np.zeros((dst, 4), dtype=F32)
    indices = np.zeros((dst, 4), dtype=np.int64)
    for k in range(4):
        tap = bases + k - 1
        diff = centers - tap.astype(F32)
        weights[:, k] = _catmull_rom(diff)
        indices[:, k] = np.clip(tap, 0, src_len - 1)
    return weights, indices


def _inv_sum(weights: np.ndarray) -> np.ndarray:
    """1 / somme des 4 poids (ordre k=0..3), 0 si somme nulle - comme le C++."""
    s = weights[:, 0].copy()
    s = s + weights[:, 1]
    s = s + weights[:, 2]
    s = s + weights[:, 3]
    inv = np.zeros_like(s, dtype=F32)
    nz = s != F32(0.0)
    inv[nz] = F32(1.0) / s[nz]
    return inv


def catmull_rom_upscale(lr: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
    """Bicubique Catmull-Rom separable : passe horizontale puis verticale.

    lr : [H, W, C] uint8. Renvoie [dst_height, dst_width, C] uint8.
    """
    src = np.asarray(lr)
    if src.ndim != 3:
        raise ValueError(f"attendu [H,W,C], recu {src.shape}")
    src_h, src_w, _ch = src.shape
    src_f = src.astype(F32)

    # Passe 1 : horizontale (lignes source -> buffer de largeur dst_width)
    centers_x, bases_x = _coords(src_w, dst_width)
    wx, ix = _weights(centers_x, bases_x, src_w)
    invx = _inv_sum(wx)
    gathered_h = src_f[:, ix, :]                    # (src_h, dst_w, 4, C)
    acc = wx[None, :, 0, None] * gathered_h[:, :, 0, :]
    for k in range(1, 4):
        acc = acc + wx[None, :, k, None] * gathered_h[:, :, k, :]
    temp = (acc * invx[None, :, None]).astype(F32)  # (src_h, dst_w, C)

    # Passe 2 : verticale (colonnes du buffer -> destination)
    centers_y, bases_y = _coords(src_h, dst_height)
    wy, iy = _weights(centers_y, bases_y, src_h)
    invy = _inv_sum(wy)
    gathered_v = temp[iy, :, :]                     # (dst_h, 4, dst_w, C)
    accv = wy[:, 0, None, None] * gathered_v[:, 0, :, :]
    for k in range(1, 4):
        accv = accv + wy[:, k, None, None] * gathered_v[:, k, :, :]
    res = (accv * invy[:, None, None]).astype(F32)

    return clamp_to_byte(res)


def area_downscale(img: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
    """Moyennage de surface - boucles et ordre identiques au C++ (float32).

    Reproduit m3gss::resampler::downscaleArea.
    """
    src = np.asarray(img)
    if src.ndim != 3:
        raise ValueError(f"attendu [H,W,C], recu {src.shape}")
    src_h, src_w, ch = src.shape
    if dst_width < 1 or dst_height < 1:
        raise ValueError("dimensions de destination invalides")
    src_f = src.astype(F32)
    out = np.zeros((dst_height, dst_width, ch), dtype=np.uint8)

    xr = F32(src_w) / F32(dst_width)
    yr = F32(src_h) / F32(dst_height)

    for oy in range(dst_height):
        y0 = F32(oy) * yr
        y1 = F32(oy + 1) * yr
        iy0 = max(0, int(y0))
        iy1 = min(src_h - 1, max(iy0, int(np.ceil(y1)) - 1))
        for ox in range(dst_width):
            x0 = F32(ox) * xr
            x1 = F32(ox + 1) * xr
            ix0 = max(0, int(x0))
            ix1 = min(src_w - 1, max(ix0, int(np.ceil(x1)) - 1))

            acc = np.zeros(ch, dtype=F32)
            total = F32(0.0)

            for iy in range(iy0, iy1 + 1):
                overlap_y = min(y1, F32(iy + 1)) - max(y0, F32(iy))
                if overlap_y <= F32(0.0):
                    continue
                for ix in range(ix0, ix1 + 1):
                    overlap_x = min(x1, F32(ix + 1)) - max(x0, F32(ix))
                    if overlap_x <= F32(0.0):
                        continue
                    weight = overlap_x * overlap_y
                    total = total + weight
                    acc = acc + weight * src_f[iy, ix, :]

            inv = F32(1.0) / total if total > F32(0.0) else F32(0.0)
            out[oy, ox, :] = clamp_to_byte(acc * inv)

    return out


def baseline_upscale(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Reproduit BaselineBicubicUpscaler::run : area reduction puis bicubique.

    Dimensions basse = max(1, lround(dim * factor)) comme le C++.
    """
    if not (0.0 < factor < 1.0):
        raise ValueError("facteur attendu dans ]0;1[")
    src = np.asarray(img)
    src_h, src_w = src.shape[0], src.shape[1]
    low_w = max(1, int(np.floor(src_w * factor + 0.5)))
    low_h = max(1, int(np.floor(src_h * factor + 0.5)))
    low = area_downscale(src, low_w, low_h)
    return catmull_rom_upscale(low, src_w, src_h)


def make_test_image(width: int = 64, height: int = 48) -> np.ndarray:
    """Image RGB [H,W,3] uint8 entierement mathematique (aucun alea).

    Degres R/G, damier B, bord dur vertical, rectangle a fort contraste,
    diagonale blanche et coin sombre - pour exercer le noyau pres des contours.
    """
    ys = np.arange(height, dtype=np.int64)
    xs = np.arange(width, dtype=np.int64)
    x_grid, y_grid = np.meshgrid(xs, ys)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[..., 0] = (x_grid * 255 // max(1, width - 1)).astype(np.uint8)
    img[..., 1] = (y_grid * 255 // max(1, height - 1)).astype(np.uint8)
    img[..., 2] = (((x_grid // 4 + y_grid // 4) % 2) * 255).astype(np.uint8)

    img[:, width // 2 :, 0] = 255
    img[height // 4 : 3 * height // 4, width // 4 : width // 2, 1] = 0
    for i in range(min(width, height)):
        img[i, min(width - 1, i * width // height), :] = 255
    img[: height // 8, : width // 8, :] = 0

    return img
