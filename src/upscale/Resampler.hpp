#pragma once

#include <cstdint>
#include <string>

#include "core/Image.hpp"

namespace m3gss::resampler
{
    // Convertit une valeur flottante en pixel 8 bits [0; 255] (arrondi + saturation).
    std::uint8_t clampToByte(float value);

    // Reduction par moyennage de surface (area averaging).
    // METHODE PARTAGEE : utilisee par la baseline d'upscaling ET par le generateur
    // de dataset, pour garantir des images LR strictement comparables.
    Image downscaleArea(const Image& source, int dstWidth, int dstHeight);

    // Reduction entiere (facteur >= 2) reproduisant exactement la baseline :
    //   ratio = 1.0 / factor
    //   dims  = max(1, lround(dim * ratio))  [meme formule que BaselineBicubicUpscaler]
    //   puis downscaleArea.
    // Renvoie une image vide et detaille 'error' si l'image est vide ou le facteur
    // invalide, ou si les dimensions LR ne correspondent pas a la formule.
    //
    // Point d'extension : d'autres degradations (x3, x4, bruit, JPEG...) pourront
    // coexister dans ce module sans toucher au moteur ni au generateur.
    Image reduceByIntegerFactor(const Image& source, int factor, std::string* error = nullptr);
}
