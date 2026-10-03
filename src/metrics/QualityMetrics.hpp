#pragma once

#include <string>

#include "core/Image.hpp"

namespace m3gss::metrics
{
    // Metriques de qualite d'une reconstruction par rapport a l'image de reference.
    struct QualityMetrics
    {
        double mse = 0.0;       // erreur quadratique moyenne, tous canaux [0; 65025]
        double psnr = 0.0;      // rapport signal/bruit en dB ; +infini si mse == 0
        double ssim = 0.0;      // similarite structurelle globale [-1; 1]
        bool identical = false; // vrai si et seulement si mse == 0
    };

    // PSNR derive directement du MSE : 10 * log10(MAX^2 / MSE), MAX = 255 (8 bits).
    // Cas MSE = 0 gere explicitement : renvoie +infini (images identiques).
    double psnrFromMse(double mse);

    // Calcule MSE, PSNR et SSIM entre 'reference' et 'candidate'.
    // Contrainte : memes dimensions et meme nombre de canaux (architecture Image).
    // Renvoie false et detaille la raison dans 'error' si les images sont
    // incompatibles ; 'metrics' est alors remis a zero.
    bool compute(const Image& reference, const Image& candidate,
                 QualityMetrics* metrics, std::string* error = nullptr);
}
