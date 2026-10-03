// Tests de validation deterministes du module de metriques.
// Executable autonome sans framework ni dependance externe, lance via CTest.
// Aucun generateeur aleatoire : toutes les images sont construites par formule.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "core/Image.hpp"
#include "metrics/QualityMetrics.hpp"

namespace
{
    int g_failures = 0;

    void check(bool condition, const std::string& label)
    {
        if (condition)
        {
            std::cout << "[ok]    " << label << std::endl;
        }
        else
        {
            ++g_failures;
            std::cerr << "[ECHEC] " << label << std::endl;
        }
    }

    void checkNear(double actual, double expected, double tolerance, const std::string& label)
    {
        const bool close = std::fabs(actual - expected) <= tolerance;
        check(close, label + " | attendu=" + std::to_string(expected)
            + " obtenu=" + std::to_string(actual) + " tol=" + std::to_string(tolerance));
    }

    // Motif deterministe (aucune alea) : depend uniquement de x, y, canal et sel.
    m3gss::Image makePattern(int width, int height, int channels, int salt)
    {
        m3gss::Image image(width, height, channels);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                for (int c = 0; c < channels; ++c)
                {
                    const int value = (x * 7 + y * 13 + c * 29 + salt * 31) % 256;
                    image.pixel(x, y)[c] = static_cast<std::uint8_t>(value);
                }
            }
        }
        return image;
    }

    m3gss::Image makeConstant(int width, int height, int channels, std::uint8_t value)
    {
        m3gss::Image image(width, height, channels);
        std::fill(image.pixels.begin(), image.pixels.end(), value);
        return image;
    }
}

int main()
{
    // --- 1. Images identiques : MSE = 0 gere explicitement ---
    {
        const m3gss::Image reference = makePattern(16, 16, 3, 1);
        const m3gss::Image candidate = reference;
        m3gss::metrics::QualityMetrics metrics;
        std::string error;

        check(m3gss::metrics::compute(reference, candidate, &metrics, &error),
              "identiques : compute reussi");
        check(metrics.identical, "identiques : flag identical");
        checkNear(metrics.mse, 0.0, 0.0, "identiques : MSE == 0");
        check(std::isinf(metrics.psnr) && metrics.psnr > 0.0, "identiques : PSNR = +infini");
        checkNear(metrics.ssim, 1.0, 1e-9, "identiques : SSIM == 1");
    }

    // --- 2. Decalage constant de 1 : MSE = 1 -> PSNR = 10*log10(255^2) ---
    {
        const m3gss::Image reference = makeConstant(8, 8, 1, 100);
        const m3gss::Image candidate = makeConstant(8, 8, 1, 101);
        m3gss::metrics::QualityMetrics metrics;
        std::string error;

        check(m3gss::metrics::compute(reference, candidate, &metrics, &error),
              "decalage : compute reussi");
        checkNear(metrics.mse, 1.0, 1e-12, "decalage : MSE == 1");
        checkNear(metrics.psnr, 10.0 * std::log10(255.0 * 255.0), 1e-9,
                  "decalage : PSNR = 10*log10(65025)");
        check(!metrics.identical, "decalage : flag non identique");
    }

    // --- 3. MSE verifie a la main : ecarts 0, 2, 4, 6 -> (0+4+16+36)/4 = 14 ---
    {
        m3gss::Image reference(2, 2, 1);
        m3gss::Image candidate(2, 2, 1);
        reference.pixels = {0, 10, 20, 30};
        candidate.pixels = {0, 8, 24, 24};
        m3gss::metrics::QualityMetrics metrics;
        std::string error;

        check(m3gss::metrics::compute(reference, candidate, &metrics, &error),
              "mini-jeu : compute reussi");
        checkNear(metrics.mse, 14.0, 1e-12, "mini-jeu : MSE == 14");
        checkNear(metrics.psnr, 10.0 * std::log10(255.0 * 255.0 / 14.0), 1e-9,
                  "mini-jeu : PSNR f(MSE)");
    }

    // --- 4. SSIM analytique : images constantes, variances et covariance nulles ---
    {
        const m3gss::Image reference = makeConstant(8, 8, 1, 100);
        const m3gss::Image candidate = makeConstant(8, 8, 1, 110);
        m3gss::metrics::QualityMetrics metrics;
        std::string error;

        check(m3gss::metrics::compute(reference, candidate, &metrics, &error),
              "ssim constant : compute reussi");
        const double c1 = 6.5025; // (0.01 * 255)^2
        const double expected = (2.0 * 100.0 * 110.0 + c1)
            / (100.0 * 100.0 + 110.0 * 110.0 + c1);
        checkNear(metrics.ssim, expected, 1e-6, "ssim constant : valeur analytique");
    }

    // --- 5. SSIM : symetrie et bornes sur motifs deterministes ---
    {
        const m3gss::Image a = makePattern(16, 16, 3, 1);
        const m3gss::Image b = makePattern(16, 16, 3, 4);
        m3gss::metrics::QualityMetrics ab;
        m3gss::metrics::QualityMetrics ba;
        std::string error;

        check(m3gss::metrics::compute(a, b, &ab, &error)
              && m3gss::metrics::compute(b, a, &ba, &error),
              "ssim : compute reussi (sens ab et ba)");
        checkNear(ab.ssim, ba.ssim, 1e-12, "ssim : symetrie ab == ba");
        check(ab.ssim <= 1.0 + 1e-6, "ssim : borne superieure");
        check(ab.ssim >= -1.0 - 1e-6, "ssim : borne inferieure");
    }

    // --- 6. psnrFromMse : cas limites ---
    {
        check(std::isinf(m3gss::metrics::psnrFromMse(0.0)),
              "psnrFromMse(0) = +infini");
        checkNear(m3gss::metrics::psnrFromMse(65025.0), 0.0, 1e-9,
                  "psnrFromMse(65025) = 0 dB");
        checkNear(m3gss::metrics::psnrFromMse(1.0), 10.0 * std::log10(65025.0), 1e-9,
                  "psnrFromMse(1)");
    }

    // --- 7. Images incompatibles rejetees explicitement ---
    {
        const m3gss::Image reference = makePattern(8, 8, 3, 1);
        const m3gss::Image wrongSize = makePattern(9, 8, 3, 1);
        const m3gss::Image wrongChannels = makePattern(8, 8, 1, 1);
        m3gss::metrics::QualityMetrics metrics;
        std::string error;

        check(!m3gss::metrics::compute(reference, wrongSize, &metrics, &error)
              && !error.empty(),
              "incompatibles : dimensions differentes rejetees");
        checkNear(metrics.mse, 0.0, 0.0, "incompatibles : metrics remis a zero");

        check(!m3gss::metrics::compute(reference, wrongChannels, &metrics, &error)
              && !error.empty(),
              "incompatibles : canaux differents rejetes");
    }

    // --- 8. Un seul canal (niveaux de gris) : SSIM = 1 sur image identique ---
    {
        const m3gss::Image reference = makePattern(16, 16, 1, 7);
        m3gss::metrics::QualityMetrics metrics;
        std::string error;

        check(m3gss::metrics::compute(reference, reference, &metrics, &error),
              "1 canal : compute reussi");
        checkNear(metrics.ssim, 1.0, 1e-9, "1 canal : SSIM == 1");
    }

    if (g_failures == 0)
    {
        std::cout << "Tous les tests de metriques sont passes." << std::endl;
        return 0;
    }

    std::cerr << g_failures << " test(s) en echec." << std::endl;
    return 1;
}
