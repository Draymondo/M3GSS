#include "metrics/QualityMetrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
    // Constantes du SSIM (Wang, Bovik, Sheikh & Simoncelli, 2004) :
    // L = 255 (dynamique 8 bits), K1 = 0.01, K2 = 0.03.
    constexpr double kC1 = 6.5025;   // (K1 * L)^2
    constexpr double kC2 = 58.5225;  // (K2 * L)^2
    constexpr double kSsimSigma = 1.5;
    constexpr int kMaxSsimRadius = 5; // fenetre 11x11 par defaut

    // Passe separable : convolution gaussienne horizontale (data -> tmp),
    // puis verticale (tmp -> data), bords reproduits (clamp).
    // Separe en O(N*K) au lieu de O(N*K^2), et fixe un ordre d'accumulation
    // constant : resultats deterministes.
    void sepGaussian(std::vector<double>& data, std::vector<double>& tmp,
                     int width, int height, const std::vector<double>& kernel)
    {
        const int radius = (static_cast<int>(kernel.size()) - 1) / 2;

        for (int y = 0; y < height; ++y)
        {
            const double* row = data.data() + static_cast<std::size_t>(y) * width;
            double* out = tmp.data() + static_cast<std::size_t>(y) * width;

            for (int x = 0; x < width; ++x)
            {
                double sum = 0.0;
                for (int k = -radius; k <= radius; ++k)
                {
                    const int sx = std::min(std::max(x + k, 0), width - 1);
                    sum += kernel[static_cast<std::size_t>(k + radius)] * row[sx];
                }
                out[x] = sum;
            }
        }

        for (int x = 0; x < width; ++x)
        {
            for (int y = 0; y < height; ++y)
            {
                double sum = 0.0;
                for (int k = -radius; k <= radius; ++k)
                {
                    const int sy = std::min(std::max(y + k, 0), height - 1);
                    sum += kernel[static_cast<std::size_t>(k + radius)]
                        * tmp[static_cast<std::size_t>(sy) * width + x];
                }
                data[static_cast<std::size_t>(y) * width + x] = sum;
            }
        }
    }

    // SSIM global : moyenne des SSIM locaux sur toutes les fenetres valides
    // et tous les canaux de l'image.
    double computeSsim(const m3gss::Image& reference, const m3gss::Image& candidate)
    {
        const int width = reference.width;
        const int height = reference.height;
        const int channels = reference.channels;
        const std::size_t planeSize =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

        // Fenetre gaussienne 11x11 (sigma = 1.5) par defaut ; retrecie si l'image
        // est plus petite, afin de conserver toujours au moins une fenetre valide.
        const int radius = std::min(kMaxSsimRadius, (std::min(width, height) - 1) / 2);

        std::vector<double> kernel;
        kernel.reserve(static_cast<std::size_t>(2 * radius + 1));
        double kernelSum = 0.0;
        for (int t = -radius; t <= radius; ++t)
        {
            const double weight =
                std::exp(-(static_cast<double>(t) * static_cast<double>(t))
                         / (2.0 * kSsimSigma * kSsimSigma));
            kernel.push_back(weight);
            kernelSum += weight;
        }
        for (double& weight : kernel)
        {
            weight /= kernelSum;
        }

        std::vector<double> planeX(planeSize);
        std::vector<double> planeY(planeSize);
        std::vector<double> momentX2(planeSize);
        std::vector<double> momentY2(planeSize);
        std::vector<double> momentXY(planeSize);
        std::vector<double> scratch(planeSize);

        double ssimSum = 0.0;
        std::size_t sampleCount = 0;

        for (int c = 0; c < channels; ++c)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const std::size_t i = static_cast<std::size_t>(y) * width + x;
                    const double vx = static_cast<double>(reference.pixel(x, y)[c]);
                    const double vy = static_cast<double>(candidate.pixel(x, y)[c]);
                    planeX[i] = vx;
                    planeY[i] = vy;
                    momentX2[i] = vx * vx;
                    momentY2[i] = vy * vy;
                    momentXY[i] = vx * vy;
                }
            }

            sepGaussian(planeX, scratch, width, height, kernel);    // moyenne mu_x
            sepGaussian(momentX2, scratch, width, height, kernel);  // moment E[x^2]
            sepGaussian(planeY, scratch, width, height, kernel);    // moyenne mu_y
            sepGaussian(momentY2, scratch, width, height, kernel);  // moment E[y^2]
            sepGaussian(momentXY, scratch, width, height, kernel);  // covariance E[xy]

            for (int y = radius; y < height - radius; ++y)
            {
                for (int x = radius; x < width - radius; ++x)
                {
                    const std::size_t i = static_cast<std::size_t>(y) * width + x;
                    const double muX = planeX[i];
                    const double muY = planeY[i];
                    // var = E[x^2] - mu^2 : bornee a 0 par securite (arrondis).
                    const double varX = std::max(0.0, momentX2[i] - muX * muX);
                    const double varY = std::max(0.0, momentY2[i] - muY * muY);
                    const double covariance = momentXY[i] - muX * muY;

                    const double numerator = (2.0 * muX * muY + kC1) * (2.0 * covariance + kC2);
                    const double denominator =
                        (muX * muX + muY * muY + kC1) * (varX + varY + kC2);

                    ssimSum += numerator / denominator;
                    ++sampleCount;
                }
            }
        }

        if (sampleCount == 0)
        {
            return 0.0;
        }
        return ssimSum / static_cast<double>(sampleCount);
    }
}

namespace m3gss::metrics
{
    double psnrFromMse(double mse)
    {
        // Cas explicite MSE = 0 : images identiques, PSNR illimite.
        if (mse <= 0.0)
        {
            return std::numeric_limits<double>::infinity();
        }
        // PSNR (dB) = 10 * log10(MAX^2 / MSE), MAX = 255 sur 8 bits.
        return 10.0 * std::log10((255.0 * 255.0) / mse);
    }

    bool compute(const Image& reference, const Image& candidate,
                 QualityMetrics* metrics, std::string* error)
    {
        if (error != nullptr)
        {
            error->clear();
        }

        if (metrics == nullptr)
        {
            if (error != nullptr)
            {
                *error = "pointeur 'metrics' nul";
            }
            return false;
        }

        *metrics = QualityMetrics{};

        if (reference.empty() || candidate.empty())
        {
            if (error != nullptr)
            {
                *error = "image vide";
            }
            return false;
        }

        if (reference.width != candidate.width || reference.height != candidate.height)
        {
            if (error != nullptr)
            {
                *error = "dimensions differentes ("
                    + std::to_string(reference.width) + "x" + std::to_string(reference.height)
                    + " vs " + std::to_string(candidate.width) + "x"
                    + std::to_string(candidate.height) + ")";
            }
            return false;
        }

        if (reference.channels != candidate.channels)
        {
            if (error != nullptr)
            {
                *error = "nombre de canaux different ("
                    + std::to_string(reference.channels) + " vs "
                    + std::to_string(candidate.channels) + ")";
            }
            return false;
        }

        // MSE : accumulation en virgule flottante double (53 bits de mantisse) ;
        // les carres d'ecart sont des entiers <= 65025, la somme reste exacte
        // jusqu'a environ 1e10 pixels, et precise bien au-dela.
        double squaredErrorSum = 0.0;
        const std::size_t sampleCount = reference.pixels.size();
        for (std::size_t i = 0; i < sampleCount; ++i)
        {
            const double difference = static_cast<double>(reference.pixels[i])
                - static_cast<double>(candidate.pixels[i]);
            squaredErrorSum += difference * difference;
        }

        metrics->mse = squaredErrorSum / static_cast<double>(sampleCount);
        metrics->identical = (metrics->mse == 0.0);
        metrics->psnr = psnrFromMse(metrics->mse);
        metrics->ssim = computeSsim(reference, candidate);

        return true;
    }
}
