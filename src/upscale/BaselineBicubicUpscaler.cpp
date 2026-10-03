#include "upscale/BaselineBicubicUpscaler.hpp"
#include "upscale/Resampler.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
    // Noyau bicubique de Catmull-Rom (a = -0.5) : filtre classique haute qualite.
    float catmullRom(float x)
    {
        x = std::fabs(x);
        if (x < 1.0f)
        {
            return ((1.5f * x - 2.5f) * x) * x + 1.0f;
        }
        if (x < 2.0f)
        {
            return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
        }
        return 0.0f;
    }

    // Reconstruction : bicubique Catmull-Rom separable (passe horizontale puis verticale).
    m3gss::Image upscaleBicubic(const m3gss::Image& source, int dstWidth, int dstHeight)
    {
        m3gss::Image dst(dstWidth, dstHeight, source.channels);
        const int channels = source.channels;

        // Passe 1 : lignes de 'source' -> buffer flottant de largeur 'dstWidth'.
        std::vector<float> temp(static_cast<std::size_t>(dstWidth)
                                * static_cast<std::size_t>(source.height)
                                * static_cast<std::size_t>(channels));
        const float ratioX = static_cast<float>(source.width) / static_cast<float>(dstWidth);

        for (int y = 0; y < source.height; ++y)
        {
            const std::uint8_t* srcRow = source.row(y);
            float* tmpRow = temp.data()
                + static_cast<std::size_t>(y) * static_cast<std::size_t>(dstWidth)
                    * static_cast<std::size_t>(channels);

            for (int x = 0; x < dstWidth; ++x)
            {
                const float center = (static_cast<float>(x) + 0.5f) * ratioX - 0.5f;
                const int base = static_cast<int>(std::floor(center));

                float weights[4];
                int indices[4];
                float weightSum = 0.0f;

                for (int k = 0; k < 4; ++k)
                {
                    const int tap = base + k - 1;
                    weights[k] = catmullRom(center - static_cast<float>(tap));
                    indices[k] = std::min(std::max(tap, 0), source.width - 1);
                    weightSum += weights[k];
                }

                const float invSum = (weightSum != 0.0f) ? (1.0f / weightSum) : 0.0f;

                for (int c = 0; c < channels; ++c)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < 4; ++k)
                    {
                        sum += weights[k]
                            * static_cast<float>(srcRow[indices[k] * channels + c]);
                    }
                    tmpRow[x * channels + c] = sum * invSum;
                }
            }
        }

        // Passe 2 : colonnes du buffer flottant -> destination.
        const float ratioY = static_cast<float>(source.height) / static_cast<float>(dstHeight);

        for (int y = 0; y < dstHeight; ++y)
        {
            const float center = (static_cast<float>(y) + 0.5f) * ratioY - 0.5f;
            const int base = static_cast<int>(std::floor(center));

            float weights[4];
            int indices[4];
            float weightSum = 0.0f;

            for (int k = 0; k < 4; ++k)
            {
                const int tap = base + k - 1;
                weights[k] = catmullRom(center - static_cast<float>(tap));
                indices[k] = std::min(std::max(tap, 0), source.height - 1);
                weightSum += weights[k];
            }

            const float invSum = (weightSum != 0.0f) ? (1.0f / weightSum) : 0.0f;
            std::uint8_t* dstRow = dst.row(y);

            for (int x = 0; x < dstWidth; ++x)
            {
                for (int c = 0; c < channels; ++c)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < 4; ++k)
                    {
                        const float* srcCol = temp.data()
                            + static_cast<std::size_t>(indices[k])
                                * static_cast<std::size_t>(dstWidth)
                                * static_cast<std::size_t>(channels);
                        sum += weights[k] * srcCol[x * channels + c];
                    }
                    dstRow[x * channels + c] = m3gss::resampler::clampToByte(sum * invSum);
                }
            }
        }

        return dst;
    }
}

namespace m3gss
{
    std::string BaselineBicubicUpscaler::name() const
    {
        return "Baseline non-IA : reduction area + reconstruction bicubique (Catmull-Rom)";
    }

    UpscaleResult BaselineBicubicUpscaler::run(const Image& source, double downscaleFactor) const
    {
        UpscaleResult result;

        if (source.empty() || !(downscaleFactor > 0.0) || !(downscaleFactor < 1.0))
        {
            return result;
        }

        const int lowWidth = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(source.width) * downscaleFactor)));
        const int lowHeight = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(source.height) * downscaleFactor)));

        result.lowResolution = m3gss::resampler::downscaleArea(source, lowWidth, lowHeight);
        result.reconstruction = upscaleBicubic(result.lowResolution, source.width, source.height);

        return result;
    }
}
