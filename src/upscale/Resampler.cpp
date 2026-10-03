#include "upscale/Resampler.hpp"

#include <algorithm>
#include <cmath>

namespace m3gss::resampler
{
    std::uint8_t clampToByte(float value)
    {
        const float rounded = value + 0.5f;
        if (rounded <= 0.0f)
        {
            return 0;
        }
        if (rounded >= 255.0f)
        {
            return 255;
        }
        return static_cast<std::uint8_t>(rounded);
    }

    Image downscaleArea(const Image& source, int dstWidth, int dstHeight)
    {
        if (source.empty() || dstWidth < 1 || dstHeight < 1)
        {
            return {};
        }

        Image dst(dstWidth, dstHeight, source.channels);

        const float xr = static_cast<float>(source.width) / static_cast<float>(dstWidth);
        const float yr = static_cast<float>(source.height) / static_cast<float>(dstHeight);

        std::vector<float> accumulator(static_cast<std::size_t>(source.channels), 0.0f);

        for (int oy = 0; oy < dstHeight; ++oy)
        {
            const float y0 = static_cast<float>(oy) * yr;
            const float y1 = static_cast<float>(oy + 1) * yr;
            const int iy0 = std::max(0, static_cast<int>(y0));
            const int iy1 = std::min(source.height - 1,
                                     std::max(iy0, static_cast<int>(std::ceil(y1)) - 1));

            for (int ox = 0; ox < dstWidth; ++ox)
            {
                const float x0 = static_cast<float>(ox) * xr;
                const float x1 = static_cast<float>(ox + 1) * xr;
                const int ix0 = std::max(0, static_cast<int>(x0));
                const int ix1 = std::min(source.width - 1,
                                         std::max(ix0, static_cast<int>(std::ceil(x1)) - 1));

                std::fill(accumulator.begin(), accumulator.end(), 0.0f);
                float totalArea = 0.0f;

                for (int iy = iy0; iy <= iy1; ++iy)
                {
                    const float overlapY = std::min(y1, static_cast<float>(iy) + 1.0f)
                        - std::max(y0, static_cast<float>(iy));
                    if (overlapY <= 0.0f)
                    {
                        continue;
                    }

                    for (int ix = ix0; ix <= ix1; ++ix)
                    {
                        const float overlapX = std::min(x1, static_cast<float>(ix) + 1.0f)
                            - std::max(x0, static_cast<float>(ix));
                        if (overlapX <= 0.0f)
                        {
                            continue;
                        }

                        const float weight = overlapX * overlapY;
                        totalArea += weight;

                        const std::uint8_t* src = source.pixel(ix, iy);
                        for (int c = 0; c < source.channels; ++c)
                        {
                            accumulator[static_cast<std::size_t>(c)] +=
                                weight * static_cast<float>(src[c]);
                        }
                    }
                }

                std::uint8_t* out = dst.pixel(ox, oy);
                const float invArea = (totalArea > 0.0f) ? (1.0f / totalArea) : 0.0f;
                for (int c = 0; c < dst.channels; ++c)
                {
                    out[c] = clampToByte(accumulator[static_cast<std::size_t>(c)] * invArea);
                }
            }
        }

        return dst;
    }

    Image reduceByIntegerFactor(const Image& source, int factor, std::string* error)
    {
        if (error != nullptr)
        {
            error->clear();
        }

        if (factor < 2)
        {
            if (error != nullptr)
            {
                *error = "facteur entier >= 2 attendu";
            }
            return {};
        }

        if (source.empty())
        {
            if (error != nullptr)
            {
                *error = "image vide";
            }
            return {};
        }

        // Meme formule que BaselineBicubicUpscaler (downscaleFactor = 1 / factor).
        const double ratio = 1.0 / static_cast<double>(factor);
        const int lowWidth = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(source.width) * ratio)));
        const int lowHeight = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(source.height) * ratio)));

        Image low = downscaleArea(source, lowWidth, lowHeight);

        if (low.empty() || low.width != lowWidth || low.height != lowHeight)
        {
            if (error != nullptr)
            {
                *error = "dimensions LR incoherentes apres reduction";
            }
            return {};
        }

        return low;
    }
}
