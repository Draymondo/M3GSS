#pragma once

#include <cstdint>
#include <vector>

namespace m3gss
{
    // Image 8 bits, pixels contigus ligne par ligne (1 a 4 canaux).
    struct Image
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        std::vector<std::uint8_t> pixels;

        Image() = default;

        Image(int imageWidth, int imageHeight, int channelCount)
            : width(imageWidth)
            , height(imageHeight)
            , channels(channelCount)
            , pixels(static_cast<std::size_t>(imageWidth) * static_cast<std::size_t>(imageHeight)
                         * static_cast<std::size_t>(channelCount),
                     0)
        {
        }

        bool empty() const { return pixels.empty(); }

        int stride() const { return width * channels; }

        std::uint8_t* row(int y)
        {
            return pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride());
        }

        const std::uint8_t* row(int y) const
        {
            return pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride());
        }

        std::uint8_t* pixel(int x, int y)
        {
            return row(y) + static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);
        }

        const std::uint8_t* pixel(int x, int y) const
        {
            return row(y) + static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);
        }
    };
}
