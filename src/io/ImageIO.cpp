#include "ImageIO.hpp"

#include <cstring>

#include "stb_image.h"
#include "stb_image_write.h"

namespace m3gss::io
{
    Image loadImage(const std::string& path, std::string* error)
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        std::uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 0);

        if (data == nullptr)
        {
            if (error != nullptr)
            {
                const char* reason = stbi_failure_reason();
                *error = (reason != nullptr) ? reason : "erreur inconnue";
            }
            return {};
        }

        Image image(width, height, channels);
        std::memcpy(image.pixels.data(), data, image.pixels.size());
        stbi_image_free(data);

        if (error != nullptr)
        {
            error->clear();
        }
        return image;
    }

    bool savePng(const std::string& path, const Image& image, std::string* error)
    {
        if (image.empty())
        {
            if (error != nullptr)
            {
                *error = "image vide";
            }
            return false;
        }

        const int ok = stbi_write_png(path.c_str(), image.width, image.height, image.channels,
                                      image.pixels.data(), image.stride());

        if (ok == 0)
        {
            if (error != nullptr)
            {
                *error = "echec d'ecriture PNG";
            }
            return false;
        }

        if (error != nullptr)
        {
            error->clear();
        }
        return true;
    }
}
