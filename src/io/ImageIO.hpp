#pragma once

#include <string>

#include "core/Image.hpp"

namespace m3gss::io
{
    // Charge une image (PNG/JPEG/...) via stb_image.
    // Renvoie une image vide en cas d'echec ; 'error' recoit alors la raison.
    Image loadImage(const std::string& path, std::string* error = nullptr);

    // Ecrit une image au format PNG via stb_image_write.
    bool savePng(const std::string& path, const Image& image, std::string* error = nullptr);
}
