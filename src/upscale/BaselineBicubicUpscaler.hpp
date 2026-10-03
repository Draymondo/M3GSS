#pragma once

#include "upscale/Upscaler.hpp"

namespace m3gss
{
    // Baseline non-IA, sans CUDA, sans dependance externe :
    //  1. reduction volontaire par moyennage de surface (area averaging) ;
    //  2. reconstruction bicubique Catmull-Rom (filtre classique haute qualite,
    //     applique de facon separable horizontalement puis verticalement).
    class BaselineBicubicUpscaler final : public Upscaler
    {
    public:
        std::string name() const override;
        UpscaleResult run(const Image& source, double downscaleFactor) const override;
    };
}
