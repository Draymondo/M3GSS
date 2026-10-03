#pragma once

#include <string>

#include "core/Image.hpp"

namespace m3gss
{
    // Resultat d'un pipeline d'upscaling :
    //  - 'lowResolution'  : reduction volontaire de la source ;
    //  - 'reconstruction' : image reconstruite a la resolution de la source.
    struct UpscaleResult
    {
        Image lowResolution;
        Image reconstruction;
    };

    // Contrat commun a toutes les strategies d'upscaling.
    // La baseline actuelle (BaselineBicubicUpscaler) et le futur algorithme
    // M3GSS implementeront cette meme interface, sans changer l'appelant.
    class Upscaler
    {
    public:
        virtual ~Upscaler() = default;

        // Libelle de l'algorithme, affiche dans les rapports.
        virtual std::string name() const = 0;

        // Reduce volontairement 'source' puis reconstruit vers sa resolution d'origine.
        // 'downscaleFactor' est un facteur ]0;1[ (0.5 = resolution divisee par 2).
        virtual UpscaleResult run(const Image& source, double downscaleFactor) const = 0;
    };
}
