#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "upscale/BaselineBicubicUpscaler.hpp"
#include "metrics/QualityMetrics.hpp"

namespace
{
    void printUsage(std::ostream& stream)
    {
        stream << "Usage : M3GSS <entree.png|jpg> <sortie.png>" << std::endl;
        stream << "        M3GSS upscale <entree.png|jpg> <sortie.png> [facteur]" << std::endl;
    }

    // Conversion simple : lecture image (PNG/JPEG/...) puis ecriture PNG.
    int convertToPng(const std::string& inputPath, const std::string& outputPath)
    {
        std::string error;
        const m3gss::Image image = m3gss::io::loadImage(inputPath, &error);

        if (image.empty())
        {
            std::cerr << "Echec de lecture de '" << inputPath << "' : " << error << std::endl;
            return 1;
        }

        std::cout << "Charge : " << inputPath
                  << " (" << image.width << "x" << image.height
                  << ", " << image.channels << " canaux)" << std::endl;

        if (!m3gss::io::savePng(outputPath, image, &error))
        {
            std::cerr << "Echec d'ecriture de '" << outputPath << "' : " << error << std::endl;
            return 1;
        }

        std::cout << "Ecrit : " << outputPath << std::endl;
        return 0;
    }

    // Baseline non-IA : reduction volontaire puis reconstruction bicubique.
    int runUpscale(const std::string& inputPath, const std::string& outputPath, double factor)
    {
        if (!(factor > 0.0 && factor < 1.0))
        {
            std::cerr << "Facteur de reduction attendu dans ]0;1[ (obtenu : " << factor
                      << ")." << std::endl;
            return 1;
        }

        std::string error;
        const m3gss::Image source = m3gss::io::loadImage(inputPath, &error);

        if (source.empty())
        {
            std::cerr << "Echec de lecture de '" << inputPath << "' : " << error << std::endl;
            return 1;
        }

        const m3gss::BaselineBicubicUpscaler upscaler;

        const auto start = std::chrono::steady_clock::now();
        const m3gss::UpscaleResult result = upscaler.run(source, factor);
        const auto end = std::chrono::steady_clock::now();
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(end - start).count();

        if (result.reconstruction.empty()
            || result.reconstruction.width != source.width
            || result.reconstruction.height != source.height)
        {
            std::cerr << "Erreur interne : la reconstruction ne correspond pas a la resolution source."
                      << std::endl;
            return 1;
        }

        if (!m3gss::io::savePng(outputPath, result.reconstruction, &error))
        {
            std::cerr << "Echec d'ecriture de '" << outputPath << "' : " << error << std::endl;
            return 1;
        }

        std::cout << "--- Baseline upscaling (non-IA) ---" << std::endl;
        std::cout << "Algorithme       : " << upscaler.name() << std::endl;
        std::cout << "Source           : " << source.width << "x" << source.height << std::endl;
        std::cout << "Basse resolution : " << result.lowResolution.width << "x"
                  << result.lowResolution.height << " (facteur " << factor << ")" << std::endl;
        std::cout << "Reconstruite     : " << result.reconstruction.width << "x"
                  << result.reconstruction.height << std::endl;
        std::cout << "Canaux           : " << source.channels << std::endl;
        std::cout << "Temps traitement : " << elapsedMs << " ms" << std::endl;
        std::cout << "Sortie           : " << outputPath << std::endl;

        // Metriques de qualite : reference = image originale, candidate = reconstruction.
        m3gss::metrics::QualityMetrics quality;
        std::string metricsError;
        std::cout << "--- Qualite vs image originale ---" << std::endl;
        if (m3gss::metrics::compute(source, result.reconstruction, &quality, &metricsError))
        {
            std::cout << std::setprecision(6);
            std::cout << "MSE              : " << quality.mse << std::endl;
            if (quality.identical)
            {
                std::cout << "PSNR             : images identiques (MSE = 0)" << std::endl;
            }
            else
            {
                std::cout << "PSNR             : " << quality.psnr << " dB" << std::endl;
            }
            std::cout << "SSIM             : " << quality.ssim << std::endl;
        }
        else
        {
            std::cout << "Metriques        : indisponibles (" << metricsError << ")" << std::endl;
        }
        return 0;
    }
}

int main(int argc, char* argv[])
{
    if (argc >= 2 && std::string(argv[1]) == "upscale")
    {
        if (argc < 4 || argc > 5)
        {
            std::cerr << "Nombre d'arguments invalide pour la commande 'upscale'." << std::endl;
            printUsage(std::cerr);
            return 1;
        }

        double factor = 0.5;
        if (argc == 5)
        {
            try
            {
                std::size_t parsed = 0;
                factor = std::stod(argv[4], &parsed);
                if (parsed != std::string(argv[4]).size())
                {
                    throw std::invalid_argument("facteur incomplet");
                }
            }
            catch (const std::exception&)
            {
                std::cerr << "Facteur de reduction invalide : '" << argv[4] << "'." << std::endl;
                return 1;
            }
        }

        return runUpscale(argv[2], argv[3], factor);
    }

    if (argc == 3)
    {
        return convertToPng(argv[1], argv[2]);
    }

    if (argc == 1)
    {
        std::cout << "M3GSS v0.1.0 - Prototype initial" << std::endl;
        std::cout << "Moteur de reconstruction d'image en preparation." << std::endl;
        printUsage(std::cout);
        return 0;
    }

    std::cerr << "Arguments invalides." << std::endl;
    printUsage(std::cerr);
    return 1;
}
