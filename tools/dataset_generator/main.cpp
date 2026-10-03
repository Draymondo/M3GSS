#include <exception>
#include <iostream>
#include <string>

#include "dataset_generator.hpp"

namespace
{
    void printUsage(std::ostream& stream)
    {
        stream << "Usage : M3GSS_dataset_generator <dossier_source> <dossier_HR> <dossier_LR> [facteur]"
               << std::endl;
        stream << "  - PNG et JPEG accepts en entree ; sorties PNG (HR et LR, meme nom de fichier)."
               << std::endl;
        stream << "  - facteur entier >= 2 (defaut 2 : LR = source divisee par 2)." << std::endl;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4 && argc != 5)
    {
        std::cerr << "Nombre d'argument invalide." << std::endl;
        printUsage(std::cerr);
        return 1;
    }

    m3gss::tools::DatasetOptions options;
    options.sourceDir = argv[1];
    options.hrDir = argv[2];
    options.lrDir = argv[3];
    options.factor = 2;

    if (argc == 5)
    {
        try
        {
            std::size_t parsed = 0;
            const int value = std::stoi(argv[4], &parsed);
            if (parsed != std::string(argv[4]).size() || value < 2)
            {
                std::cerr << "Facteur invalide : '" << argv[4] << "' (entier >= 2 attendu)."
                          << std::endl;
                return 1;
            }
            options.factor = value;
        }
        catch (const std::exception&)
        {
            std::cerr << "Facteur invalide : '" << argv[4] << "' (entier >= 2 attendu)."
                      << std::endl;
            return 1;
        }
    }

    m3gss::tools::DatasetStats stats;
    std::string error;

    if (!m3gss::tools::generate(options, &stats, &error, &std::cout))
    {
        std::cerr << "Echec : " << error << std::endl;
        return 1;
    }

    std::cout << "--- Resume dataset ---" << std::endl;
    std::cout << "Images traitees   : " << stats.processed << std::endl;
    std::cout << "Images ignorees   : " << stats.ignored << std::endl;

    if (stats.processed > 0)
    {
        std::cout << "Dimensions HR     : " << stats.hrWidth << "x" << stats.hrHeight
                  << std::endl;
        std::cout << "Dimensions LR     : " << stats.lrWidth << "x" << stats.lrHeight
                  << " (facteur " << options.factor << ")" << std::endl;
    }
    else
    {
        std::cout << "Dimensions HR     : n/a" << std::endl;
        std::cout << "Dimensions LR     : n/a" << std::endl;
    }

    std::cout << "Temps total       : " << stats.totalMs << " ms" << std::endl;

    if (stats.processed > 0)
    {
        std::cout << "Temps moyen / img : "
                  << (stats.totalMs / static_cast<double>(stats.processed)) << " ms" << std::endl;
    }
    else
    {
        std::cout << "Temps moyen / img : n/a" << std::endl;
    }

    return 0;
}
