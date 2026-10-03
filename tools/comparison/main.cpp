#include <exception>
#include <iostream>
#include <string>

#include "comparison.hpp"

namespace
{
    void printUsage(std::ostream& stream)
    {
        stream << "Usage : M3GSS_compare <original> <reconstructed> <output_directory>"
               << std::endl;
        stream << "  Genere comparison.png + 3 crops (face, texture, edges) pour la"
               << std::endl;
        stream << "  validation visuelle de la baseline bicubique." << std::endl;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Nombre d'argument invalide." << std::endl;
        printUsage(std::cerr);
        return 1;
    }

    m3gss::tools::ComparisonOptions options;
    options.originalPath = argv[1];
    options.reconstructedPath = argv[2];
    options.outputDir = argv[3];
    options.factor = 2; // LR a 50 %

    m3gss::tools::ComparisonStats stats;
    std::string error;

    if (!m3gss::tools::generateComparison(options, &stats, &error, &std::cout))
    {
        std::cerr << "Echec : " << error << std::endl;
        return 1;
    }

    std::cout << "--- Comparaison visuelle (baseline) ---" << std::endl;
    std::cout << "Original          : " << stats.originalWidth << "x" << stats.originalHeight
              << " (" << stats.originalChannels << " canaux)" << std::endl;
    std::cout << "Low resolution    : " << stats.lrWidth << "x" << stats.lrHeight
              << " (facteur " << options.factor << ")" << std::endl;
    std::cout << "Reconstruction    : " << stats.reconstructedWidth << "x"
              << stats.reconstructedHeight << std::endl;
    std::cout << "Planche           : " << stats.boardWidth << "x" << stats.boardHeight
              << std::endl;
    std::cout << "Crops             : " << stats.cropBoardWidth << "x"
              << stats.cropBoardHeight << " (x3, panneau " << stats.cropSize << "x"
              << stats.cropSize << ")" << std::endl;
    std::cout << "Dossier de sortie : " << options.outputDir.string() << std::endl;
    std::cout << "Temps de generation : " << stats.totalMs << " ms" << std::endl;

    return 0;
}
