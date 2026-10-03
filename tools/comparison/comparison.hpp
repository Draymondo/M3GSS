#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>

#include "core/Image.hpp"

namespace m3gss::tools
{
    // Hauteur du bandeau de label au-dessus de chaque panneau.
    // Exposee pour que les tests puissent verifier les dimensions des planches.
    constexpr int kLabelBandHeight = 18;

    struct ComparisonOptions
    {
        std::filesystem::path originalPath;
        std::filesystem::path reconstructedPath;
        std::filesystem::path outputDir;
        int factor = 2; // LR recalculee a 50 % : meme reduction que la baseline
    };

    struct ComparisonStats
    {
        int originalWidth = 0;
        int originalHeight = 0;
        int originalChannels = 0;   // canaux de l'original charge
        int lrWidth = 0;
        int lrHeight = 0;
        int reconstructedWidth = 0;
        int reconstructedHeight = 0;
        int boardWidth = 0;         // planche comparison.png
        int boardHeight = 0;
        int cropSize = 0;           // cote d'un panneau de crop
        int cropBoardWidth = 0;     // planche d'un crop (3 panneaux)
        int cropBoardHeight = 0;
        double totalMs = 0.0;
    };

    // Charge l'original et la reconstruction, verifie leurs dimensions, recalcule
    // la LR a 50 % avec le code partage de Resampler (aucun second algorithme de
    // reconstruction), puis ecrit dans outputDir :
    //   comparison.png       : planche Original | Low Resolution | Baseline Bicubic
    //   comparison_face.png  : 3 crops alignes sur exactement la meme zone
    //   comparison_texture.png
    //   comparison_edges.png
    // Le panneau Low Resolution est agrandi en plus-proche-voisin UNIQUEMENT pour
    // l'affichage de comparaison.
    // Renvoie false et detaille 'error' en cas d'echec : fichier absent, image
    // invalide, dimensions incompatibles, dossier ou ecriture PNG impossibles.
    bool generateComparison(const ComparisonOptions& options, ComparisonStats* stats,
                            std::string* error, std::ostream* log = nullptr);
}
