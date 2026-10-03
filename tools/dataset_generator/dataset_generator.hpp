#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>

namespace m3gss::tools
{
    // Options du generateur de dataset.
    //  HR = image haute resolution (PNG identique aux pixels decodes de la source),
    //  LR = meme image reduite par un facteur entier (x2 par defaut).
    struct DatasetOptions
    {
        std::filesystem::path sourceDir;
        std::filesystem::path hrDir;
        std::filesystem::path lrDir;
        int factor = 2; // x2 aujourd'hui ; x3, x4 et autres degradations a venir
    };

    // Dimensions observees sur une image traitee avec succes.
    struct ProcessResult
    {
        int hrWidth = 0;
        int hrHeight = 0;
        int lrWidth = 0;
        int lrHeight = 0;
    };

    // Statistiques du parcours d'un dossier.
    struct DatasetStats
    {
        int processed = 0;
        int ignored = 0;
        int hrWidth = 0;  // dimensions HR de la derniere image valide traitee
        int hrHeight = 0;
        int lrWidth = 0;  // dimensions LR de la derniere image valide traitee
        int lrHeight = 0;
        double totalMs = 0.0;
    };

    // Traite un fichier image unique :
    //  - lecture PNG/JPEG via stb_image ;
    //  - refuse les images plus petites que le facteur ;
    //  - ecrit HR et LR sous le MEME nom de fichier (correspondance deterministe) ;
    //  - utilise la reduction partagee avec la baseline (Resampler) ;
    //  - verifie les dimensions LR ;
    //  - ignore proprement les fichiers invalides (raison dans 'reason').
    // Renvoie false si le fichier est ignore ; 'result' n'est rempli qu'en succes.
    bool processFile(const std::filesystem::path& file, const DatasetOptions& options,
                     ProcessResult* result, std::string* reason);

    // Parcourt sourceDir (tri lexical deterministe, non recursif) et remplit
    // hrDir / lrDir. Les extensions non supportees sont sautes silencieusement
    // (elles ne sont pas des candidats) ; les candidats invalides sont comptes
    // dans stats->ignored. 'log' recoit une ligne par fichier si non nul.
    // Renvoie false et detaille 'error' si les dossiers sont inutilisables.
    bool generate(const DatasetOptions& options, DatasetStats* stats,
                  std::string* error, std::ostream* log = nullptr);
}
