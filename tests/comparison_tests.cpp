// Tests de validation deterministes de l'outil de comparaison (M3GSS_compare).
// Executable autonome sans framework ni dependance externe, lance via CTest.
// Aucun alea : motifs construits par formule, dossiers temporaires recrees.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "comparison.hpp"
#include "core/Image.hpp"
#include "io/ImageIO.hpp"

namespace
{
    int g_failures = 0;

    void check(bool condition, const std::string& label)
    {
        if (condition)
        {
            std::cout << "[ok]    " << label << std::endl;
        }
        else
        {
            ++g_failures;
            std::cerr << "[ECHEC] " << label << std::endl;
        }
    }

    // Motif deterministe (aucune alea).
    m3gss::Image makePattern(int width, int height, int channels, int salt)
    {
        m3gss::Image image(width, height, channels);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                for (int c = 0; c < channels; ++c)
                {
                    const int value = (x * 7 + y * 13 + c * 29 + salt * 31) % 256;
                    image.pixel(x, y)[c] = static_cast<std::uint8_t>(value);
                }
            }
        }
        return image;
    }

    std::filesystem::path makeCase(const std::string& name)
    {
        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / "m3gss_comparison_tests" / name;
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        return root;
    }
}

int main()
{
    const int labelH = m3gss::tools::kLabelBandHeight;

    // --- 1. Chargement valide, LR, planche, dimensions, sauvegarde PNG ---
    {
        const std::filesystem::path root = makeCase("valid");
        const std::filesystem::path originalPath = root / "original.png";
        const std::filesystem::path reconPath = root / "reconstructed.png";

        const m3gss::Image original = makePattern(96, 64, 3, 1);
        const m3gss::Image reconstructed = makePattern(96, 64, 3, 2);
        check(m3gss::io::savePng(originalPath.string(), original, nullptr),
              "setup : original ecrit");
        check(m3gss::io::savePng(reconPath.string(), reconstructed, nullptr),
              "setup : reconstruction ecrite");

        m3gss::tools::ComparisonOptions options;
        options.originalPath = originalPath;
        options.reconstructedPath = reconPath;
        options.outputDir = root / "out";
        options.factor = 2;

        m3gss::tools::ComparisonStats stats;
        std::string error;
        const bool ok =
            m3gss::tools::generateComparison(options, &stats, &error, nullptr);
        check(ok, std::string("generation reussie") + (ok ? "" : " (erreur: " + error + ")"));
        if (!ok)
        {
            return 1;
        }

        check(stats.originalWidth == 96 && stats.originalHeight == 64,
              "chargement : dimensions original 96x64");
        check(stats.originalChannels == 3, "chargement : 3 canaux");
        check(stats.lrWidth == 48 && stats.lrHeight == 32,
              "generation LR : 48x32 (facteur 2)");
        check(stats.reconstructedWidth == 96 && stats.reconstructedHeight == 64,
              "chargement : dimensions reconstruction 96x64");

        // Planche = 3 panneaux (96x64) + 3 bandeaux de label.
        check(stats.boardWidth == 96 && stats.boardHeight == 3 * (labelH + 64),
              "planche : dimensions 96x246");
        // Crops : minDim/4 = 16 (pair, >= 8) -> 3 panneaux de 16.
        check(stats.cropSize == 16, "crops : cote 16");
        check(stats.cropBoardWidth == 48 && stats.cropBoardHeight == labelH + 16,
              "crops : dimensions 48x34");
        check(stats.totalMs >= 0.0, "temps de generation renseigne");

        // Sauvegarde PNG : les 4 fichiers existent et se rechargent aux dimensions attendues.
        const char* names[4] = {
            "comparison.png", "comparison_face.png", "comparison_texture.png",
            "comparison_edges.png"};
        const int expectedWidth[4] = {96, 48, 48, 48};
        const int expectedHeight[4] = {246, 34, 34, 34};
        for (int i = 0; i < 4; ++i)
        {
            const std::filesystem::path path = options.outputDir / names[i];
            check(std::filesystem::exists(path), std::string("existe : ") + names[i]);
            const m3gss::Image loaded = m3gss::io::loadImage(path.string(), nullptr);
            check(!loaded.empty() && loaded.width == expectedWidth[i]
                      && loaded.height == expectedHeight[i],
                  std::string("dimensions PNG : ") + names[i]);
        }
    }

    // --- 2. Fichier invalide -> erreur explicite ---
    {
        const std::filesystem::path root = makeCase("invalid");
        const std::filesystem::path originalPath = root / "broken.png";
        {
            std::ofstream broken(originalPath, std::ios::binary);
            broken << "ceci n'est pas une image";
        }
        const std::filesystem::path reconPath = root / "reconstructed.png";
        check(m3gss::io::savePng(reconPath.string(), makePattern(96, 64, 3, 3), nullptr),
              "setup : reconstruction valide");

        m3gss::tools::ComparisonOptions options;
        options.originalPath = originalPath;
        options.reconstructedPath = reconPath;
        options.outputDir = root / "out";

        m3gss::tools::ComparisonStats stats;
        std::string error;
        check(!m3gss::tools::generateComparison(options, &stats, &error, nullptr)
              && error.find("invalide") != std::string::npos,
              "fichier invalide : erreur explicite");
        check(!std::filesystem::exists(options.outputDir / "comparison.png"),
              "fichier invalide : aucune planche ecrite");
    }

    // --- 3. Fichier absent -> erreur explicite ---
    {
        const std::filesystem::path root = makeCase("missing");
        const std::filesystem::path reconPath = root / "reconstructed.png";
        check(m3gss::io::savePng(reconPath.string(), makePattern(96, 64, 3, 4), nullptr),
              "setup : reconstruction valide");

        m3gss::tools::ComparisonOptions options;
        options.originalPath = root / "manquant.png";
        options.reconstructedPath = reconPath;
        options.outputDir = root / "out";

        m3gss::tools::ComparisonStats stats;
        std::string error;
        check(!m3gss::tools::generateComparison(options, &stats, &error, nullptr)
              && error.find("absent") != std::string::npos,
              "fichier absent : erreur explicite");
    }

    // --- 4. Dimensions incompatibles -> erreur explicite ---
    {
        const std::filesystem::path root = makeCase("incompatible");
        const std::filesystem::path originalPath = root / "original.png";
        const std::filesystem::path reconPath = root / "reconstructed.png";
        check(m3gss::io::savePng(originalPath.string(), makePattern(96, 64, 3, 5), nullptr),
              "setup : original 96x64");
        check(m3gss::io::savePng(reconPath.string(), makePattern(64, 64, 3, 6), nullptr),
              "setup : reconstruction 64x64");

        m3gss::tools::ComparisonOptions options;
        options.originalPath = originalPath;
        options.reconstructedPath = reconPath;
        options.outputDir = root / "out";

        m3gss::tools::ComparisonStats stats;
        std::string error;
        check(!m3gss::tools::generateComparison(options, &stats, &error, nullptr)
              && error.find("incompatibles") != std::string::npos,
              "dimensions incompatibles : erreur explicite");
    }

    // Nettoyage de la racine de tests.
    std::error_code cleanupCode;
    std::filesystem::remove_all(
        std::filesystem::temp_directory_path() / "m3gss_comparison_tests", cleanupCode);

    if (g_failures == 0)
    {
        std::cout << "Tous les tests de l'outil de comparaison sont passes." << std::endl;
        return 0;
    }

    std::cerr << g_failures << " test(s) en echec." << std::endl;
    return 1;
}
