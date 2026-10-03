// Tests de validation deterministes du generateur de dataset.
// Executable autonome sans framework ni dependance externe, lance via CTest.
// Aucun generateur aleatoire : motifs construits par formule, dossiers temporaires
// recrees depuis zero a chaque execution.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "core/Image.hpp"
#include "dataset_generator.hpp"
#include "io/ImageIO.hpp"
#include "upscale/Resampler.hpp"
#include "stb_image_write.h"

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

    // Racine de test fraiche : <temp>/m3gss_dataset_tests/<name>/src cree.
    std::filesystem::path makeCase(const std::string& name)
    {
        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / "m3gss_dataset_tests" / name;
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "src");
        return root;
    }

    m3gss::tools::DatasetOptions optionsFor(const std::filesystem::path& root)
    {
        m3gss::tools::DatasetOptions options;
        options.sourceDir = root / "src";
        options.hrDir = root / "hr";
        options.lrDir = root / "lr";
        options.factor = 2;
        return options;
    }
}

int main()
{
    // --- 1. Image valide : 64x64x3 -> HR identique, LR 32x32 ---
    {
        const std::filesystem::path root = makeCase("valid");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        const m3gss::Image source = makePattern(64, 64, 3, 1);
        check(m3gss::io::savePng((options.sourceDir / "valid.png").string(), source, nullptr),
              "valide : ecriture source");

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr), "valide : generate reussi");
        check(stats.processed == 1, "valide : 1 image traitee");
        check(stats.ignored == 0, "valide : 0 image ignoree");
        check(stats.hrWidth == 64 && stats.hrHeight == 64, "valide : dimensions HR 64x64");
        check(stats.lrWidth == 32 && stats.lrHeight == 32, "valide : dimensions LR 32x32");

        const m3gss::Image hr =
            m3gss::io::loadImage((options.hrDir / "valid.png").string(), nullptr);
        check(!hr.empty() && hr.width == 64 && hr.height == 64
              && hr.pixels == source.pixels,
              "valide : HR pixel-identique a la source");

        const m3gss::Image lr =
            m3gss::io::loadImage((options.lrDir / "valid.png").string(), nullptr);
        check(!lr.empty() && lr.width == 32 && lr.height == 32,
              "valide : LR ecrite en 32x32");
    }

    // --- 2. Dimensions impaires : 127x95 -> LR 64x48 (arrondi lround) ---
    {
        const std::filesystem::path root = makeCase("odd");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        const m3gss::Image source = makePattern(127, 95, 1, 2);
        check(m3gss::io::savePng((options.sourceDir / "odd.png").string(), source, nullptr),
              "impaires : ecriture source");

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr),
              "impaires : generate reussi");
        check(stats.processed == 1, "impaires : 1 image traitee");
        check(stats.hrWidth == 127 && stats.hrHeight == 95, "impaires : HR 127x95");
        check(stats.lrWidth == 64 && stats.lrHeight == 48, "impaires : LR 64x48");

        const m3gss::Image lr =
            m3gss::io::loadImage((options.lrDir / "odd.png").string(), nullptr);
        check(!lr.empty() && lr.width == 64 && lr.height == 48,
              "impaires : LR relue en 64x48");
    }

    // --- 3. Image trop petite : 1x1 pour un facteur 2 -> ignoree ---
    {
        const std::filesystem::path root = makeCase("tiny");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        const m3gss::Image source = makePattern(1, 1, 1, 2);
        check(m3gss::io::savePng((options.sourceDir / "tiny.png").string(), source, nullptr),
              "trop petite : ecriture source");

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr),
              "trop petite : generate reussi (sans erreur fatale)");
        check(stats.processed == 0, "trop petite : 0 image traitee");
        check(stats.ignored == 1, "trop petite : 1 image ignoree");

        m3gss::tools::ProcessResult result;
        std::string reason;
        check(!m3gss::tools::processFile(options.sourceDir / "tiny.png", options,
                                         &result, &reason)
              && reason.find("trop petite") != std::string::npos,
              "trop petite : raison explicite");
        check(!std::filesystem::exists(options.hrDir / "tiny.png"),
              "trop petite : aucun fichier HR ecrit");
    }

    // --- 4. Fichier invalide : contenu non image -> ignore proprement ---
    {
        const std::filesystem::path root = makeCase("invalid");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        {
            std::ofstream broken(options.sourceDir / "broken.png", std::ios::binary);
            broken << "ceci n'est pas une image PNG";
        }

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr),
              "invalide : generate reussi (sans erreur fatale)");
        check(stats.processed == 0, "invalide : 0 image traitee");
        check(stats.ignored == 1, "invalide : 1 image ignoree");

        m3gss::tools::ProcessResult result;
        std::string reason;
        check(!m3gss::tools::processFile(options.sourceDir / "broken.png", options,
                                         &result, &reason)
              && reason.find("invalide") != std::string::npos,
              "invalide : raison explicite");
        check(!std::filesystem::exists(options.hrDir / "broken.png"),
              "invalide : aucun fichier HR ecrit");
    }

    // --- 5. Correspondance HR/LR : LR == reduction partagee de la baseline ---
    {
        const std::filesystem::path root = makeCase("pair");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        const m3gss::Image source = makePattern(96, 64, 4, 3);
        check(m3gss::io::savePng((options.sourceDir / "pair.png").string(), source, nullptr),
              "paire : ecriture source");

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr), "paire : generate reussi");
        check(stats.processed == 1, "paire : 1 image traitee");

        const m3gss::Image hr =
            m3gss::io::loadImage((options.hrDir / "pair.png").string(), nullptr);
        const m3gss::Image lr =
            m3gss::io::loadImage((options.lrDir / "pair.png").string(), nullptr);
        check(!hr.empty() && hr.pixels == source.pixels,
              "paire : HR pixel-identique a la source");
        check(!lr.empty() && lr.width == 48 && lr.height == 32,
              "paire : dimensions LR 48x32");

        std::string reduceError;
        const m3gss::Image expected =
            m3gss::resampler::reduceByIntegerFactor(hr, 2, &reduceError);
        check(!expected.empty() && !lr.empty()
              && expected.width == lr.width && expected.height == lr.height
              && expected.pixels == lr.pixels,
              "paire : LR == reduceByIntegerFactor(HR) pixel par pixel");
    }

    // --- 6. JPEG accepte en entree ---
    {
        const std::filesystem::path root = makeCase("jpeg");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        const m3gss::Image source = makePattern(48, 48, 3, 4);
        const std::filesystem::path jpegPath = options.sourceDir / "photo.jpg";
        check(stbi_write_jpg(jpegPath.string().c_str(), 48, 48, 3,
                             source.pixels.data(), 90) != 0,
              "jpeg : ecriture source JPEG");

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr), "jpeg : generate reussi");
        check(stats.processed == 1, "jpeg : 1 image traitee");
        check(stats.ignored == 0, "jpeg : 0 image ignoree");
        check(stats.hrWidth == 48 && stats.hrHeight == 48, "jpeg : dimensions HR 48x48");
        check(std::filesystem::exists(options.hrDir / "photo.png")
              && std::filesystem::exists(options.lrDir / "photo.png"),
              "jpeg : sorties HR/LR en PNG avec meme nom de stem");
    }

    // --- 7. Extension non candidate sautee + dossier source manquant ---
    {
        const std::filesystem::path root = makeCase("misc");
        const m3gss::tools::DatasetOptions options = optionsFor(root);
        {
            std::ofstream note(options.sourceDir / "notes.txt");
            note << "pas une image candidate";
        }

        m3gss::tools::DatasetStats stats;
        std::string error;
        check(m3gss::tools::generate(options, &stats, &error, nullptr),
              "divers : generate reussi");
        check(stats.processed == 0 && stats.ignored == 0,
              "divers : extension non candidate sautee silencieusement");

        m3gss::tools::DatasetStats missingStats;
        std::string missingError;
        m3gss::tools::DatasetOptions missingOptions = options;
        missingOptions.sourceDir = root / "absent";
        check(!m3gss::tools::generate(missingOptions, &missingStats, &missingError, nullptr)
              && !missingError.empty(),
              "divers : dossier source manquant -> erreur explicite");
    }

    // Nettoyage de la racine de tests.
    std::error_code cleanupCode;
    std::filesystem::remove_all(
        std::filesystem::temp_directory_path() / "m3gss_dataset_tests", cleanupCode);

    if (g_failures == 0)
    {
        std::cout << "Tous les tests du generateur de dataset sont passes." << std::endl;
        return 0;
    }

    std::cerr << g_failures << " test(s) en echec." << std::endl;
    return 1;
}
