#include "dataset_generator.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ostream>
#include <set>
#include <vector>

#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "upscale/Resampler.hpp"

namespace
{
    std::string toLowerCopy(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    // Extensions acceptees : PNG et JPEG, insensibles a la casse.
    bool hasAcceptedExtension(const std::filesystem::path& file)
    {
        const std::string ext = toLowerCopy(file.extension().string());
        return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
    }
}

namespace m3gss::tools
{
    bool processFile(const std::filesystem::path& file, const DatasetOptions& options,
                     ProcessResult* result, std::string* reason)
    {
        if (reason != nullptr)
        {
            reason->clear();
        }
        if (result != nullptr)
        {
            *result = ProcessResult{};
        }

        std::string loadError;
        const m3gss::Image hr = m3gss::io::loadImage(file.string(), &loadError);
        if (hr.empty())
        {
            if (reason != nullptr)
            {
                *reason = "fichier image invalide (" + loadError + ")";
            }
            return false;
        }

        if (hr.width < options.factor || hr.height < options.factor)
        {
            if (reason != nullptr)
            {
                *reason = "image trop petite (" + std::to_string(hr.width) + "x"
                    + std::to_string(hr.height) + ") pour un facteur "
                    + std::to_string(options.factor);
            }
            return false;
        }

        // Meme reduction que la baseline : code partage via Resampler.
        std::string reduceError;
        const m3gss::Image lr =
            m3gss::resampler::reduceByIntegerFactor(hr, options.factor, &reduceError);
        if (lr.empty())
        {
            if (reason != nullptr)
            {
                *reason = "reduction impossible (" + reduceError + ")";
            }
            return false;
        }

        // Verification des dimensions LR : meme formule que la baseline.
        const double ratio = 1.0 / static_cast<double>(options.factor);
        const int expectedWidth = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(hr.width) * ratio)));
        const int expectedHeight = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(hr.height) * ratio)));
        if (lr.width != expectedWidth || lr.height != expectedHeight)
        {
            if (reason != nullptr)
            {
                *reason = "dimensions LR incoherentes (" + std::to_string(lr.width) + "x"
                    + std::to_string(lr.height) + " au lieu de "
                    + std::to_string(expectedWidth) + "x" + std::to_string(expectedHeight) + ")";
            }
            return false;
        }

        // Correspondance deterministe : meme nom de fichier dans hrDir et lrDir.
        const std::string stem = file.stem().string();
        const std::filesystem::path hrPath = options.hrDir / (stem + ".png");
        const std::filesystem::path lrPath = options.lrDir / (stem + ".png");

        std::string writeError;
        if (!m3gss::io::savePng(hrPath.string(), hr, &writeError))
        {
            if (reason != nullptr)
            {
                *reason = "ecriture HR impossible (" + writeError + ")";
            }
            return false;
        }
        if (!m3gss::io::savePng(lrPath.string(), lr, &writeError))
        {
            if (reason != nullptr)
            {
                *reason = "ecriture LR impossible (" + writeError + ")";
            }
            return false;
        }

        if (result != nullptr)
        {
            result->hrWidth = hr.width;
            result->hrHeight = hr.height;
            result->lrWidth = lr.width;
            result->lrHeight = lr.height;
        }
        return true;
    }

    bool generate(const DatasetOptions& options, DatasetStats* stats,
                  std::string* error, std::ostream* log)
    {
        if (error != nullptr)
        {
            error->clear();
        }

        if (stats == nullptr)
        {
            if (error != nullptr)
            {
                *error = "pointeur 'stats' nul";
            }
            return false;
        }
        *stats = DatasetStats{};

        if (options.factor < 2)
        {
            if (error != nullptr)
            {
                *error = "facteur entier >= 2 attendu";
            }
            return false;
        }

        std::error_code ec;
        if (!std::filesystem::is_directory(options.sourceDir, ec))
        {
            if (error != nullptr)
            {
                *error = "dossier source introuvable : " + options.sourceDir.string();
            }
            return false;
        }

        if (options.hrDir == options.sourceDir || options.lrDir == options.sourceDir
            || options.hrDir == options.lrDir)
        {
            if (error != nullptr)
            {
                *error = "les dossiers source, HR et LR doivent etre distincts";
            }
            return false;
        }

        std::filesystem::create_directories(options.hrDir, ec);
        if (ec)
        {
            if (error != nullptr)
            {
                *error = "impossible de creer le dossier HR : " + ec.message();
            }
            return false;
        }
        std::filesystem::create_directories(options.lrDir, ec);
        if (ec)
        {
            if (error != nullptr)
            {
                *error = "impossible de creer le dossier LR : " + ec.message();
            }
            return false;
        }

        // Collecte des candidats puis tri lexical : parcours deterministe, non recursif.
        std::vector<std::filesystem::path> candidates;
        try
        {
            for (const auto& entry : std::filesystem::directory_iterator(options.sourceDir))
            {
                if (entry.is_regular_file() && hasAcceptedExtension(entry.path()))
                {
                    candidates.push_back(entry.path());
                }
            }
        }
        catch (const std::filesystem::filesystem_error& ex)
        {
            if (error != nullptr)
            {
                *error = "lecture du dossier source impossible : " + std::string(ex.what());
            }
            return false;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const std::filesystem::path& a, const std::filesystem::path& b)
                  { return a.generic_string() < b.generic_string(); });

        const auto start = std::chrono::steady_clock::now();
        std::set<std::string> usedStems; // stems deja ecrits, insensibles a la casse (Windows)

        for (const std::filesystem::path& file : candidates)
        {
            const std::string stemKey = toLowerCopy(file.stem().string());

            if (usedStems.find(stemKey) != usedStems.end())
            {
                ++stats->ignored;
                if (log != nullptr)
                {
                    *log << "Ignore : " << file.filename().string()
                         << " (collision de nom HR avec un fichier precedent)" << std::endl;
                }
                continue;
            }

            ProcessResult result;
            std::string reason;
            if (processFile(file, options, &result, &reason))
            {
                usedStems.insert(stemKey);
                ++stats->processed;
                stats->hrWidth = result.hrWidth;
                stats->hrHeight = result.hrHeight;
                stats->lrWidth = result.lrWidth;
                stats->lrHeight = result.lrHeight;

                if (log != nullptr)
                {
                    *log << "OK : " << file.filename().string()
                         << " -> HR " << result.hrWidth << "x" << result.hrHeight
                         << " + LR " << result.lrWidth << "x" << result.lrHeight << std::endl;
                }
            }
            else
            {
                ++stats->ignored;
                if (log != nullptr)
                {
                    *log << "Ignore : " << file.filename().string()
                         << " (" << reason << ")" << std::endl;
                }
            }
        }

        const auto end = std::chrono::steady_clock::now();
        stats->totalMs = std::chrono::duration<double, std::milli>(end - start).count();

        return true;
    }
}
