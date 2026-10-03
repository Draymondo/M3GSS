#include "comparison.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ostream>

#include "io/ImageIO.hpp"
#include "upscale/Resampler.hpp"

namespace
{
    // Police 5x7 embarquee (aucune dependance externe).
    // Index : A-Z = 0..25, espace = 26. Bits 4..0 = pixels de gauche a droite,
    // 7 octets par glyphe (une ligne par octet).
    const std::uint8_t kFont5x7[27][7] = {
        {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // A
        {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}, // B
        {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}, // C
        {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}, // D
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}, // E
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}, // F
        {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}, // G
        {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // H
        {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}, // I
        {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}, // J
        {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}, // K
        {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}, // L
        {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}, // M
        {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}, // N
        {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // O
        {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}, // P
        {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}, // Q
        {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}, // R
        {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}, // S
        {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, // T
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // U
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}, // V
        {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}, // W
        {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}, // X
        {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}, // Y
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}, // Z
        {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // espace
    };

    int glyphIndex(char c)
    {
        if (c >= 'a' && c <= 'z')
        {
            c = static_cast<char>(c - 'a' + 'A');
        }
        if (c >= 'A' && c <= 'Z')
        {
            return c - 'A';
        }
        return 26; // espace / caractere inconnu
    }

    int textWidth(const std::string& text, int scale)
    {
        if (text.empty())
        {
            return 0;
        }
        return static_cast<int>(text.size()) * 6 * scale - scale;
    }

    void fillRect(m3gss::Image& dst, int x, int y, int width, int height,
                  std::uint8_t r, std::uint8_t g, std::uint8_t b)
    {
        const int x0 = std::max(0, x);
        const int y0 = std::max(0, y);
        const int x1 = std::min(dst.width, x + width);
        const int y1 = std::min(dst.height, y + height);
        for (int yy = y0; yy < y1; ++yy)
        {
            for (int xx = x0; xx < x1; ++xx)
            {
                std::uint8_t* p = dst.pixel(xx, yy);
                p[0] = r;
                p[1] = g;
                p[2] = b;
            }
        }
    }

    void drawText(m3gss::Image& dst, int x, int y, const std::string& text, int scale,
                  std::uint8_t r, std::uint8_t g, std::uint8_t b)
    {
        int cursor = x;
        for (const char c : text)
        {
            const std::uint8_t* glyph = kFont5x7[glyphIndex(c)];
            for (int row = 0; row < 7; ++row)
            {
                const std::uint8_t bits = glyph[row];
                for (int col = 0; col < 5; ++col)
                {
                    if ((bits & static_cast<std::uint8_t>(1u << (4 - col))) != 0)
                    {
                        fillRect(dst, cursor + col * scale, y + row * scale, scale, scale,
                                 r, g, b);
                    }
                }
            }
            cursor += 6 * scale;
        }
    }

    // Bandeau sombre + label (echelle 2 puis 1 selon la place disponible).
    void drawLabelBand(m3gss::Image& dst, int x, int y, int width, const std::string& label)
    {
        fillRect(dst, x, y, width, m3gss::tools::kLabelBandHeight, 24, 24, 24);
        const int available = width - 8;
        int scale = 0;
        if (textWidth(label, 2) <= available)
        {
            scale = 2;
        }
        else if (textWidth(label, 1) <= available)
        {
            scale = 1;
        }
        if (scale > 0)
        {
            const int textY = y + (m3gss::tools::kLabelBandHeight - 7 * scale) / 2;
            drawText(dst, x + 4, textY, label, scale, 255, 255, 255);
        }
    }

    void blit(m3gss::Image& dst, const m3gss::Image& src, int dstX, int dstY)
    {
        if (dst.channels != src.channels)
        {
            return;
        }
        for (int y = 0; y < src.height; ++y)
        {
            const int targetY = dstY + y;
            if (targetY < 0 || targetY >= dst.height)
            {
                continue;
            }
            int startX = 0;
            if (dstX < 0)
            {
                startX = -dstX;
            }
            int copyWidth = src.width - startX;
            if (dstX + src.width > dst.width)
            {
                copyWidth = dst.width - dstX - startX;
            }
            if (copyWidth <= 0)
            {
                continue;
            }
            std::memcpy(dst.row(targetY) + static_cast<std::size_t>(dstX + startX) * dst.channels,
                        src.row(y) + static_cast<std::size_t>(startX) * src.channels,
                        static_cast<std::size_t>(copyWidth) * src.channels);
        }
    }

    // Normalisation en RGB pour que les trois panneaux soient comparables.
    m3gss::Image toRgb(const m3gss::Image& src)
    {
        if (src.channels == 3)
        {
            return src;
        }
        m3gss::Image dst(src.width, src.height, 3);
        for (int y = 0; y < src.height; ++y)
        {
            for (int x = 0; x < src.width; ++x)
            {
                const std::uint8_t* s = src.pixel(x, y);
                std::uint8_t* d = dst.pixel(x, y);
                if (src.channels == 1 || src.channels == 2)
                {
                    d[0] = s[0];
                    d[1] = s[0];
                    d[2] = s[0];
                }
                else if (src.channels >= 3)
                {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                }
            }
        }
        return dst;
    }

    // Agrandissement en plus-proche-voisin : UNIQUEMENT pour l'affichage de
    // comparaison du panneau LR. Ce n'est pas un algorithme de reconstruction.
    m3gss::Image scaleNearestForDisplay(const m3gss::Image& src, int dstWidth, int dstHeight)
    {
        m3gss::Image dst(dstWidth, dstHeight, src.channels);
        for (int y = 0; y < dstHeight; ++y)
        {
            const int sy = std::min(
                src.height - 1,
                static_cast<int>((static_cast<long long>(y) * src.height) / dstHeight));
            for (int x = 0; x < dstWidth; ++x)
            {
                const int sx = std::min(
                    src.width - 1,
                    static_cast<int>((static_cast<long long>(x) * src.width) / dstWidth));
                const std::uint8_t* s = src.pixel(sx, sy);
                std::uint8_t* d = dst.pixel(x, y);
                for (int c = 0; c < src.channels; ++c)
                {
                    d[c] = s[c];
                }
            }
        }
        return dst;
    }

    m3gss::Image cropRect(const m3gss::Image& src, int x, int y, int width, int height)
    {
        const int cx = std::min(std::max(0, x), std::max(0, src.width - 1));
        const int cy = std::min(std::max(0, y), std::max(0, src.height - 1));
        const int cw = std::max(1, std::min(width, src.width - cx));
        const int ch = std::max(1, std::min(height, src.height - cy));
        m3gss::Image dst(cw, ch, src.channels);
        for (int row = 0; row < ch; ++row)
        {
            std::memcpy(dst.row(row), src.row(cy + row) + static_cast<std::size_t>(cx) * src.channels,
                        static_cast<std::size_t>(cw) * src.channels);
        }
        return dst;
    }
}
namespace
{
    struct CropBox
    {
        int x = 0;
        int y = 0;
        int size = 0;
    };

    // Cote d'un crop : minDim/4 avec plancher raisonnable, force pair pour que
    // la zone LR correspondre exactement a la grille de reduction /2.
    int cropSide(int width, int height)
    {
        const int minDim = std::min(width, height);
        if (minDim < 1)
        {
            return 1;
        }
        int side = minDim / 4;
        if (side < 8)
        {
            side = std::min(minDim, 8);
        }
        side -= side % 2;
        if (side < 1)
        {
            side = minDim;
        }
        if (side > minDim)
        {
            side = minDim;
        }
        return side;
    }

    // Zone de crop a position proportionnelle fixe et deterministe.
    CropBox makeCropBox(double centerXRatio, double centerYRatio, int width, int height)
    {
        CropBox box;
        box.size = cropSide(width, height);
        int x = static_cast<int>(centerXRatio * static_cast<double>(width)
                                 - 0.5 * static_cast<double>(box.size));
        int y = static_cast<int>(centerYRatio * static_cast<double>(height)
                                 - 0.5 * static_cast<double>(box.size));
        if (x < 0)
        {
            x = 0;
        }
        if (y < 0)
        {
            y = 0;
        }
        if (x > width - box.size)
        {
            x = width - box.size;
        }
        if (y > height - box.size)
        {
            y = height - box.size;
        }
        box.x = x;
        box.y = y;
        return box;
    }

    // Projection d'une coordonnee entre deux resolutions (arrondi, deterministe).
    int mapCoord(int value, int srcSize, int dstSize)
    {
        if (srcSize <= 0)
        {
            return 0;
        }
        const long long numerator = static_cast<long long>(value) * dstSize + srcSize / 2;
        int mapped = static_cast<int>(numerator / srcSize);
        if (mapped < 0)
        {
            mapped = 0;
        }
        if (mapped > dstSize)
        {
            mapped = dstSize;
        }
        return mapped;
    }

    // Planche principale : 3 panneaux de meme taille + bandeau label par panneau.
    m3gss::Image buildBoard(const m3gss::Image& original, const m3gss::Image& lowDisplay,
                            const m3gss::Image& reconstructed)
    {
        const int width = original.width;
        const int panelHeight = original.height;
        m3gss::Image board(width, 3 * (m3gss::tools::kLabelBandHeight + panelHeight), 3);

        const m3gss::Image* panels[3] = {&original, &lowDisplay, &reconstructed};
        const char* labels[3] = {"ORIGINAL", "LOW RESOLUTION", "BASELINE BICUBIC"};

        int y = 0;
        for (int i = 0; i < 3; ++i)
        {
            drawLabelBand(board, 0, y, width, labels[i]);
            blit(board, *panels[i], 0, y + m3gss::tools::kLabelBandHeight);
            y += m3gss::tools::kLabelBandHeight + panelHeight;
        }
        return board;
    }

    // Planche de crop : Original | Low Resolution (affichage) | Baseline,
    // alignes sur exactement la meme zone de l'image.
    m3gss::Image buildCropBoard(const m3gss::Image& original, const m3gss::Image& lowResolution,
                                const m3gss::Image& reconstructed, const CropBox& box)
    {
        const int side = box.size;

        const m3gss::Image cropOriginal = cropRect(original, box.x, box.y, side, side);
        const m3gss::Image cropReconstructed =
            cropRect(reconstructed, box.x, box.y, side, side);

        // Meme zone projetee en coordonnees LR.
        const int lrX0 = mapCoord(box.x, original.width, lowResolution.width);
        const int lrY0 = mapCoord(box.y, original.height, lowResolution.height);
        const int lrX1 = mapCoord(box.x + side, original.width, lowResolution.width);
        const int lrY1 = mapCoord(box.y + side, original.height, lowResolution.height);
        const m3gss::Image cropLrHalf = cropRect(lowResolution, lrX0, lrY0,
                                                 std::max(1, lrX1 - lrX0),
                                                 std::max(1, lrY1 - lrY0));
        // Agrandissement pour AFFICHAGE uniquement (plus-proche-voisin).
        const m3gss::Image cropLr = scaleNearestForDisplay(cropLrHalf, side, side);

        m3gss::Image board(3 * side, m3gss::tools::kLabelBandHeight + side, 3);
        drawLabelBand(board, 0, 0, side, "ORIGINAL");
        drawLabelBand(board, side, 0, side, "LOW RESOLUTION");
        drawLabelBand(board, 2 * side, 0, side, "BASELINE BICUBIC");
        blit(board, cropOriginal, 0, m3gss::tools::kLabelBandHeight);
        blit(board, cropLr, side, m3gss::tools::kLabelBandHeight);
        blit(board, cropReconstructed, 2 * side, m3gss::tools::kLabelBandHeight);
        return board;
    }
}

namespace m3gss::tools
{
    bool generateComparison(const ComparisonOptions& options, ComparisonStats* stats,
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
        *stats = ComparisonStats{};

        if (options.factor < 2)
        {
            if (error != nullptr)
            {
                *error = "facteur entier >= 2 attendu";
            }
            return false;
        }

        // 1. Fichiers presents ?
        if (!std::filesystem::exists(options.originalPath))
        {
            if (error != nullptr)
            {
                *error = "fichier absent : " + options.originalPath.string();
            }
            return false;
        }
        if (!std::filesystem::exists(options.reconstructedPath))
        {
            if (error != nullptr)
            {
                *error = "fichier absent : " + options.reconstructedPath.string();
            }
            return false;
        }

        // 2. Chargement via ImageIO / stb_image.
        std::string loadError;
        const m3gss::Image original =
            m3gss::io::loadImage(options.originalPath.string(), &loadError);
        if (original.empty())
        {
            if (error != nullptr)
            {
                *error = "image invalide (original) : " + loadError;
            }
            return false;
        }
        const m3gss::Image reconstructed =
            m3gss::io::loadImage(options.reconstructedPath.string(), &loadError);
        if (reconstructed.empty())
        {
            if (error != nullptr)
            {
                *error = "image invalide (reconstruction) : " + loadError;
            }
            return false;
        }

        // 3. Dimensions compatibles ?
        if (original.width != reconstructed.width || original.height != reconstructed.height)
        {
            if (error != nullptr)
            {
                *error = "dimensions incompatibles (original "
                    + std::to_string(original.width) + "x" + std::to_string(original.height)
                    + " vs reconstruction " + std::to_string(reconstructed.width) + "x"
                    + std::to_string(reconstructed.height) + ")";
            }
            return false;
        }

        // 4. LR a 50 % : MECANISME DE REDUCTION PARTAGE AVEC LA BASELINE.
        std::string reduceError;
        const m3gss::Image lowResolution =
            m3gss::resampler::reduceByIntegerFactor(original, options.factor, &reduceError);
        if (lowResolution.empty())
        {
            if (error != nullptr)
            {
                *error = "generation LR impossible : " + reduceError;
            }
            return false;
        }

        // 5. Normalisation RGB des trois panneaux.
        const m3gss::Image originalRgb = toRgb(original);
        const m3gss::Image lowRgb = toRgb(lowResolution);
        const m3gss::Image reconstructedRgb = toRgb(reconstructed);

        // 6. Dossier de sortie.
        std::error_code ec;
        std::filesystem::create_directories(options.outputDir, ec);
        if (ec || !std::filesystem::is_directory(options.outputDir))
        {
            if (error != nullptr)
            {
                *error = "impossible de creer le dossier de sortie : "
                    + options.outputDir.string() + (ec ? " (" + ec.message() + ")" : "");
            }
            return false;
        }

        const auto start = std::chrono::steady_clock::now();

        // 7. Planche principale (le LR est agrandi pour AFFICHAGE uniquement).
        const m3gss::Image lowDisplay =
            scaleNearestForDisplay(lowRgb, originalRgb.width, originalRgb.height);
        const m3gss::Image board = buildBoard(originalRgb, lowDisplay, reconstructedRgb);

        // 8. Trois crops a positions proportionnelles fixes et deterministes :
        //    visage (0.50; 0.40), texture (0.75; 0.70), contours (0.25; 0.75).
        const CropBox faceBox = makeCropBox(0.50, 0.40, original.width, original.height);
        const CropBox textureBox = makeCropBox(0.75, 0.70, original.width, original.height);
        const CropBox edgesBox = makeCropBox(0.25, 0.75, original.width, original.height);

        const m3gss::Image cropFace =
            buildCropBoard(originalRgb, lowRgb, reconstructedRgb, faceBox);
        const m3gss::Image cropTexture =
            buildCropBoard(originalRgb, lowRgb, reconstructedRgb, textureBox);
        const m3gss::Image cropEdges =
            buildCropBoard(originalRgb, lowRgb, reconstructedRgb, edgesBox);

        // 9. Ecritures PNG.
        struct OutputFile
        {
            const m3gss::Image* image;
            const char* name;
        };
        const OutputFile outputs[4] = {
            {&board, "comparison.png"},
            {&cropFace, "comparison_face.png"},
            {&cropTexture, "comparison_texture.png"},
            {&cropEdges, "comparison_edges.png"},
        };

        for (const OutputFile& output : outputs)
        {
            const std::filesystem::path path = options.outputDir / output.name;
            std::string writeError;
            if (!m3gss::io::savePng(path.string(), *output.image, &writeError))
            {
                if (error != nullptr)
                {
                    *error = "erreur d'ecriture PNG : " + path.string() + " (" + writeError
                        + ")";
                }
                return false;
            }
            if (log != nullptr)
            {
                *log << "Ecrit : " << path.string() << std::endl;
            }
        }

        // 10. Statistiques.
        stats->originalWidth = original.width;
        stats->originalHeight = original.height;
        stats->originalChannels = original.channels;
        stats->lrWidth = lowResolution.width;
        stats->lrHeight = lowResolution.height;
        stats->reconstructedWidth = reconstructed.width;
        stats->reconstructedHeight = reconstructed.height;
        stats->boardWidth = board.width;
        stats->boardHeight = board.height;
        stats->cropSize = faceBox.size;
        stats->cropBoardWidth = cropFace.width;
        stats->cropBoardHeight = cropFace.height;

        const auto end = std::chrono::steady_clock::now();
        stats->totalMs = std::chrono::duration<double, std::milli>(end - start).count();

        return true;
    }
}