cmake_minimum_required(VERSION 3.20)

project(M3GSS
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(M3GSS
    src/main.cpp
    src/io/ImageIO.cpp
    src/metrics/QualityMetrics.cpp
    src/upscale/BaselineBicubicUpscaler.cpp
    src/upscale/Resampler.cpp
    third_party/stb/stb_impl.cpp
)

target_include_directories(M3GSS PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# Tests de validation deterministes du module de metriques (aucune dependance externe)
enable_testing()

add_executable(M3GSS_metrics_tests
    tests/metrics_tests.cpp
    src/metrics/QualityMetrics.cpp
)

target_include_directories(M3GSS_metrics_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_test(NAME metrics_tests COMMAND M3GSS_metrics_tests)

# Outil independant du moteur : generateur de dataset HR/LR pour l'entrainement futur
add_executable(M3GSS_dataset_generator
    tools/dataset_generator/main.cpp
    tools/dataset_generator/dataset_generator.cpp
    src/io/ImageIO.cpp
    src/upscale/Resampler.cpp
    third_party/stb/stb_impl.cpp
)

target_include_directories(M3GSS_dataset_generator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/dataset_generator
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# Tests deterministes du generateur de dataset
add_executable(M3GSS_dataset_tests
    tests/dataset_generator_tests.cpp
    tools/dataset_generator/dataset_generator.cpp
    src/io/ImageIO.cpp
    src/upscale/Resampler.cpp
    third_party/stb/stb_impl.cpp
)

target_include_directories(M3GSS_dataset_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/dataset_generator
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

add_test(NAME dataset_tests COMMAND M3GSS_dataset_tests)

# Outil temporaire de comparaison visuelle pour la baseline (test reel RE9)
add_executable(M3GSS_compare
    tools/comparison/main.cpp
    tools/comparison/comparison.cpp
    src/io/ImageIO.cpp
    src/upscale/Resampler.cpp
    third_party/stb/stb_impl.cpp
)

target_include_directories(M3GSS_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# Tests deterministes de l'outil de comparaison
add_executable(M3GSS_compare_tests
    tests/comparison_tests.cpp
    tools/comparison/comparison.cpp
    src/io/ImageIO.cpp
    src/upscale/Resampler.cpp
    third_party/stb/stb_impl.cpp
)

target_include_directories(M3GSS_compare_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

add_test(NAME comparison_tests COMMAND M3GSS_compare_tests)

# Code partage, outils et tests : niveau 4 strict (/W4), warnings traites comme erreurs (/WX)
set_source_files_properties(
    src/metrics/QualityMetrics.cpp
    src/upscale/Resampler.cpp
    tools/dataset_generator/dataset_generator.cpp
    tools/dataset_generator/main.cpp
    tools/comparison/comparison.cpp
    tools/comparison/main.cpp
    tests/metrics_tests.cpp
    tests/dataset_generator_tests.cpp
    tests/comparison_tests.cpp
    PROPERTIES COMPILE_OPTIONS "/W4;/WX"
)
