// Shared tri-mode surface for Path (s) and Paths (h)(g)(q).

#ifndef MATRIX_MODE_HPP
#define MATRIX_MODE_HPP

#include <cstring>

enum class MatrixMode { FP64, FP64_MP, FP64_MP_TF32, FP32_FULL };

inline MatrixMode parse_matrix_mode(const char* s) {
    if (!s) return MatrixMode::FP64;
    if (std::strcmp(s, "fp64") == 0) return MatrixMode::FP64;
    if (std::strcmp(s, "fp64mp") == 0) return MatrixMode::FP64_MP;
    if (std::strcmp(s, "fp64mp_tf32") == 0) return MatrixMode::FP64_MP_TF32;
    if (std::strcmp(s, "fp32full") == 0) return MatrixMode::FP32_FULL;
    return MatrixMode::FP64;
}

inline const char* matrix_mode_tag(MatrixMode m) {
    switch (m) {
        case MatrixMode::FP64_MP: return "fp64mp";
        case MatrixMode::FP64_MP_TF32: return "fp64mp_tf32";
        case MatrixMode::FP32_FULL: return "fp32full";
        default: return "fp64";
    }
}

#endif
