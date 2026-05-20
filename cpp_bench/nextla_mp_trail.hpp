// Shared predicates for mixed-precision trailing (FP64 panel + reduced-precision trail).
// Used by Path (s) slab/BC and Paths (h)(g) BC so FP64_MP and FP64_MP_TF32 stay consistent.
#ifndef NEXTLA_MP_TRAIL_HPP
#define NEXTLA_MP_TRAIL_HPP

#include "matrix_mode.hpp"

inline bool nextla_is_mp_trail_matrix(MatrixMode m) {
    return m == MatrixMode::FP64_MP || m == MatrixMode::FP64_MP_TF32;
}

inline bool nextla_requests_tf32_matrix(MatrixMode m) {
    return m == MatrixMode::FP64_MP_TF32;
}

#endif  // NEXTLA_MP_TRAIL_HPP
