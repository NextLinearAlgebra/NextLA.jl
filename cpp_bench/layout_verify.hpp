// layout_verify.hpp — Host-side checks for the row×column slab layout used in
// scqr3_full25d_bench.cu (Conflux-compatible assignment for square P₁ with
// px=py=√P₁, pz=c).  True ScaLAPACK-style block-cyclic π(I,J) from SC'21 is
// documented in qr_schur_xpartition.tex §A3aprime; this bench uses the slab
// specialization consistent with the file header mapping.
//
// Use these for small-N CI / smoke: total element count and divisibility.

#ifndef LAYOUT_VERIFY_HPP
#define LAYOUT_VERIFY_HPP

#include <cstdint>

inline bool layout_slab_divisible(int N, int Px, int Py, int Pz) {
    if (Px <= 0 || Py <= 0 || Pz <= 0 || N <= 0) return false;
    return (N % (Px * Pz) == 0) && (N % Py == 0);
}

inline std::int64_t layout_slab_elements_per_rank(int N, int Px, int Py, int Pz) {
    if (!layout_slab_divisible(N, Px, Py, Pz)) return -1;
    std::int64_t m_loc = N / (std::int64_t)(Px * Pz);
    std::int64_t n_loc = N / (std::int64_t)Py;
    return m_loc * n_loc;
}

inline bool layout_slab_total_matches_global(int N, int P, int Px, int Py, int Pz) {
    if (Px * Py * Pz != P) return false;
    std::int64_t el = layout_slab_elements_per_rank(N, Px, Py, Pz);
    if (el < 0) return false;
    return el * (std::int64_t)P == (std::int64_t)N * (std::int64_t)N;
}

#endif  // LAYOUT_VERIFY_HPP
