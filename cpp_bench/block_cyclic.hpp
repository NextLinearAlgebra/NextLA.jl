// ScaLAPACK-style 2D block-cyclic ownership on a P_r × P_c process grid with
// block size MB × NB (TeX / Conflux π(I,J) specialization: one tile per block).
// Used when Path (s) runs with --layout=blockcyclic (Pz=1 only in this bench).

#ifndef BLOCK_CYCLIC_HPP
#define BLOCK_CYCLIC_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

// Number of rows or columns owned by process iproc in a 1D block-cyclic dim.
inline std::int64_t numroc(std::int64_t n, std::int64_t nb, int iproc, int isrc, int nprocs) {
    std::int64_t mydist = (nprocs + iproc - isrc) % nprocs;
    std::int64_t nblocks = n / nb;
    std::int64_t nr = (nblocks / nprocs) * nb;
    std::int64_t extrablocks = nblocks % nprocs;
    if (mydist < extrablocks) nr += nb;
    else if (mydist == extrablocks) nr += n % nb;
    return nr;
}

// Global (gi,gj) 0-based in [0,N) × [0,N): owning process in P_r × P_c grid.
inline void owner_of_block_cyclic(std::int64_t gi, std::int64_t gj, std::int64_t N,
                                  int MB, int NB, int Pr, int Pc, int& prow, int& pcol) {
    (void)N;
    prow = (int)((gi / MB) % Pr);
    pcol = (int)((gj / NB) % Pc);
}

inline bool block_cyclic_valid(int N, int MB, int NB, int Pr, int Pc) {
    if (N <= 0 || MB <= 0 || NB <= 0 || Pr <= 0 || Pc <= 0) return false;
    if (N % MB != 0 || N % NB != 0) return false;
    return true;
}

inline std::int64_t total_elements_bc(int N, int MB, int NB, int Pr, int Pc) {
    std::int64_t sum = 0;
    for (int ir = 0; ir < Pr; ++ir)
        for (int ic = 0; ic < Pc; ++ic)
            sum += numroc(N, MB, ir, 0, Pr) * numroc(N, NB, ic, 0, Pc);
    return sum;
}

// 2D block-cyclic (square blocks MB=NB=b): global (gi,gj) -> local (li,lj) if owned by (my_px,my_py).
inline bool global_to_local_bc(int gi, int gj, int b, int Px, int Py, int my_px, int my_py,
                               std::int64_t& li, std::int64_t& lj) {
    int IBR = gi / b;
    int IBC = gj / b;
    if ((IBR % Px) != my_px || (IBC % Py) != my_py) return false;
    int LBR = IBR / Px;
    int LBC = IBC / Py;
    li = (std::int64_t)LBR * b + (gi % b);
    lj = (std::int64_t)LBC * b + (gj % b);
    return true;
}

inline void local_to_global_bc(std::int64_t li, std::int64_t lj, int b, int Px, int Py,
                               int my_px, int my_py, int& gi, int& gj) {
    int LBR = (int)(li / b);
    int LBC = (int)(lj / b);
    int ri = (int)(li % b);
    int rj = (int)(lj % b);
    int IBR = LBR * Px + my_px;
    int IBC = LBC * Py + my_py;
    gi = IBR * b + ri;
    gj = IBC * b + rj;
}

#endif
