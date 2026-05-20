// bc25d_helpers.cuh — Shared helpers for the four "true small-tile BC 2.5D"
// variant runners (scqr3, householder, givens, qdwh).
//
// Mathematical contract (Kwasniewski SC'21 Fig.6 + tex §A.3a):
//   Tile (I, J) with I, J in [0, N/b) is owned by rank
//     pi(I, J) = (I mod Px, J mod Py)   within each Pz layer.
//   Each Pz layer holds a complete replica of all tiles, so the storage
//     per rank is locr × locc doubles (independent of Pz).
//   numroc(N, b, my_px, 0, Px), numroc(N, b, my_py, 0, Py) — ScaLAPACK style.
//   Local indexing: tile (I, J) at local (LBR*b + r, LBC*b + s)
//     where LBR = I / Px, LBC = J / Py, r = i % b, s = j % b.
//
// Pz layers split the panel-row reduction: same data, partial sums over the
// (locr / Pz)-row slice, ncclAllReduce within col_comm to aggregate.

#ifndef BC25D_HELPERS_CUH
#define BC25D_HELPERS_CUH

#include <cstdint>
#include "block_cyclic.hpp"

// First local column whose global column index >= trail_start, owned by my_py.
// Returns locc if no owned cols satisfy the constraint.
static inline std::int64_t bc_first_trail_lcol(int trail_start, int b, int Py, int my_py,
                                                 std::int64_t locc) {
    if (trail_start <= 0) {
        // Owned cols start from LBC=0
        return 0;
    }
    int J_min = (trail_start + b - 1) / b;   // smallest tile-col J with J*b >= trail_start
    int residual = J_min % Py;
    int offset = ((my_py - residual) + Py) % Py;
    int J_first = J_min + offset;
    int LBC_first = (J_first - my_py) / Py;
    std::int64_t lc = (std::int64_t)LBC_first * b;
    if (lc > locc) lc = locc;
    return lc;
}

// Local column offset of global column k (assuming caller already checked
// my_py == (k/b) mod Py).  Caller's panel slice starts at this offset.
static inline std::int64_t bc_panel_lcol(int k, int b, int Py) {
    int J = k / b;
    int LBC = J / Py;
    int col_in_tile = k % b;
    return (std::int64_t)LBC * b + col_in_tile;
}

// Is this rank an owner of the panel column k?
static inline bool bc_my_py_owns_panel(int k, int b, int Py, int my_py) {
    return ((k / b) % Py) == my_py;
}

// Compute (locr, locc) for the current rank.
static inline void bc_local_dims(int N, int b, int Px, int Py, int my_px, int my_py,
                                   std::int64_t* locr_out, std::int64_t* locc_out) {
    *locr_out = numroc(N, b, my_px, 0, Px);
    *locc_out = numroc(N, b, my_py, 0, Py);
}

// Snap a candidate block size down to the largest divisor of N that's
// <= b_in, with a floor of 64.  Used by the BC 2.5D dispatch to ensure
// N % b == 0 while staying within the manuscript's admissible §A3b window
// (c ≤ b ≤ N/√P1) — we only ever shrink b, never grow it.  When the
// derived b already divides N this is a no-op.
static inline int snap_b_to_divisor(int N, int b_in) {
    if (b_in <= 0 || N <= 0) return b_in;
    if (N % b_in == 0) return b_in;
    int hi = (b_in > N) ? N : b_in;
    for (int cand = hi; cand >= 64; --cand) {
        if (N % cand == 0) return cand;
    }
    return b_in;
}

// Validation: for each rank (col_rank==0 in its col_group), scan diag(Q'Q)
// over its owned cols and report max|diag - 1|.  Caller AllReduce-MAXes
// across MPI_COMM_WORLD.
//
// d_GG: locc × locc column-major upper-triangular (output of Dsyrk after
//        col_comm AllReduce-sum).
// my_py owned global col j corresponds to local col lj = LBC*b + (j mod b)
//   for LBC = lj/b, J = LBC*Py + my_py, j = J*b + (lj % b).

#endif  // BC25D_HELPERS_CUH
