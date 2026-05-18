// full25d_grid.hpp — Shared 2.5D Conflux grid + col/row sub-communicator
// setup for the four single-path benches (Householder, Givens, QDWH; the
// Path-s variant lives in scqr3_full25d_bench.cu and uses this same
// scheduling math via derived_schedule.hpp).
//
// Mathematical contract (tex §A.3 + Kwasniewski SC'21):
//   * P = P_x * P_y * P_z processors.
//   * P_1 = P_x * P_y = P / c; c = P_z ∈ [1, floor(min(P*M/N², P^{1/3}))].
//   * Per-rank slab: rank (p_x, p_y, p_z) owns rows
//       [(p_z * P_x + p_x) * m_loc, (p_z * P_x + p_x + 1) * m_loc)
//     and columns [p_y * n_loc, (p_y + 1) * n_loc) of A,
//     with m_loc = N / (P_x * P_z), n_loc = N / P_y.
//   * Column group  (same p_y, |group|=P_1 c = P_x P_z): Phase Q3 / Q4 AllReduces.
//   * Row group     (same (p_x, p_z), |group|=P_y):       Phase Q1 Q-broadcast.
//
// Public API:
//   struct Full25DGrid       — resolved (Px, Py, Pz, c, m_loc, n_loc, ...).
//   resolve_full25d_grid()   — pick (Px, Py, Pz) from CLI hints, else derive
//                              from compute_derived_schedule(P, N, M).
//   build_full25d_subcomms() — MPI_Comm_split + ncclCommInitRank for col_comm
//                              and row_comm.  Returns a small POD with both
//                              MPI handles and NCCL handles + ranks/sizes.
//
// All work is host-side (no CUDA kernels here); benches still own their
// per-precision allocations + kernels.

#ifndef FULL25D_GRID_HPP
#define FULL25D_GRID_HPP

#include <mpi.h>
#include <nccl.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>

#include "derived_schedule.hpp"

#define FULL25D_NCCL_CHECK(stmt) do { \
    ncclResult_t r__ = (stmt); \
    if (r__ != ncclSuccess) { \
        int wr__ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &wr__); \
        fprintf(stderr, "[r%d] NCCL %s @ %s:%d\n", wr__, ncclGetErrorString(r__), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 4); \
    } \
} while (0)

struct Full25DGrid {
    int P = 1;
    int Px = 1, Py = 1, Pz = 1;
    int c = 1;
    int N = 0;
    int m_loc = 0, n_loc = 0;
    // This rank's coordinates: rank = pz*Px*Py + px*Py + py.
    int my_px = 0, my_py = 0, my_pz = 0;
    // Auto-derived flag (true ⇒ from compute_derived_schedule; false ⇒ from CLI).
    bool from_derived = false;
};

struct Full25DSubcomms {
    MPI_Comm mpi_col = MPI_COMM_NULL;
    MPI_Comm mpi_row = MPI_COMM_NULL;
    int col_size = 0, col_rank = 0;
    int row_size = 0, row_rank = 0;
    ncclComm_t nccl_col = nullptr;
    ncclComm_t nccl_row = nullptr;
};

// Pick (Px, Py, Pz) for this run.  Order of precedence:
//   1. Explicit --px/--py/--pz from CLI (px*py*pz must equal P).
//   2. compute_derived_schedule(P, N, M_fp64_words) when M > 0.
//   3. Legacy 1×P×1 row partition.
inline Full25DGrid resolve_full25d_grid(int P, int rank_world, int N,
                                         int px_cli, int py_cli, int pz_cli,
                                         std::int64_t M_fp64_words) {
    Full25DGrid G;
    G.P = P;
    G.N = N;
    if (px_cli > 0 && py_cli > 0 && pz_cli > 0) {
        if (px_cli * py_cli * pz_cli != P) {
            if (rank_world == 0) fprintf(stderr, "px*py*pz=%d != P=%d\n", px_cli*py_cli*pz_cli, P);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        G.Px = px_cli; G.Py = py_cli; G.Pz = pz_cli;
    } else if (M_fp64_words > 0) {
        DerivedSchedule D = compute_derived_schedule(P, N, M_fp64_words, false);
        if (D.ok) {
            G.Px = D.px; G.Py = D.py; G.Pz = D.pz;
            G.from_derived = true;
        } else {
            // Fallback to legacy 1×P×1 if the derivation fails.
            G.Px = 1; G.Py = P; G.Pz = 1;
        }
    } else {
        // No M info: default to 1×P×1 row partition.
        G.Px = 1; G.Py = P; G.Pz = 1;
    }
    G.c = G.Pz;
    if (N % (G.Px * G.Pz) != 0 || N % G.Py != 0) {
        if (rank_world == 0)
            fprintf(stderr, "N=%d not divisible by Px*Pz=%d or Py=%d\n",
                    N, G.Px * G.Pz, G.Py);
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    G.m_loc = N / (G.Px * G.Pz);
    G.n_loc = N / G.Py;
    G.my_pz = rank_world / (G.Px * G.Py);
    G.my_px = (rank_world / G.Py) % G.Px;
    G.my_py = rank_world % G.Py;
    return G;
}

// Build MPI column / row sub-communicators and the corresponding NCCL comms.
//   col_comm: ranks sharing my_py (size = Px*Pz);
//             rank within col_comm = my_pz*Px + my_px (deterministic).
//   row_comm: ranks sharing (my_px, my_pz) (size = Py);
//             rank within row_comm = my_py.
inline Full25DSubcomms build_full25d_subcomms(const Full25DGrid& G) {
    Full25DSubcomms S;
    int col_color = G.my_py;
    int col_key   = G.my_pz * G.Px + G.my_px;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, col_key, &S.mpi_col);
    int row_color = G.my_pz * G.Px + G.my_px;
    int row_key   = G.my_py;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, row_key, &S.mpi_row);
    MPI_Comm_size(S.mpi_col, &S.col_size);
    MPI_Comm_rank(S.mpi_col, &S.col_rank);
    MPI_Comm_size(S.mpi_row, &S.row_size);
    MPI_Comm_rank(S.mpi_row, &S.row_rank);

    ncclUniqueId id_col, id_row;
    if (S.col_rank == 0) FULL25D_NCCL_CHECK(ncclGetUniqueId(&id_col));
    MPI_Bcast(&id_col, sizeof(id_col), MPI_BYTE, 0, S.mpi_col);
    FULL25D_NCCL_CHECK(ncclCommInitRank(&S.nccl_col, S.col_size, id_col, S.col_rank));
    if (S.row_rank == 0) FULL25D_NCCL_CHECK(ncclGetUniqueId(&id_row));
    MPI_Bcast(&id_row, sizeof(id_row), MPI_BYTE, 0, S.mpi_row);
    FULL25D_NCCL_CHECK(ncclCommInitRank(&S.nccl_row, S.row_size, id_row, S.row_rank));
    return S;
}

inline void destroy_full25d_subcomms(Full25DSubcomms& S) {
    if (S.nccl_col) { ncclCommDestroy(S.nccl_col); S.nccl_col = nullptr; }
    if (S.nccl_row) { ncclCommDestroy(S.nccl_row); S.nccl_row = nullptr; }
    if (S.mpi_col != MPI_COMM_NULL) { MPI_Comm_free(&S.mpi_col); }
    if (S.mpi_row != MPI_COMM_NULL) { MPI_Comm_free(&S.mpi_row); }
}

// Print the resolved grid (called by rank 0).
inline void print_full25d_grid(const Full25DGrid& G, const char* bench_tag,
                                std::int64_t M_fp64_words, int b) {
    printf("=================================================================\n");
    printf(" Full 2.5D  bench=%s  N=%d b=%d\n", bench_tag, G.N, b);
    printf("   derived (c, P1=Px*Py, Px, Py, Pz) = (%d, %d, %d, %d, %d)  %s\n",
           G.c, G.Px * G.Py, G.Px, G.Py, G.Pz, G.from_derived ? "[auto]" : "[CLI]");
    printf("   m_loc=%d  n_loc=%d  M_fp64_words=%lld\n",
           G.m_loc, G.n_loc, (long long)M_fp64_words);
    printf("=================================================================\n");
    fflush(stdout);
}

// Clip a panel-column block [k, k+sb) so it stays entirely within one
// p_y-block of A's columns.  Required when n_loc < b and the panel
// crosses a p_y boundary.
inline int sb_clipped_full25d(int k, int b, int N, int n_loc) {
    int py_panel = k / n_loc;
    int next_py_boundary = (py_panel + 1) * n_loc;
    int avail = std::min(N - k, next_py_boundary - k);
    return std::min(b, avail);
}

#endif  // FULL25D_GRID_HPP
