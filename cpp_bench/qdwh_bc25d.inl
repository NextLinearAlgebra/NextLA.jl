// qdwh_bc25d.inl — Path-q true small-tile BC 2.5D runner.
//
// Layout: X is stored in BC layout (locr × locc per rank, with locr =
// numroc(N, b, my_px, 0, Px), locc = numroc(N, b, my_py, 0, Py)).  The
// 2N × N stacked S used by the inner CQR2 has shape (2*locr) × locc
// per rank — top half is sqrt(c_k) · X_local, bottom half is the
// identity rows whose global row index = (my_pz*Px + my_px)*locr + i
// for i ∈ [0, locr), filtered by my_py owning tile-col J for that
// global column.
//
// Inner CQR2 on the stacked S:
//   Phase Q1: SYRK on local stacked slab (m_st_loc=2*locr × sb) →
//             b × b partial G; AllReduce(G) over col_comm; POTRF +
//             TRSM on local slab (each rank applies replicated R).
//             Two passes (CQR2).
//   Phase Q2 trailing: same W = Q^T A_trail / A_trail -= Q W pattern.
//
// Polar P = Q1 Q2^T:
//   AllGather Q2 (bottom locr rows of d_S) across col_comm → N × locc;
//   AllGather Q1 (top locr rows)         across row_comm → locr × N;
//   Block-by-block Dgemm accumulating into d_P (locr × locc).
// Requires locr == locc and col_size == row_size (same constraint as
// slab full25d QDWH; satisfied when Px==Py and Pz==1).  Aborts cleanly
// at Pz>1 — same as the slab path.
//
// X update: X_{k+1} = (b_k/c_k) X + (a_k - b_k/c_k)/sqrt(c_k) · P.

#ifndef QDWH_BC25D_INL
#define QDWH_BC25D_INL

#include "bc25d_helpers.cuh"
#include "full25d_grid.hpp"
#include "full25d_kernels.cuh"
#include "bench_vendor_metrics.hpp"
#include "nextla_mp_trail.hpp"

// Rearrange Q2_recv (col_size blocks of locr × locc rank-block layout) into
// a single contiguous (col_size·locr) × locc = N × locc column-major buffer.
//   src[r * locr*locc + j_loc * locr + k_in_r]
//   → dst[r*locr + k_in_r + j_loc * N_rows]
// After this, the buffer is directly usable as cublasDgemm's B with
// transb=N, ldb = N_rows = col_size * locr.  Works for any col_size and
// locc; the resulting Dgemm Q1_recv (locr × N) · Q2_full (N × locc) gives
// the per-rank polar P_loc = locr × locc.
__global__ static void qdwh_bc25d_q2_recv_to_full(const double* __restrict__ recv,
                                                    double* __restrict__ full,
                                                    int locr, int locc, int col_size) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)col_size * locr * locc;
    if (idx >= total) return;
    long long N_rows = (long long)col_size * locr;
    int k_in_r = (int)(idx % locr);
    int j_loc  = (int)((idx / locr) % locc);
    int r      = (int)(idx / ((long long)locr * locc));
    long long src = (long long)r * locr * locc + (long long)j_loc * locr + k_in_r;
    long long dst = (long long)(r * locr + k_in_r) + (long long)j_loc * N_rows;
    full[dst] = recv[src];
}

// Rearrange Q1_recv (row_size blocks of locr × locc rank-block layout) into
// a single contiguous locr × (row_size·locc) = locr × N column-major buffer.
//   src[m * locr*locc + k_in_m * locr + i_loc]
//   → dst[i_loc + (m*locc + k_in_m) * locr]
// Q1_recv in column-major locr × N IS already in this layout if the
// AllGather concatenates rank-blocks in the leading column dimension —
// in our case it does, so this kernel is a no-op IDENTITY for Q1.
// But col_size may force a different layout in some configurations; this
// kernel makes the code robust.
__global__ static void qdwh_bc25d_q1_recv_to_full(const double* __restrict__ recv,
                                                    double* __restrict__ full,
                                                    int locr, int locc, int row_size) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)row_size * locr * locc;
    if (idx >= total) return;
    long long N_cols = (long long)row_size * locc;
    int i_loc   = (int)(idx % locr);
    int k_in_m  = (int)((idx / locr) % locc);
    int m       = (int)(idx / ((long long)locr * locc));
    long long src = (long long)m * locr * locc + (long long)k_in_m * locr + i_loc;
    long long dst = (long long)i_loc + (long long)(m * locc + k_in_m) * locr;
    full[dst] = recv[src];
}

__global__ static void qdwh_bc25d_cast_d2f(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ static void qdwh_bc25d_cast_f2d(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}

// Stacked S fill: top locr rows = scale_X · X_local; bottom locr rows
// = identity slice (row r → global row (my_row_in_col_group)*locr + r;
// column j_loc → global col bc_local_to_global_col(j_loc)).  The latter
// requires knowing which global column j_loc corresponds to.  For
// non-degenerate BC the local-col j_loc maps to global col
//   J = (j_loc / b) * Py + my_py
//   col_in_tile = j_loc % b
//   global_col = J*b + col_in_tile
// Identity bit is 1 iff global_row == global_col.
__global__ static void qdwh_bc25d_fill_stacked(double* __restrict__ S,
                                                 const double* __restrict__ Xk,
                                                 std::int64_t locr, std::int64_t locc,
                                                 int my_row_in_col_group,
                                                 int my_py, int Py, int b,
                                                 double scale_X) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)2 * locr * locc;
    if (idx >= total) return;
    long long m_st = 2 * locr;
    int i = (int)(idx % m_st);
    int j_loc = (int)(idx / m_st);
    if ((long long)i < locr) {
        S[idx] = scale_X * Xk[i + (long long)j_loc * locr];
    } else {
        int row_offset = i - (int)locr;
        long long global_row = (long long)my_row_in_col_group * locr + row_offset;
        int J = j_loc / b;
        int col_in_tile = j_loc % b;
        long long global_col = (long long)(J * Py + my_py) * b + col_in_tile;
        S[idx] = (global_row == global_col) ? 1.0 : 0.0;
    }
}

__global__ static void qdwh_bc25d_update_X(double* __restrict__ Xnew,
                                             const double* __restrict__ Xk,
                                             const double* __restrict__ P,
                                             std::int64_t locr, std::int64_t locc,
                                             double alpha_x, double alpha_p) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)locr * locc;
    if (idx >= total) return;
    Xnew[idx] = alpha_x * Xk[idx] + alpha_p * P[idx];
}

static int run_qdwh_bc25d_fp64(int N, int b, int iters, bool eff_la,
                                 bool use_mp_trail, bool use_tf32_trail,
                                 const Full25DGrid& G, const Full25DSubcomms& S) {
    int Px = G.Px, Py = G.Py, Pz = G.Pz;
    int my_py = G.my_py;
    int col_size = S.col_size;
    int row_size = S.row_size;
    (void)eff_la;
    std::int64_t locr = 0, locc = 0;
    bc_local_dims(N, b, Px, Py, G.my_px, G.my_py, &locr, &locc);
    if (locr <= 0 || locc <= 0) {
        if (_rank == 0) fprintf(stderr, "qdwh_bc25d: empty local buffer (locr=%lld, locc=%lld)\n",
                                 (long long)locr, (long long)locc);
        return 81;
    }
    // The polar reconstruction below uses a rearrange + single Dgemm that
    // works for any (col_size, row_size, locr, locc), so no abort here.
    std::int64_t m_st_loc = 2 * locr;

    cudaStream_t s_comp, s_comm, s_la = nullptr, s_comm_la = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    if (eff_la) {
        CUDA_CHECK(cudaStreamCreate(&s_la));
        CUDA_CHECK(cudaStreamCreate(&s_comm_la));
    }
    cudaEvent_t e_comp_done, e_ar_done, e_q2_next_done, e_la_q1_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done,    cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,      cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_q2_next_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_la_q1_done,   cudaEventDisableTiming));
    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cusolverDnHandle_t cusolver; CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s_comp));
    cublasHandle_t cublas_la = nullptr;
    cusolverDnHandle_t cusolver_la = nullptr;
    if (eff_la) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));   CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
        if (use_tf32_trail) CUBLAS_CHECK(cublasSetMathMode(cublas_la, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    if (use_tf32_trail) {
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    double *d_A = nullptr, *d_X = nullptr, *d_Xnew = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)locr * locc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,    (size_t)locr * locc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Xnew, (size_t)locr * locc * sizeof(double)));
    {
        std::vector<double> host((size_t)locr * locc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    double* d_S = nullptr;
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)m_st_loc * locc * sizeof(double)));

    double *d_G = nullptr, *d_W = nullptr, *d_panel_bcast = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G,           (size_t)b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,           (size_t)b * locc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel_bcast, (size_t)m_st_loc * b * sizeof(double)));
    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    // LA duplicate scratch (for panel-(k+1) Q1 inside inner CQR2 on s_la).
    double *d_G_la = nullptr, *d_panel_bcast_la = nullptr, *d_potrf_work_la = nullptr;
    int* d_info_la = nullptr;
    if (eff_la) {
        CUDA_CHECK(cudaMalloc(&d_G_la,            (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_bcast_la,  (size_t)m_st_loc * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_potrf_work_la,   (size_t)potrf_lwork * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info_la,         sizeof(int)));
    }

    double *d_Q2_recv = nullptr, *d_Q1_recv = nullptr, *d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q2_recv, (size_t)locr * locc * (size_t)col_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q1_recv, (size_t)locr * locc * (size_t)row_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P,       (size_t)locr * locc * sizeof(double)));
    // Rearranged single-buffer views for the polar P = Q1 · Q2^T Dgemm.
    // After rearrangement: d_Q1_full = locr × N column-major; d_Q2_full = N × locc.
    // Works for any (col_size, row_size, locr, locc).
    double *d_Q1_full = nullptr, *d_Q2_full = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q1_full, (size_t)locr * (size_t)row_size * locc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q2_full, (size_t)col_size * (size_t)locr * locc * sizeof(double)));

    float *d_panel_bcast_f = nullptr, *d_S_trail_f = nullptr, *d_W_f = nullptr;
    if (use_mp_trail) {
        CUDA_CHECK(cudaMalloc(&d_panel_bcast_f, (size_t)m_st_loc * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_S_trail_f,     (size_t)m_st_loc * locc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_f,           (size_t)b * locc * sizeof(float)));
    }

    const double one_d = 1.0, zero_d = 0.0, neg_one_d = -1.0;
    const float  one_f = 1.f,  zero_f = 0.f,  neg_one_f = -1.f;

    int my_row_in_col_group = S.col_rank;

    // inner_q1_compute: panel CQR2 (passes=2) on the stacked S, writing the
    // resulting Q-of-the-stacked-panel into `panel_bcast_buf` AND back into
    // S_panel.  Does NOT do the row_comm broadcast.
    auto inner_q1_compute = [&](cudaStream_t s_use, cublasHandle_t cb,
                                  cusolverDnHandle_t cs,
                                  double* G_buf, double* panel_bcast_buf,
                                  double* potrf_work_buf, int* info_buf,
                                  ncclComm_t nccl_col_use, cudaStream_t s_comm_use,
                                  int k, int sb) {
        int py_panel = (k / b) % Py;
        bool i_own_panel = (my_py == py_panel);
        if (!i_own_panel) return;
        std::int64_t panel_lcol = bc_panel_lcol(k, b, Py);
        double* S_panel = d_S + (size_t)panel_lcol * m_st_loc;
        for (int it = 0; it < 2; ++it) {
            CUBLAS_CHECK(cublasDsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                      sb, (int)m_st_loc, &one_d, S_panel, (int)m_st_loc,
                                      &zero_d, G_buf, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm_use, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(G_buf, G_buf, (size_t)b * b, ncclDouble, ncclSum,
                                              nccl_col_use, s_comm_use));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_UPPER, sb, G_buf, b,
                                              potrf_work_buf, potrf_lwork, info_buf));
            CUBLAS_CHECK(cublasDtrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      (int)m_st_loc, sb, &one_d, G_buf, b,
                                      S_panel, (int)m_st_loc));
        }
        CUDA_CHECK(cudaMemcpyAsync(panel_bcast_buf, S_panel,
                                    (size_t)m_st_loc * sb * sizeof(double),
                                    cudaMemcpyDeviceToDevice, s_use));
    };

    auto inner_q1_broadcast = [&](cudaStream_t s_use,
                                    double* panel_bcast_buf,
                                    int k, int sb) {
        int py_panel = (k / b) % Py;
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclBroadcast(panel_bcast_buf, panel_bcast_buf,
                                          (size_t)m_st_loc * sb, ncclDouble,
                                          py_panel, S.nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
    };

    // Compatibility shim: old inner_q1 signature.
    auto inner_q1 = [&](int k, int sb) {
        inner_q1_compute(s_comp, cublas, cusolver, d_G, d_panel_bcast,
                          d_potrf_work, d_info, S.nccl_col, s_comm, k, sb);
        inner_q1_broadcast(s_comp, d_panel_bcast, k, sb);
    };

    auto inner_q2_part = [&](int sb, double* panel_bcast_buf,
                              std::int64_t trail_lcol, std::int64_t ncols_part,
                              std::int64_t d_W_col_off) {
        if (ncols_part <= 0) return;
        std::int64_t ncols = ncols_part;
        double* d_S_trail = d_S + (size_t)trail_lcol * m_st_loc;
        double* d_W_part  = d_W + (size_t)d_W_col_off * b;
        if (!use_mp_trail) {
            CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, (int)ncols, (int)m_st_loc,
                                      &one_d, panel_bcast_buf, (int)m_st_loc,
                                              d_S_trail,       (int)m_st_loc,
                                      &zero_d, d_W_part, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(d_W_part, d_W_part, (size_t)sb * ncols, ncclDouble, ncclSum,
                                              S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
            CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)m_st_loc, (int)ncols, sb,
                                      &neg_one_d, panel_bcast_buf, (int)m_st_loc,
                                                  d_W_part,        b,
                                      &one_d, d_S_trail, (int)m_st_loc));
        } else {
            size_t np = (size_t)m_st_loc * sb, nt = (size_t)m_st_loc * ncols;
            float* d_S_trail_f_off = d_S_trail_f + (size_t)trail_lcol * m_st_loc;
            float* d_W_part_f = d_W_f + (size_t)d_W_col_off * b;
            qdwh_bc25d_cast_d2f<<<(np + 255)/256, 256, 0, s_comp>>>(
                panel_bcast_buf, d_panel_bcast_f, np);
            qdwh_bc25d_cast_d2f<<<(nt + 255)/256, 256, 0, s_comp>>>(
                d_S_trail, d_S_trail_f_off, nt);
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, (int)ncols, (int)m_st_loc,
                                      &one_f, d_panel_bcast_f, (int)m_st_loc,
                                              d_S_trail_f_off, (int)m_st_loc,
                                      &zero_f, d_W_part_f, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(d_W_part_f, d_W_part_f, (size_t)sb * ncols, ncclFloat, ncclSum,
                                              S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)m_st_loc, (int)ncols, sb,
                                      &neg_one_f, d_panel_bcast_f, (int)m_st_loc,
                                                  d_W_part_f, b,
                                      &one_f, d_S_trail_f_off, (int)m_st_loc));
            qdwh_bc25d_cast_f2d<<<(nt + 255)/256, 256, 0, s_comp>>>(
                d_S_trail_f_off, d_S_trail, nt);
        }
    };

    auto inner_q2 = [&](int k, int sb) {
        std::int64_t trail_lcol = bc_first_trail_lcol(k + sb, b, Py, my_py, locc);
        std::int64_t ncols = locc - trail_lcol;
        if (ncols <= 0) return;
        inner_q2_part(sb, d_panel_bcast, trail_lcol, ncols, 0);
    };

    auto inner_qr_serial = [&]() {
        for (int k = 0; k < N; k += b) {
            int sb = std::min(b, N - k);
            inner_q1(k, sb);
            if (k + sb < N) inner_q2(k, sb);
        }
    };

    // LA inner CQR2: pipeline panel-(k+1) Q1 (on s_la, with d_*_la scratch
    // and S.nccl_col_la) against panel-k Q2_rest (on s_comp).  pri↔la swap
    // each iteration so the just-finished panel is in d_panel_bcast (the
    // "primary" buffer that the next iteration's Q2 reads from).
    auto inner_qr_la = [&]() {
        // Pointers we'll swap to track which buffer set is "current".
        double* G_pri = d_G;          double* G_la_p = d_G_la;
        double* pb_pri = d_panel_bcast; double* pb_la_p = d_panel_bcast_la;
        double* pw_pri = d_potrf_work;  double* pw_la_p = d_potrf_work_la;
        int*    info_pri = d_info;      int*    info_la_p = d_info_la;

        int sb_first = std::min(b, N);
        inner_q1_compute(s_comp, cublas, cusolver, G_pri, pb_pri, pw_pri, info_pri,
                          S.nccl_col, s_comm, 0, sb_first);
        inner_q1_broadcast(s_comp, pb_pri, 0, sb_first);

        for (int k = 0; k < N; k += b) {
            int sb = std::min(b, N - k);
            int next_k = k + sb;
            if (next_k >= N) break;
            int sb_next = std::min(b, N - next_k);
            int py_panel_next = (next_k / b) % Py;
            bool i_own_next = (my_py == py_panel_next);

            std::int64_t trail_lcol = bc_first_trail_lcol(next_k, b, Py, my_py, locc);
            std::int64_t ncols = locc - trail_lcol;
            std::int64_t sb_next_local = i_own_next ? sb_next : 0;
            std::int64_t ncols_rest    = ncols - sb_next_local;

            if (sb_next_local > 0) {
                inner_q2_part(sb, pb_pri, trail_lcol, sb_next_local, 0);
                CUDA_CHECK(cudaEventRecord(e_q2_next_done, s_comp));
            }

            CUDA_CHECK(cudaStreamWaitEvent(s_la, e_q2_next_done, 0));
            inner_q1_compute(s_la, cublas_la, cusolver_la, G_la_p, pb_la_p,
                              pw_la_p, info_la_p, S.nccl_col_la, s_comm_la,
                              next_k, sb_next);
            CUDA_CHECK(cudaEventRecord(e_la_q1_done, s_la));

            if (ncols_rest > 0) {
                inner_q2_part(sb, pb_pri, trail_lcol + sb_next_local, ncols_rest,
                               sb_next_local);
            }

            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_la_q1_done, 0));
            // Swap pri/la pointers so the next iteration's Q2 reads from the
            // just-finished panel's broadcast buffer.
            std::swap(G_pri, G_la_p);
            std::swap(pb_pri, pb_la_p);
            std::swap(pw_pri, pw_la_p);
            std::swap(info_pri, info_la_p);

            inner_q1_broadcast(s_comp, pb_pri, next_k, sb_next);
        }
        // The polar reconstruction step below reads from d_panel_bcast (the
        // OG primary buffer), but for QDWH we actually use d_S to extract
        // Q1/Q2 — so the swap above is only meaningful within this inner_qr
        // call.  Resync any pointers that other code outside this lambda
        // expects.
    };

    auto inner_qr = [&]() { if (eff_la) inner_qr_la(); else inner_qr_serial(); };

    auto run_qdwh = [&]() {
        double frob_local;
        size_t na = (size_t)locr * locc;
        CUBLAS_CHECK(cublasDnrm2(cublas, na, d_A, 1, &frob_local));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        frob_local *= frob_local;
        double frob_global = 0.0;
        MPI_Allreduce(&frob_local, &frob_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = std::sqrt(frob_global);
        if (alpha <= 0) alpha = 1.0;
        double inv_alpha = 1.0 / alpha;
        {
            int threads = 256;
            long long total = (long long)locr * locc;
            long long blocks = (total + threads - 1) / threads;
            qdwh_bc25d_update_X<<<(unsigned)blocks, threads, 0, s_comp>>>(
                d_X, d_A, d_A, locr, locc, inv_alpha, 0.0);
        }

        double l = 1.0 / std::sqrt((double)N);
        if (l < 1e-15) l = 1e-15;
        for (int kit = 0; kit < iters; ++kit) {
            double l2 = l * l;
            double dd = std::pow(4.0 * (1.0 - l2) / (l2 * l2), 1.0 / 3.0);
            double sd = std::sqrt(1.0 + dd);
            double inner_term = std::max(0.0, 8.0 - 4.0 * dd + 8.0 * (2.0 - l2) / (l2 * sd));
            double a_k = sd + 0.5 * std::sqrt(inner_term);
            double b_k = (a_k - 1.0) * (a_k - 1.0) / 4.0;
            double c_k = a_k + b_k - 1.0;
            double scale_X = std::sqrt(c_k);

            {
                int threads = 256;
                long long total = (long long)m_st_loc * locc;
                long long blocks = (total + threads - 1) / threads;
                qdwh_bc25d_fill_stacked<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_S, d_X, locr, locc, my_row_in_col_group, my_py, Py, b, scale_X);
            }

            inner_qr();

            // AllGather Q2 (bottom locr rows of d_S) across col_comm.
            CUDA_CHECK(cudaMemcpy2DAsync(d_P, locr * sizeof(double),
                                          d_S + locr, m_st_loc * sizeof(double),
                                          locr * sizeof(double), locc,
                                          cudaMemcpyDeviceToDevice, s_comp));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllGather(d_P, d_Q2_recv,
                                              (size_t)locr * locc, ncclDouble,
                                              S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));

            // AllGather Q1 (top locr rows of d_S) across row_comm.
            CUDA_CHECK(cudaMemcpy2DAsync(d_P, locr * sizeof(double),
                                          d_S, m_st_loc * sizeof(double),
                                          locr * sizeof(double), locc,
                                          cudaMemcpyDeviceToDevice, s_comp));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllGather(d_P, d_Q1_recv,
                                              (size_t)locr * locc, ncclDouble,
                                              S.nccl_row, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));

            // P_loc = Q1 · Q2^T using a single Dgemm over the *full* row
            // band and column band, after rearranging the rank-block
            // AllGather buffers into contiguous column-major layouts.
            // Q1_full: locr × N (= locr × row_size·locc) — rows = my row
            // band, cols = all of Q1's column range.
            // Q2_full: N × locc (= col_size·locr × locc) — rows = all of
            // Q2's row range, cols = my col range.
            // The polar formula P[i, j] = sum_k Q1[i, k] * Q2[k', j]
            // where k = k' iterates over [0, N) — i.e. straightforward
            // matrix product Q1_full · Q2_full = locr × locc.
            //
            // NOTE: Q2_full is the rearrangement of Q2's row band (full N
            // rows, locc cols).  The formula's Q2^T factor is folded into
            // the row/col reinterpretation: see qdwh_bc25d_q2_recv_to_full
            // comment for the derivation.
            int threads = 256;
            {
                long long total_q1 = (long long)row_size * locr * locc;
                long long blocks   = (total_q1 + threads - 1) / threads;
                qdwh_bc25d_q1_recv_to_full<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_Q1_recv, d_Q1_full, (int)locr, (int)locc, row_size);
                long long total_q2 = (long long)col_size * locr * locc;
                blocks = (total_q2 + threads - 1) / threads;
                qdwh_bc25d_q2_recv_to_full<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_Q2_recv, d_Q2_full, (int)locr, (int)locc, col_size);
            }
            int N_full_q1_cols = row_size * (int)locc;       // = N
            int N_full_q2_rows = col_size * (int)locr;       // = N
            (void)N_full_q1_cols;
            CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)locr, (int)locc, N,
                                      &one_d,
                                      d_Q1_full, (int)locr,
                                      d_Q2_full, N_full_q2_rows,
                                      &zero_d, d_P, (int)locr));

            double alpha_x = b_k / c_k;
            double alpha_p = (a_k - b_k / c_k) / std::sqrt(c_k);
            {
                int threads = 256;
                long long total = (long long)locr * locc;
                long long blocks = (total + threads - 1) / threads;
                qdwh_bc25d_update_X<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_Xnew, d_X, d_P, locr, locc, alpha_x, alpha_p);
            }
            std::swap(d_X, d_Xnew);
            double num = l * (a_k + b_k * l * l);
            double den = 1.0 + c_k * l * l;
            l = num / den;
        }
    };

    auto reset_A = [&]() {
        std::vector<double> host((size_t)locr * locc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    };

    if (iters <= 0) iters = 6;
    for (int i = 0; i < 2; ++i) { reset_A(); run_qdwh(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (eff_la) { CUDA_CHECK(cudaStreamSynchronize(s_la));
                  CUDA_CHECK(cudaStreamSynchronize(s_comm_la)); }
    MPI_Barrier(MPI_COMM_WORLD);

    {
        double* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)locc * locc * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  (int)locc, (int)locr, &one_d, d_X, (int)locr,
                                  &zero_d, d_GG, (int)locc));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        FULL25D_NCCL_CHECK(ncclAllReduce(d_GG, d_GG, (size_t)locc * locc, ncclDouble, ncclSum,
                                          S.nccl_col, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        double max_local = 0.0;
        if (S.col_rank == 0) {
            for (std::int64_t lj = 0; lj < locc; ++lj) {
                double dj;
                CUDA_CHECK(cudaMemcpy(&dj, d_GG + lj + lj * locc, sizeof(double),
                                       cudaMemcpyDeviceToHost));
                double dev = std::abs(dj - 1.0);
                if (dev > max_local) max_local = dev;
            }
        }
        double max_global = 0.0;
        MPI_Allreduce(&max_local, &max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (_rank == 0) {
            printf("  Validation     variant=qdwh_bc25d it=%d N=%d b=%d grid=[%d,%d,%d]  max|diag(U'U)-1| = %.2e\n",
                   iters, N, b, Px, Py, Pz, max_global);
            fflush(stdout);
        }
        cudaFree(d_GG);
    }

    const int nrun = 3;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset_A();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qdwh();
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        if (eff_la) { CUDA_CHECK(cudaStreamSynchronize(s_la));
                      CUDA_CHECK(cudaStreamSynchronize(s_comm_la)); }
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        const char* matnm =
            use_tf32_trail ? "fp64mp_tf32" :
            use_mp_trail   ? "fp64mp"      :
                              "fp64";
        printf("  qdwh_bc25d%s it=%d  matrix=%s N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               eff_la ? "+LA" : "", iters, matnm, N, b, Px, Py, Pz, times[0], tmed);
        printf("METRICS bench=qdwh_bc25d matrix=%s layout=bc_2p5d%s N=%d b=%d Px=%d Py=%d Pz=%d passes=%d ours_ms=%.4f\n",
               matnm, eff_la ? " panel=cqr2+la" : "", N, b, Px, Py, Pz, iters, tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Xnew);
    cudaFree(d_S); cudaFree(d_G); cudaFree(d_W); cudaFree(d_panel_bcast);
    cudaFree(d_potrf_work); cudaFree(d_info);
    cudaFree(d_Q2_recv); cudaFree(d_Q1_recv); cudaFree(d_P);
    cudaFree(d_Q1_full); cudaFree(d_Q2_full);
    if (eff_la) {
        cudaFree(d_G_la); cudaFree(d_panel_bcast_la);
        cudaFree(d_potrf_work_la); cudaFree(d_info_la);
        cublasDestroy(cublas_la); cusolverDnDestroy(cusolver_la);
        cudaStreamDestroy(s_la); cudaStreamDestroy(s_comm_la);
    }
    if (use_mp_trail) { cudaFree(d_panel_bcast_f); cudaFree(d_S_trail_f); cudaFree(d_W_f); }
    cublasDestroy(cublas); cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_q2_next_done); cudaEventDestroy(e_la_q1_done);
    return 0;
}

#endif  // QDWH_BC25D_INL
