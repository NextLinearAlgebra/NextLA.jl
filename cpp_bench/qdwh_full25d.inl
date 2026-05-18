// qdwh_full25d.inl — Path-q full 2.5D (Pz>1, Px*Py>1) FP64 runner.
//
// QDWH polar decomposition (Nakatsukasa-Higham 2013) on the [Px, Py, Pz]
// grid from full25d_grid.hpp.  Each Halley iteration runs an inner CQR2
// on the 2n × n stacked matrix S = [sqrt(c_k) X; I_n], using:
//   * col_comm (size Px*Pz) for the inner-QR Phase Q3 AllReduce(G) and
//     Phase Q4 AllReduce(W);
//   * row_comm (size Py)    for the panel Q broadcast across py-blocks.
// After the inner QR, Q_1 (top n rows) and Q_2 (bottom n rows) are
// extracted from the stacked rank-local layout.  The outer
// P := Q_1 Q_2^T is computed as two AllGathers + one local DGEMM:
//   AllGather Q_2 within col_comm  → rank gets N × n_loc block of Q_2;
//   AllGather Q_1 within row_comm  → rank gets m_loc × N block of Q_1;
//   P_loc = Q_1_full · (Q_2_full)^T  is m_loc × n_loc, all local.
// Then update X_{k+1} = (b_k/c_k) X + (a_k - b_k/c_k)/sqrt(c_k) P.
//
// Stacked-matrix interleaving: rank (px, py, pz) holds 2*m_loc rows of S,
// with rows split as [top m_loc = sqrt(c_k) X_loc;  bottom m_loc = identity
// rows with global indices (pz*Px + px)*m_loc .. (pz*Px+px+1)*m_loc - 1].
// QR is row-permutation-invariant so this interleaving is exact.

#ifndef QDWH_FULL25D_INL
#define QDWH_FULL25D_INL

#include "full25d_grid.hpp"
#include "full25d_kernels.cuh"
#include "bench_vendor_metrics.hpp"

__global__ static void qdwh_f25d_fill_stacked_kernel(double* __restrict__ S,
                                                       const double* __restrict__ Xk,
                                                       int m_loc, int n_loc,
                                                       int my_row_in_col_group,
                                                       int my_py, int Py,
                                                       double scale_X) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)2 * m_loc * n_loc;
    if (idx >= total) return;
    int i = (int)(idx % (2 * m_loc));
    int j = (int)(idx / (2 * m_loc));
    if (i < m_loc) {
        S[idx] = scale_X * Xk[i + (long long)j * m_loc];
    } else {
        // Identity rows: global row index = my_row_in_col_group * m_loc + (i - m_loc),
        // global col index = my_py * n_loc + j.
        int row_in_I = my_row_in_col_group * m_loc + (i - m_loc);
        int col_in_I = my_py * n_loc + j;
        S[idx] = (row_in_I == col_in_I) ? 1.0 : 0.0;
    }
}

__global__ static void qdwh_f25d_update_X_kernel(double* __restrict__ Xnew,
                                                  const double* __restrict__ Xk,
                                                  const double* __restrict__ P,
                                                  int m_loc, int n_loc,
                                                  double alpha_x, double alpha_p) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_loc * n_loc;
    if (idx >= total) return;
    Xnew[idx] = alpha_x * Xk[idx] + alpha_p * P[idx];
}

static int run_qdwh_full25d_fp64(const Args& A_args,
                                   const Full25DGrid& G,
                                   const Full25DSubcomms& S_subc) {
    int N = A_args.N, b = A_args.b;
    int m_loc = G.m_loc, n_loc = G.n_loc;
    int Px = G.Px, Py = G.Py, Pz = G.Pz;
    int my_py = G.my_py;
    int col_size = S_subc.col_size;
    int row_size = S_subc.row_size;
    int m_st_loc = 2 * m_loc;

    cudaStream_t s_comp, s_comm;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    cudaEvent_t e_comp_done, e_ar_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,   cudaEventDisableTiming));
    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cusolverDnHandle_t cusolver; CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s_comp));

    double *d_A = nullptr, *d_X = nullptr, *d_Xnew = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)m_loc * n_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,    (size_t)m_loc * n_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Xnew, (size_t)m_loc * n_loc * sizeof(double)));
    {
        std::vector<double> host((size_t)m_loc * n_loc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Stacked matrix S = [sqrt(c) X; I], distributed as 2*m_loc x n_loc per rank.
    double* d_S = nullptr;
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)m_st_loc * n_loc * sizeof(double)));

    // Inner-QR scratch (CQR2 style on stacked S).
    double *d_G = nullptr, *d_W = nullptr, *d_panel_bcast = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G,           (size_t)b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,           (size_t)b * n_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel_bcast, (size_t)m_st_loc * b * sizeof(double)));
    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Outer Q1*Q2^T compute scratch.
    //   d_Q2_recv:  AllGather Q2 (m_loc x n_loc per rank) within col_comm
    //                → N × n_loc on every col_group rank.
    //   d_Q1_recv:  AllGather Q1 within row_comm
    //                → m_loc × N on every row_group rank.
    //   d_P:        local Q1*Q2^T result, m_loc × n_loc.
    double *d_Q2_recv = nullptr, *d_Q1_recv = nullptr, *d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q2_recv, (size_t)m_loc * n_loc * (size_t)col_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q1_recv, (size_t)m_loc * n_loc * (size_t)row_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P,       (size_t)m_loc * n_loc * sizeof(double)));

    const double one_d = 1.0, zero_d = 0.0, neg_one_d = -1.0;

    auto sb_clipped = [&](int k) -> int {
        return sb_clipped_full25d(k, b, N, n_loc);
    };

    // Inner CQR2 Phase Q1 on stacked S (m_st_loc rows × n_loc cols per rank).
    auto inner_q1 = [&](int k, int sb) {
        int py_panel = k / n_loc;
        int local_k  = k - py_panel * n_loc;
        if (my_py == py_panel) {
            double* S_panel = d_S + (size_t)local_k * m_st_loc;
            for (int it = 0; it < 2; ++it) {
                CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                          sb, m_st_loc, &one_d, S_panel, m_st_loc,
                                          &zero_d, d_G, b));
                CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
                CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
                FULL25D_NCCL_CHECK(ncclAllReduce(d_G, d_G, (size_t)b * b, ncclDouble, ncclSum,
                                                  S_subc.nccl_col, s_comm));
                CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
                CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
                CUSOLVER_CHECK(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G, b,
                                                  d_potrf_work, potrf_lwork, d_info));
                CUBLAS_CHECK(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                          CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                          m_st_loc, sb, &one_d, d_G, b, S_panel, m_st_loc));
            }
            CUDA_CHECK(cudaMemcpyAsync(d_panel_bcast, S_panel,
                                        (size_t)m_st_loc * sb * sizeof(double),
                                        cudaMemcpyDeviceToDevice, s_comp));
        }
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclBroadcast(d_panel_bcast, d_panel_bcast,
                                          (size_t)m_st_loc * sb, ncclDouble,
                                          py_panel, S_subc.nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
    };

    // Inner Phase Q2 trailing on stacked S.
    auto inner_q2 = [&](int k, int sb) {
        int my_col_start_global = my_py * n_loc;
        int trail_global_start  = k + sb;
        int local_start = std::max(0, trail_global_start - my_col_start_global);
        int local_end   = n_loc;
        int ncols = local_end - local_start;
        if (ncols <= 0) return;
        double* d_S_trail = d_S + (size_t)local_start * m_st_loc;
        CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  sb, ncols, m_st_loc,
                                  &one_d, d_panel_bcast, m_st_loc, d_S_trail, m_st_loc,
                                  &zero_d, d_W, b));
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum,
                                          S_subc.nccl_col, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
        CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m_st_loc, ncols, sb,
                                  &neg_one_d, d_panel_bcast, m_st_loc, d_W, b,
                                  &one_d, d_S_trail, m_st_loc));
    };

    auto inner_qr = [&]() {
        int k = 0;
        while (k < N) {
            int sb = sb_clipped(k);
            inner_q1(k, sb);
            if (k + sb < N) inner_q2(k, sb);
            k += sb;
        }
    };

    // QDWH outer Halley loop.
    int iters = A_args.iters;
    if (iters <= 0) iters = 6;
    int my_row_in_col_group = S_subc.col_rank;
    auto run_qdwh = [&]() {
        // X_0 = A / ||A||_F (approximate sigma_max bound).
        double frob_local;
        size_t na = (size_t)m_loc * n_loc;
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
            long long total = (long long)m_loc * n_loc;
            long long blocks = (total + threads - 1) / threads;
            qdwh_f25d_update_X_kernel<<<(unsigned)blocks, threads, 0, s_comp>>>(
                d_X, d_A, d_A, m_loc, n_loc, inv_alpha, 0.0);
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

            // Build stacked S.
            {
                int threads = 256;
                long long total = (long long)m_st_loc * n_loc;
                long long blocks = (total + threads - 1) / threads;
                qdwh_f25d_fill_stacked_kernel<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_S, d_X, m_loc, n_loc, my_row_in_col_group, my_py, Py, scale_X);
            }

            // Inner QR on stacked S → d_S overwritten with Q (top m_loc=Q1, bottom m_loc=Q2 per rank).
            inner_qr();

            // AllGather Q2 (bottom m_loc rows of d_S) across col_comm → N × n_loc on each rank.
            //   Pack Q2 contiguously: cudaMemcpy2DAsync from d_S + m_loc (stride m_st_loc)
            //   into d_panel_bcast layout-of-size m_loc × n_loc contig.  We reuse d_panel_bcast
            //   as the temporary contig pack since size m_st_loc * b >= m_loc * n_loc when b >= n_loc/2.
            //   Safer: allocate a dedicated pack buffer.
            CUDA_CHECK(cudaMemcpy2DAsync(d_P, m_loc * sizeof(double),
                                          d_S + m_loc, m_st_loc * sizeof(double),
                                          m_loc * sizeof(double), n_loc,
                                          cudaMemcpyDeviceToDevice, s_comp));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllGather(d_P, d_Q2_recv,
                                              (size_t)m_loc * n_loc, ncclDouble,
                                              S_subc.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));

            // AllGather Q1 (top m_loc rows of d_S) across row_comm → m_loc × N on each rank.
            CUDA_CHECK(cudaMemcpy2DAsync(d_P, m_loc * sizeof(double),
                                          d_S, m_st_loc * sizeof(double),
                                          m_loc * sizeof(double), n_loc,
                                          cudaMemcpyDeviceToDevice, s_comp));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllGather(d_P, d_Q1_recv,
                                              (size_t)m_loc * n_loc, ncclDouble,
                                              S_subc.nccl_row, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));

            // Now d_Q2_recv has Q2 layout: col_size m_loc × n_loc slabs concatenated.
            // For the GEMM, treat it as N × n_loc column-major with ld=m_loc (per-slab) — but
            // memory is contiguous so we need block-by-block multiplies (col_size blocks of
            // m_loc × n_loc).  d_Q1_recv has row_size m_loc × n_loc slabs; for the (m_loc × N)
            // result we treat each row_size block as an m_loc × n_loc column-major piece.
            //
            // P_loc[i, j] = sum over (r_row, k_r) and (r_col, l_c):
            //   Q1[i + 0 .. , k_r + r_row*n_loc] * Q2[j + 0 .. , l_c + r_col*m_loc]
            // The straightforward decomposition: sum over col_size blocks for the "k" reduction
            // dimension (Q1's columns), with each block also requiring the right Q2 slab.

            // Simpler: do c (= col_size) block-multiplies, each with one piece of Q1 and the
            // matching piece of Q2 in Q2_recv.
            // Q1 block r_row: stored at d_Q1_recv + r_row * m_loc * n_loc, layout m_loc × n_loc
            //   col-major, ld=m_loc. Corresponds to Q1 rows [my_pxpz*m_loc : ...], cols [r_row*n_loc : ...].
            // Q2 block r_col: stored at d_Q2_recv + r_col * m_loc * n_loc, layout m_loc × n_loc
            //   col-major, ld=m_loc. Corresponds to Q2 rows [r_col*m_loc : ...], cols [my_py*n_loc : ...].
            //
            // Q2 has N rows; row blocks r_col cover Q2 rows [r_col*m_loc : (r_col+1)*m_loc).
            // For P[i, j] = sum_k Q1[i, k] * Q2[j, k]:
            //   Q1[i, k] has k spanning N columns split by row_size = Py: k in py'*n_loc range.
            //   Q1 block r_row stores Q1 columns [r_row*n_loc : (r_row+1)*n_loc).
            //   k_idx_in_block = k - r_row*n_loc.
            //
            // So for each row_size block, we contribute a partial sum to P_loc.  But we also
            // need to recombine Q2 according to the global k index.  Q2 is N rows (col_size = Px*Pz
            // blocks of m_loc); given the rank's col_group's m_loc-row partition, the k index
            // in Q2 row index maps to row block r_col = k / m_loc and i_in_block = k % m_loc.
            // For a row_size block contributing Q1's k range [r_row*n_loc, (r_row+1)*n_loc):
            //   r_col = (r_row*n_loc + k_in_block) / m_loc  for k_in_block in [0, n_loc).
            //
            // If n_loc == m_loc the partition aligns cleanly (each Q1-block matches one Q2-block).
            // Otherwise (n_loc != m_loc, e.g. Px*Pz != Py), we have a more complex mapping.
            //
            // For our Px=Py and Pz=1 case n_loc == m_loc.  For c=1 with Px=Py=2 (P=4) this is
            // satisfied.  For Px=Py=Pz=2 (P=8) we have m_loc = N/4, n_loc = N/2: n_loc = 2*m_loc,
            // so each Q1 row_size block (n_loc cols) spans TWO Q2 row_size blocks.
            //
            // For simplicity (and full validity at Px=Py with arbitrary Pz), we do a single
            // "fused" GEMM: assemble Q2 into N × n_loc column-major (ld=N) and Q1 into m_loc × N
            // (ld=m_loc) by physical rearrangement, then GEMM.  Rearrange via cublasDgeam strided
            // copies (NN -> NN) one block at a time.

            // Rearrange d_Q2_recv (col_size m_loc × n_loc blocks concatenated) into a single
            // N × n_loc column-major matrix at d_Q2_recv via in-place block packing? Memory layout:
            //   d_Q2_recv[ r * m_loc*n_loc + j*m_loc + i ]  --- rank r block
            // Target: d_Q2_full[(r*m_loc + i) + j*N]  ld=N
            // Index mapping:  src_off = r*m_loc*n_loc + j*m_loc + i  ↔  dst_off = r*m_loc + i + j*N
            // Different unless m_loc * (n_loc - 1) + n_loc * m_loc terms align — they don't in general.
            // → use a small kernel to rearrange both Q1 and Q2 into the desired single-matrix layouts.

            // Block-by-block GEMMs (correct for arbitrary Px*Pz and Py).  We compute
            //   P_loc += Q1_block_rrow @ Q2_block_rcol_of_rrow^T  for each (r_row, r_col) pairing
            //   that maps the same global k-range.
            // Since both AllGathers preserve the rank ordering of their respective sub-comms, and
            // col_size = row_size only when Px*Pz == Py (square Pz=1 case), the cleanest correct
            // GEMM for the general 2×2×2 case requires fewer constraints.
            //
            // Implementation here: assume Px*Pz == Py (true when c=1 and Px=Py, or when c=Pz=any
            // with the derived grid that picks Px=Py).  Under that assumption the col and row
            // sub-comms have equal size, m_loc == n_loc, and we do col_size GEMMs accumulating
            // into d_P.
            if (m_loc != n_loc || col_size != row_size) {
                if (_rank == 0)
                    fprintf(stderr, "qdwh_full25d: requires m_loc==n_loc (col_size==row_size); "
                                    "got m_loc=%d n_loc=%d col_size=%d row_size=%d. "
                                    "Use a (Px=Py, Pz=anything) grid.\n",
                            m_loc, n_loc, col_size, row_size);
                MPI_Abort(MPI_COMM_WORLD, 92);
            }
            // m_loc == n_loc, col_size == row_size.  Q1 row_size blocks have n_loc cols each;
            // Q2 col_size blocks have m_loc rows each.  Block r maps to the same global k range
            // for both (because the col_comm and row_comm have matched key ordering when
            // Pz>1 with Px=Py).
            // P_loc = sum over r of Q1_recv[r] · Q2_recv[r]^T   (each m_loc × m_loc).
            CUBLAS_CHECK(cublasDscal(cublas, (int)((size_t)m_loc * n_loc), &zero_d, d_P, 1));
            for (int r = 0; r < col_size; ++r) {
                const double* Q1_blk = d_Q1_recv + (size_t)r * m_loc * n_loc;
                const double* Q2_blk = d_Q2_recv + (size_t)r * m_loc * n_loc;
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                          m_loc, n_loc, n_loc,
                                          &one_d, Q1_blk, m_loc, Q2_blk, m_loc,
                                          &one_d, d_P, m_loc));
            }

            // Update X.
            double alpha_x = b_k / c_k;
            double alpha_p = (a_k - b_k / c_k) / std::sqrt(c_k);
            {
                int threads = 256;
                long long total = (long long)m_loc * n_loc;
                long long blocks = (total + threads - 1) / threads;
                qdwh_f25d_update_X_kernel<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_Xnew, d_X, d_P, m_loc, n_loc, alpha_x, alpha_p);
            }
            std::swap(d_X, d_Xnew);
            double num = l * (a_k + b_k * l * l);
            double den = 1.0 + c_k * l * l;
            l = num / den;
        }
    };

    auto reset_A = [&]() {
        std::vector<double> host((size_t)m_loc * n_loc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    };

    for (int i = 0; i < 2; ++i) { reset_A(); run_qdwh(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    MPI_Barrier(MPI_COMM_WORLD);

    {
        double* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)n_loc * n_loc * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  n_loc, m_loc, &one_d, d_X, m_loc,
                                  &zero_d, d_GG, n_loc));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        FULL25D_NCCL_CHECK(ncclAllReduce(d_GG, d_GG, (size_t)n_loc * n_loc, ncclDouble, ncclSum,
                                          S_subc.nccl_col, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        double max_local = 0.0;
        if (S_subc.col_rank == 0) {
            for (int j = 0; j < n_loc; ++j) {
                double dj;
                CUDA_CHECK(cudaMemcpy(&dj, d_GG + j + (size_t)j * n_loc, sizeof(double),
                                       cudaMemcpyDeviceToHost));
                double dev = std::abs(dj - 1.0);
                if (dev > max_local) max_local = dev;
            }
        }
        double max_global = 0.0;
        MPI_Allreduce(&max_local, &max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (_rank == 0) {
            printf("  Validation     variant=qdwh_fp64_p25d_it=%d N=%d grid=[%d,%d,%d]  max|diag(U'U)-1| = %.2e\n",
                   iters, N, Px, Py, Pz, max_global);
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
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  qdwh_fp64_p25d_it=%d  N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               iters, N, b, Px, Py, Pz, times[0], tmed);
        printf("METRICS bench=qdwh_full25d matrix=fp64 layout=full25d N=%d b=%d Px=%d Py=%d Pz=%d passes=%d ours_ms=%.4f\n",
               N, b, Px, Py, Pz, iters, tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Xnew);
    cudaFree(d_S); cudaFree(d_G); cudaFree(d_W); cudaFree(d_panel_bcast);
    cudaFree(d_potrf_work); cudaFree(d_info);
    cudaFree(d_Q2_recv); cudaFree(d_Q1_recv); cudaFree(d_P);
    cublasDestroy(cublas); cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    return 0;
}

#endif  // QDWH_FULL25D_INL
