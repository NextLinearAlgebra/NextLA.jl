// givens_full25d.inl — Path-g full 2.5D (Pz>1, Px*Py>1) FP64 runner.
//
// Same column-group / row-group skeleton as householder_full25d.inl, but
// the Phase Q1 panel factorization is the tournament-Givens parallel
// kernel from givens_2p5d_bench.cu (Θ(b log m) depth) instead of cuSolver
// Dgeqrf + Dorgqr.  After the kernel, form_Q_from_givens_kernel replays
// the rotation list on I_m[:, 0:b] to materialize the thin Q.  The
// trailing update is identical to Path-h.
//
// Requires: `givens_panel_kernel(double*, m, b, lda, cs)`,
//           `init_thinQ_identity(double*, m, b)`,
//           `form_Q_from_givens_kernel(double*, m, b, ldq, cs)`,
//           `f25d_rearrange_recv_to_panel(...)` defined before include.

#ifndef GIVENS_FULL25D_INL
#define GIVENS_FULL25D_INL

#include "full25d_grid.hpp"
#include "full25d_kernels.cuh"
#include "bench_vendor_metrics.hpp"
#include "nextla_mp_trail.hpp"

__global__ static void gg_f25d_d2f(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ static void gg_f25d_f2d(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}
__global__ static void gg_f25d_rearrange_recv_to_panel_f(const float* __restrict__ recv,
                                                          float* __restrict__ full,
                                                          int m_loc, int sb, int /*P_col*/,
                                                          int m_total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_total * sb;
    if (idx >= total) return;
    int i = (int)(idx % m_total);
    int j = (int)(idx / m_total);
    int r = i / m_loc;
    int i_local = i - r * m_loc;
    full[idx] = recv[(long long)r * m_loc * sb + (long long)j * m_loc + i_local];
}

static int run_givens_full25d_fp64(const Args& A_args,
                                    const Full25DGrid& G,
                                    const Full25DSubcomms& S,
                                    bool use_mp_trail = false,
                                    bool use_tf32_trail = false) {
    int N = A_args.N, b = A_args.b;
    int m_loc = G.m_loc, n_loc = G.n_loc;
    int Px = G.Px, Py = G.Py, Pz = G.Pz;
    int my_py = G.my_py;
    int col_size = S.col_size;

    cudaStream_t s_comp, s_comm, s_la;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    const bool eff_la = A_args.lookahead;
    if (eff_la) CUDA_CHECK(cudaStreamCreate(&s_la));
    cudaEvent_t e_comp_done, e_ar_done, e_panel_done, e_next_ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,    cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_panel_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_next_ready, cudaEventDisableTiming));

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cublasHandle_t cublas_la{};
    if (eff_la) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));
        CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
    }

    double *d_A = nullptr, *d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,      (size_t)m_loc * n_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_loc * n_loc * sizeof(double)));
    {
        std::vector<double> host((size_t)m_loc * n_loc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A, (size_t)m_loc * n_loc * sizeof(double), cudaMemcpyDeviceToDevice));

    double *d_panel_recv = nullptr, *d_panel_full = nullptr, *d_Q_full = nullptr;
    double *d_cs = nullptr;       // 2 * b * (m-1) entries for tournament Givens
    CUDA_CHECK(cudaMalloc(&d_panel_recv, (size_t)m_loc * b * (size_t)col_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel_full, (size_t)N * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q_full,     (size_t)N * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cs,         (size_t)2 * b * (size_t)(N - 1) * sizeof(double)));

    double *d_panel_bcast = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel_bcast, (size_t)m_loc * b * sizeof(double)));
    double* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * n_loc * sizeof(double)));

    float *d_panel_bcast_f = nullptr, *d_A_trail_f = nullptr, *d_W_f = nullptr;
    if (use_mp_trail) {
        CUDA_CHECK(cudaMalloc(&d_panel_bcast_f, (size_t)m_loc * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_trail_f,     (size_t)m_loc * n_loc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_f,           (size_t)b * n_loc * sizeof(float)));
        if (use_tf32_trail) {
#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
            CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));
#endif
        }
    }

    double *d_panel_recv2=nullptr, *d_panel_full2=nullptr, *d_Q_full2=nullptr,
           *d_cs2=nullptr, *d_panel_bcast2=nullptr;
    if (eff_la) {
        CUDA_CHECK(cudaMalloc(&d_panel_recv2,  (size_t)m_loc * b * (size_t)col_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_full2,  (size_t)N * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Q_full2,      (size_t)N * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cs2,          (size_t)2 * b * (size_t)(N - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_bcast2, (size_t)m_loc * b * sizeof(double)));
    }

    const double one_d = 1.0, zero_d = 0.0, neg_one_d = -1.0;

    auto sb_clipped = [&](int k) -> int {
        return sb_clipped_full25d(k, b, N, n_loc);
    };

    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t /*cb*/,
                        double* p_recv, double* p_full, double* p_Q,
                        double* cs_buf, double* p_bcast,
                        int k, int sb) {
        int py_panel = k / n_loc;
        int local_k  = k - py_panel * n_loc;
        if (my_py == py_panel) {
            // 1. AllGather within column group → full N × sb panel rearranged.
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllGather(d_A + (size_t)local_k * m_loc,
                                              p_recv,
                                              (size_t)m_loc * sb, ncclDouble,
                                              S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            {
                long long total = (long long)N * sb;
                int threads = 256;
                long long blocks = (total + threads - 1) / threads;
                f25d_rearrange_recv_to_panel<<<(unsigned)blocks, threads, 0, s_use>>>(
                    p_recv, p_full, m_loc, sb, col_size, N);
            }
            // 2. Tournament-Givens panel kernel (replicated).
            givens_panel_kernel<<<1, 256, 0, s_use>>>(p_full, N, sb, N, cs_buf);
            // 3. Materialize Q.
            {
                long long total = (long long)N * sb;
                int threads = 256;
                long long blocks = (total + threads - 1) / threads;
                init_thinQ_identity<<<(unsigned)blocks, threads, 0, s_use>>>(p_Q, N, sb);
                form_Q_from_givens_kernel<<<1, 256, 0, s_use>>>(p_Q, N, sb, N, cs_buf);
            }
            // 4. Copy this rank's Q slice back into local A + p_bcast.
            int my_row_in_col_group = S.col_rank;
            CUDA_CHECK(cudaMemcpy2DAsync(d_A + (size_t)local_k * m_loc, (size_t)m_loc * sizeof(double),
                                          p_Q + (size_t)my_row_in_col_group * m_loc,
                                          (size_t)N * sizeof(double),
                                          (size_t)m_loc * sizeof(double), (size_t)sb,
                                          cudaMemcpyDeviceToDevice, s_use));
            CUDA_CHECK(cudaMemcpyAsync(p_bcast, d_A + (size_t)local_k * m_loc,
                                        (size_t)m_loc * sb * sizeof(double),
                                        cudaMemcpyDeviceToDevice, s_use));
        }
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclBroadcast(p_bcast, p_bcast, (size_t)m_loc * sb, ncclDouble,
                                          py_panel, S.nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
    };

    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;
    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb,
                        double* p_bcast, int k, int sb) {
        int my_col_start_global = my_py * n_loc;
        int trail_global_start  = k + sb;
        int local_start = std::max(0, trail_global_start - my_col_start_global);
        int local_end   = n_loc;
        int ncols = local_end - local_start;
        if (ncols <= 0) return;
        double* d_A_trail = d_A + (size_t)local_start * m_loc;
        if (!use_mp_trail) {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_loc,
                                      &one_d, p_bcast, m_loc, d_A_trail, m_loc,
                                      &zero_d, d_W, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum,
                                              S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_loc, ncols, sb,
                                      &neg_one_d, p_bcast, m_loc, d_W, b,
                                      &one_d, d_A_trail, m_loc));
        } else {
            size_t np = (size_t)m_loc * sb, nt = (size_t)m_loc * ncols;
            gg_f25d_d2f<<<(np + 255)/256, 256, 0, s_use>>>(p_bcast, d_panel_bcast_f, np);
            gg_f25d_d2f<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_trail, d_A_trail_f, nt);
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_loc,
                                      &one_f, d_panel_bcast_f, m_loc, d_A_trail_f, m_loc,
                                      &zero_f, d_W_f, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(d_W_f, d_W_f, (size_t)sb * ncols, ncclFloat, ncclSum,
                                              S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_loc, ncols, sb,
                                      &neg_one_f, d_panel_bcast_f, m_loc, d_W_f, b,
                                      &one_f, d_A_trail_f, m_loc));
            gg_f25d_f2d<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_trail_f, d_A_trail, nt);
        }
    };

    auto run_qr = [&]() {
        int k = 0;
        if (!eff_la) {
            while (k < N) {
                int sb = sb_clipped(k);
                phase_q1(s_comp, cublas, d_panel_recv, d_panel_full, d_Q_full, d_cs, d_panel_bcast, k, sb);
                if (k + sb < N) phase_q2(s_comp, cublas, d_panel_bcast, k, sb);
                k += sb;
            }
        } else {
            int sb = sb_clipped(k);
            phase_q1(s_comp, cublas, d_panel_recv, d_panel_full, d_Q_full, d_cs, d_panel_bcast, k, sb);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? sb_clipped(next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    phase_q2(s_comp, cublas, d_panel_bcast, k, sb);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, d_panel_recv2, d_panel_full2, d_Q_full2, d_cs2, d_panel_bcast2, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                    CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                    std::swap(d_panel_bcast, d_panel_bcast2);
                    std::swap(d_panel_recv,  d_panel_recv2);
                    std::swap(d_panel_full,  d_panel_full2);
                    std::swap(d_Q_full,      d_Q_full2);
                    std::swap(d_cs,          d_cs2);
                    k = next_k; sb = next_sb;
                } else {
                    if (n_rest > 0) phase_q2(s_comp, cublas, d_panel_bcast, k, sb);
                    break;
                }
            }
        }
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, (size_t)m_loc * n_loc * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) { reset_A(); run_qr(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    // Validation
    {
        double* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)n_loc * n_loc * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  n_loc, m_loc, &one_d, d_A, m_loc,
                                  &zero_d, d_GG, n_loc));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        FULL25D_NCCL_CHECK(ncclAllReduce(d_GG, d_GG, (size_t)n_loc * n_loc, ncclDouble, ncclSum,
                                          S.nccl_col, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        double max_local = 0.0;
        if (S.col_rank == 0) {
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
            printf("  Validation     variant=gg_fp64_p25d N=%d grid=[%d,%d,%d]  max|diag(Q'Q)-1| = %.2e\n",
                   N, Px, Py, Pz, max_global);
            fflush(stdout);
        }
        cudaFree(d_GG);
    }

    const int nrun = 5;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset_A();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qr();
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  gg_fp64_p25d%s  N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               eff_la ? "+LA" : "", N, b, Px, Py, Pz, times[0], tmed);
        printf("METRICS bench=givens_full25d matrix=fp64 layout=bc_2p5d N=%d b=%d Px=%d Py=%d Pz=%d passes=1 ours_ms=%.4f\n",
               N, b, Px, Py, Pz, tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_A_orig);
    cudaFree(d_panel_recv); cudaFree(d_panel_full); cudaFree(d_Q_full); cudaFree(d_cs);
    cudaFree(d_panel_bcast); cudaFree(d_W);
    if (use_mp_trail) { cudaFree(d_panel_bcast_f); cudaFree(d_A_trail_f); cudaFree(d_W_f); }
    if (eff_la) {
        cudaFree(d_panel_recv2); cudaFree(d_panel_full2); cudaFree(d_Q_full2);
        cudaFree(d_cs2); cudaFree(d_panel_bcast2);
        cublasDestroy(cublas_la);
    }
    cublasDestroy(cublas);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm);
    if (eff_la) cudaStreamDestroy(s_la);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done); cudaEventDestroy(e_next_ready);
    return 0;
}

// FP32-full Path-g full-2.5D runner.  Uses the tournament-Givens float
// kernels (givens_panel_kernel_f / form_Q_from_givens_kernel_f) already
// defined in givens_2p5d_bench.cu.
static int run_givens_full25d_fp32(const Args& A_args,
                                    const Full25DGrid& G,
                                    const Full25DSubcomms& S) {
    int N = A_args.N, b = A_args.b;
    int m_loc = G.m_loc, n_loc = G.n_loc;
    int Px = G.Px, Py = G.Py, Pz = G.Pz;
    int my_py = G.my_py;
    int col_size = S.col_size;

    cudaStream_t s_comp, s_comm, s_la;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    const bool eff_la = A_args.lookahead;
    if (eff_la) CUDA_CHECK(cudaStreamCreate(&s_la));
    cudaEvent_t e_comp_done, e_ar_done, e_panel_done, e_next_ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,    cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_panel_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_next_ready, cudaEventDisableTiming));
    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas)); CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cublasHandle_t cublas_la{};
    if (eff_la) { CUBLAS_CHECK(cublasCreate(&cublas_la)); CUBLAS_CHECK(cublasSetStream(cublas_la, s_la)); }

    float *d_A = nullptr, *d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,      (size_t)m_loc * n_loc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_loc * n_loc * sizeof(float)));
    {
        std::vector<float> host((size_t)m_loc * n_loc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<float> nrm(0.f, 1.f);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A, (size_t)m_loc * n_loc * sizeof(float), cudaMemcpyDeviceToDevice));

    float *d_panel_recv = nullptr, *d_panel_full = nullptr, *d_Q_full = nullptr;
    float *d_cs = nullptr, *d_panel_bcast = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel_recv,  (size_t)m_loc * b * (size_t)col_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_panel_full,  (size_t)N * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_full,      (size_t)N * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cs,          (size_t)2 * b * (size_t)(N - 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_panel_bcast, (size_t)m_loc * b * sizeof(float)));
    float* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * n_loc * sizeof(float)));

    float *d_panel_recv2=nullptr, *d_panel_full2=nullptr, *d_Q_full2=nullptr,
          *d_cs2=nullptr, *d_panel_bcast2=nullptr;
    if (eff_la) {
        CUDA_CHECK(cudaMalloc(&d_panel_recv2,  (size_t)m_loc * b * (size_t)col_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_panel_full2,  (size_t)N * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q_full2,      (size_t)N * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cs2,          (size_t)2 * b * (size_t)(N - 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_panel_bcast2, (size_t)m_loc * b * sizeof(float)));
    }

    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;
    auto sb_clipped = [&](int k) -> int { return sb_clipped_full25d(k, b, N, n_loc); };

    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t /*cb*/,
                        float* p_recv, float* p_full, float* p_Q, float* cs_buf, float* p_bcast,
                        int k, int sb) {
        int py_panel = k / n_loc;
        int local_k  = k - py_panel * n_loc;
        if (my_py == py_panel) {
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllGather(d_A + (size_t)local_k * m_loc, p_recv,
                                              (size_t)m_loc * sb, ncclFloat, S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            {
                long long total = (long long)N * sb;
                int threads = 256;
                long long blocks = (total + threads - 1) / threads;
                gg_f25d_rearrange_recv_to_panel_f<<<(unsigned)blocks, threads, 0, s_use>>>(
                    p_recv, p_full, m_loc, sb, col_size, N);
            }
            givens_panel_kernel_f<<<1, 256, 0, s_use>>>(p_full, N, sb, N, cs_buf);
            {
                long long total = (long long)N * sb;
                int threads = 256;
                long long blocks = (total + threads - 1) / threads;
                init_thinQ_identity_f<<<(unsigned)blocks, threads, 0, s_use>>>(p_Q, N, sb);
                form_Q_from_givens_kernel_f<<<1, 256, 0, s_use>>>(p_Q, N, sb, N, cs_buf);
            }
            int my_row_in_col_group = S.col_rank;
            CUDA_CHECK(cudaMemcpy2DAsync(d_A + (size_t)local_k * m_loc, (size_t)m_loc * sizeof(float),
                                          p_Q + (size_t)my_row_in_col_group * m_loc,
                                          (size_t)N * sizeof(float),
                                          (size_t)m_loc * sizeof(float), (size_t)sb,
                                          cudaMemcpyDeviceToDevice, s_use));
            CUDA_CHECK(cudaMemcpyAsync(p_bcast, d_A + (size_t)local_k * m_loc,
                                        (size_t)m_loc * sb * sizeof(float),
                                        cudaMemcpyDeviceToDevice, s_use));
        }
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclBroadcast(p_bcast, p_bcast, (size_t)m_loc * sb, ncclFloat,
                                          py_panel, S.nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
    };

    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb, float* p_bcast, int k, int sb) {
        int my_col_start_global = my_py * n_loc;
        int trail_global_start  = k + sb;
        int local_start = std::max(0, trail_global_start - my_col_start_global);
        int local_end   = n_loc;
        int ncols = local_end - local_start;
        if (ncols <= 0) return;
        float* d_A_trail = d_A + (size_t)local_start * m_loc;
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                  sb, ncols, m_loc,
                                  &one_f, p_bcast, m_loc, d_A_trail, m_loc,
                                  &zero_f, d_W, b));
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclFloat, ncclSum, S.nccl_col, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m_loc, ncols, sb,
                                  &neg_one_f, p_bcast, m_loc, d_W, b,
                                  &one_f, d_A_trail, m_loc));
    };

    auto run_qr = [&]() {
        int k = 0;
        if (!eff_la) {
            while (k < N) {
                int sb = sb_clipped(k);
                phase_q1(s_comp, cublas, d_panel_recv, d_panel_full, d_Q_full, d_cs, d_panel_bcast, k, sb);
                if (k + sb < N) phase_q2(s_comp, cublas, d_panel_bcast, k, sb);
                k += sb;
            }
        } else {
            int sb = sb_clipped(k);
            phase_q1(s_comp, cublas, d_panel_recv, d_panel_full, d_Q_full, d_cs, d_panel_bcast, k, sb);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? sb_clipped(next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    phase_q2(s_comp, cublas, d_panel_bcast, k, sb);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, d_panel_recv2, d_panel_full2, d_Q_full2, d_cs2, d_panel_bcast2, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                    CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                    std::swap(d_panel_bcast, d_panel_bcast2);
                    std::swap(d_panel_recv,  d_panel_recv2);
                    std::swap(d_panel_full,  d_panel_full2);
                    std::swap(d_Q_full,      d_Q_full2);
                    std::swap(d_cs,          d_cs2);
                    k = next_k; sb = next_sb;
                } else {
                    if (n_rest > 0) phase_q2(s_comp, cublas, d_panel_bcast, k, sb);
                    break;
                }
            }
        }
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, (size_t)m_loc * n_loc * sizeof(float), cudaMemcpyDeviceToDevice));
    };
    for (int i = 0; i < 2; ++i) { reset_A(); run_qr(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp)); CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    {
        float* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)n_loc * n_loc * sizeof(float)));
        CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  n_loc, m_loc, &one_f, d_A, m_loc, &zero_f, d_GG, n_loc));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        FULL25D_NCCL_CHECK(ncclAllReduce(d_GG, d_GG, (size_t)n_loc * n_loc, ncclFloat, ncclSum, S.nccl_col, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        double max_local = 0.0;
        if (S.col_rank == 0) {
            for (int j = 0; j < n_loc; ++j) {
                float dj;
                CUDA_CHECK(cudaMemcpy(&dj, d_GG + j + (size_t)j * n_loc, sizeof(float), cudaMemcpyDeviceToHost));
                double dev = std::fabs((double)dj - 1.0);
                if (dev > max_local) max_local = dev;
            }
        }
        double max_global = 0.0;
        MPI_Allreduce(&max_local, &max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (_rank == 0) {
            printf("  Validation(fp32) variant=gg_fp32full_p25d N=%d grid=[%d,%d,%d]  max|diag(Q'Q)-1| = %.2e\n",
                   N, Px, Py, Pz, max_global);
            fflush(stdout);
        }
        cudaFree(d_GG);
    }

    const int nrun = 5;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset_A();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qr();
        CUDA_CHECK(cudaStreamSynchronize(s_comp)); CUDA_CHECK(cudaStreamSynchronize(s_comm));
        if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  gg_fp32full_p25d%s  N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               eff_la ? "+LA" : "", N, b, Px, Py, Pz, times[0], tmed);
        printf("METRICS bench=givens_full25d matrix=fp32full layout=bc_2p5d N=%d b=%d Px=%d Py=%d Pz=%d passes=1 ours_ms=%.4f\n",
               N, b, Px, Py, Pz, tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_A_orig);
    cudaFree(d_panel_recv); cudaFree(d_panel_full); cudaFree(d_Q_full); cudaFree(d_cs);
    cudaFree(d_panel_bcast); cudaFree(d_W);
    if (eff_la) {
        cudaFree(d_panel_recv2); cudaFree(d_panel_full2); cudaFree(d_Q_full2);
        cudaFree(d_cs2); cudaFree(d_panel_bcast2);
        cublasDestroy(cublas_la);
    }
    cublasDestroy(cublas);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm);
    if (eff_la) cudaStreamDestroy(s_la);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done); cudaEventDestroy(e_next_ready);
    return 0;
}

#endif  // GIVENS_FULL25D_INL
