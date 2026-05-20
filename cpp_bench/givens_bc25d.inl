// givens_bc25d.inl — Path-g true small-tile BC 2.5D runner.
//
// Phase Q1 uses butterfly Tournament-Givens (Kwasniewski SC'21,
// qr_schur_xpartition.tex §Phase Q3_g).  See the algorithm derivation
// in the earlier version of this file; this version adds:
//
//   * Item-4 (always on): ping-pong (g_cur, g_nxt) host-side pointer
//     swap in the G_self extract loop eliminates a per-stage memcpy.
//   * Item-1 + Item-2 (gated on eff_la):
//       - Phase Q2 is split column-wise into "Q2_next" (panel-(k+1)'s
//         local columns) and "Q2_rest".
//       - Panel-(k+1) Q1 runs on s_la, using a duplicate set of
//         butterfly scratch and the dedicated `S.nccl_col_la`
//         communicator on s_comm_la so its Send/Recv don't serialize
//         on the same NCCL queue as Q2's AllReduce.
//
// givens_panel_kernel, form_Q_from_givens_kernel, init_thinQ_identity
// are defined in givens_2p5d_bench.cu and reused directly.

#ifndef GIVENS_BC25D_INL
#define GIVENS_BC25D_INL

#include "bc25d_helpers.cuh"
#include "full25d_grid.hpp"
#include "full25d_kernels.cuh"
#include "bench_vendor_metrics.hpp"
#include "nextla_mp_trail.hpp"
#include "tsqr_butterfly.cuh"

__global__ static void giv_bc25d_cast_d2f(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ static void giv_bc25d_cast_f2d(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}

struct GivBcPanelCtx {
    cudaStream_t s = nullptr;
    cublasHandle_t cublas = nullptr;
    double* cs0 = nullptr;          // 2*b*(locr-1) rotation list (stage 0)
    double* R0 = nullptr;
    double* R_partner = nullptr;
    double* stacked = nullptr;
    double* cs_stage = nullptr;     // log2_pr × 2*b*(2b-1) per stage
    double* Q_stage = nullptr;      // 2b × b scratch for explicit Q^(s)
    double* Q_local = nullptr;      // locr × b explicit local-stage Q
    double* G_self = nullptr;
    double* G_tmp = nullptr;
    double* panel_bcast = nullptr;
};

static int run_givens_bc25d_fp64(int N, int b, bool eff_la,
                                   bool use_mp_trail, bool use_tf32_trail,
                                   const Full25DGrid& G, const Full25DSubcomms& S) {
    int Px = G.Px, Py = G.Py, Pz = G.Pz;
    int my_py = G.my_py;
    int col_size = S.col_size;
    int col_rank = S.col_rank;
    int log2_pr  = tsqr_butterfly_log2_ceil(col_size);
    if ((1 << log2_pr) != col_size) {
        if (_rank == 0)
            fprintf(stderr, "givens_bc25d: butterfly requires col_size (=%d) to be a power of 2\n",
                    col_size);
        return 82;
    }
    std::int64_t locr = 0, locc = 0;
    bc_local_dims(N, b, Px, Py, G.my_px, G.my_py, &locr, &locc);
    if (locr <= 0 || locc <= 0) {
        if (_rank == 0) fprintf(stderr, "givens_bc25d: empty local buffer (locr=%lld, locc=%lld)\n",
                                 (long long)locr, (long long)locc);
        return 81;
    }

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

    GivBcPanelCtx pri{}, la{};
    pri.s = s_comp;
    CUBLAS_CHECK(cublasCreate(&pri.cublas)); CUBLAS_CHECK(cublasSetStream(pri.cublas, pri.s));
    if (use_tf32_trail) CUBLAS_CHECK(cublasSetMathMode(pri.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    if (eff_la) {
        la.s = s_la;
        CUBLAS_CHECK(cublasCreate(&la.cublas)); CUBLAS_CHECK(cublasSetStream(la.cublas, la.s));
        if (use_tf32_trail) CUBLAS_CHECK(cublasSetMathMode(la.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    double *d_A = nullptr, *d_A_orig = nullptr;
    size_t locsz = (size_t)locr * (size_t)locc;
    CUDA_CHECK(cudaMalloc(&d_A,      locsz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A_orig, locsz * sizeof(double)));
    {
        std::vector<double> host(locsz);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), locsz * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A, locsz * sizeof(double), cudaMemcpyDeviceToDevice));

    int log2_max = std::max(log2_pr, 1);
    auto alloc_ctx_scratch = [&](GivBcPanelCtx& c) {
        CUDA_CHECK(cudaMalloc(&c.cs0,         (size_t)2 * b * (size_t)(locr - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.R0,          (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.R_partner,   (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.stacked,     (size_t)2 * b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.cs_stage,    (size_t)log2_max * 2 * b * (size_t)(2 * b - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.Q_stage,     (size_t)2 * b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.Q_local,     (size_t)locr * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.G_self,      (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.G_tmp,       (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.panel_bcast, (size_t)locr * b * sizeof(double)));
    };
    alloc_ctx_scratch(pri);
    if (eff_la) alloc_ctx_scratch(la);

    double *d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * locc * sizeof(double)));

    float *d_panel_bcast_f = nullptr, *d_A_trail_f = nullptr, *d_W_f = nullptr;
    if (use_mp_trail) {
        CUDA_CHECK(cudaMalloc(&d_panel_bcast_f, (size_t)locr * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_trail_f,     (size_t)locr * locc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_f,           (size_t)b * locc * sizeof(float)));
    }

    const double one_d = 1.0, zero_d = 0.0, neg_one_d = -1.0;
    const float  one_f = 1.f,  zero_f = 0.f,  neg_one_f = -1.f;

    auto phase_q1_compute = [&](GivBcPanelCtx& ctx, int k, int sb,
                                  ncclComm_t nccl_col_for_butterfly,
                                  cudaStream_t s_comm_for_butterfly) {
        int py_panel = (k / b) % Py;
        bool i_own_panel = (my_py == py_panel);
        if (!i_own_panel) return;
        std::int64_t panel_lcol = bc_panel_lcol(k, b, Py);
        double* d_panel_loc = d_A + panel_lcol * locr;

        // Stage 0: tournament-Givens on locr × sb.
        givens_panel_kernel<<<1, 256, 0, ctx.s>>>(d_panel_loc, (int)locr, sb,
                                                     (int)locr, ctx.cs0);
        tsqr_butterfly_copy_upper_R(d_panel_loc, (int)locr, ctx.R0, sb, ctx.s);

        for (int stage = 0; stage < log2_pr; ++stage) {
            int partner = tsqr_butterfly_partner(col_rank, stage);
            CUDA_CHECK(cudaEventRecord(e_comp_done, ctx.s));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm_for_butterfly, e_comp_done, 0));
            FULL25D_NCCL_CHECK(tsqr_butterfly_exchange(ctx.R0, ctx.R_partner, sb, partner,
                                                         nccl_col_for_butterfly, s_comm_for_butterfly));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm_for_butterfly));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.s, e_ar_done, 0));

            tsqr_butterfly_stack(ctx.R0, ctx.R_partner, ctx.stacked, sb,
                                  col_rank, partner, ctx.s);
            double* cs_this_stage = ctx.cs_stage +
                (size_t)stage * 2 * b * (size_t)(2 * b - 1);
            givens_panel_kernel<<<1, 256, 0, ctx.s>>>(ctx.stacked, 2 * b, sb,
                                                         2 * b, cs_this_stage);
            tsqr_butterfly_copy_upper_R(ctx.stacked, 2 * b, ctx.R0, sb, ctx.s);
        }

        // G_self extraction with ping-pong (item 4).
        double* g_cur = ctx.G_self;
        double* g_nxt = ctx.G_tmp;
        tsqr_butterfly_eye(g_cur, sb, sb, ctx.s);
        for (int stage = log2_pr - 1; stage >= 0; --stage) {
            init_thinQ_identity<<<((size_t)2 * b * sb + 255) / 256, 256, 0, ctx.s>>>(
                ctx.Q_stage, 2 * b, sb);
            double* cs_this_stage = ctx.cs_stage +
                (size_t)stage * 2 * b * (size_t)(2 * b - 1);
            form_Q_from_givens_kernel<<<1, 256, 0, ctx.s>>>(ctx.Q_stage, 2 * b, sb,
                                                             2 * b, cs_this_stage);
            int half = tsqr_butterfly_half(col_rank, stage);
            CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      sb, sb, sb,
                                      &one_d,
                                      ctx.Q_stage + (size_t)half * b, 2 * b,
                                      g_cur, b,
                                      &zero_d, g_nxt, b));
            std::swap(g_cur, g_nxt);
        }

        init_thinQ_identity<<<((size_t)locr * sb + 255) / 256, 256, 0, ctx.s>>>(
            ctx.Q_local, (int)locr, sb);
        form_Q_from_givens_kernel<<<1, 256, 0, ctx.s>>>(ctx.Q_local, (int)locr, sb,
                                                          (int)locr, ctx.cs0);
        CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  (int)locr, sb, sb,
                                  &one_d, ctx.Q_local, (int)locr,
                                          g_cur,       b,
                                  &zero_d, ctx.panel_bcast, (int)locr));
        CUDA_CHECK(cudaMemcpyAsync(d_panel_loc, ctx.panel_bcast,
                                    (size_t)locr * sb * sizeof(double),
                                    cudaMemcpyDeviceToDevice, ctx.s));
    };

    auto phase_q1_broadcast = [&](GivBcPanelCtx& ctx, int k, int sb) {
        int py_panel = (k / b) % Py;
        CUDA_CHECK(cudaEventRecord(e_comp_done, ctx.s));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclBroadcast(ctx.panel_bcast, ctx.panel_bcast,
                                          (size_t)locr * sb, ncclDouble,
                                          py_panel, S.nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.s, e_ar_done, 0));
    };

    auto phase_q2_part = [&](GivBcPanelCtx& ctx, int sb,
                              double* panel_bcast,
                              std::int64_t trail_lcol, std::int64_t ncols_part,
                              std::int64_t d_W_col_off) {
        if (ncols_part <= 0) return;
        double* d_trail = d_A + trail_lcol * locr;
        double* d_W_part = d_W + d_W_col_off * b;
        if (!use_mp_trail) {
            CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, (int)ncols_part, (int)locr,
                                      &one_d, panel_bcast, locr,
                                              d_trail, locr,
                                      &zero_d, d_W_part, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, ctx.s));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(d_W_part, d_W_part, (size_t)sb * ncols_part, ncclDouble,
                                              ncclSum, S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.s, e_ar_done, 0));
            CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)locr, (int)ncols_part, sb,
                                      &neg_one_d, panel_bcast, locr,
                                                  d_W_part, b,
                                      &one_d, d_trail, locr));
        } else {
            size_t nq = (size_t)locr * sb;
            size_t nt = (size_t)locr * ncols_part;
            float* d_panel_f = d_panel_bcast_f;
            float* d_trail_f_off = d_A_trail_f + (size_t)trail_lcol * locr;
            float* d_W_part_f = d_W_f + (size_t)d_W_col_off * b;
            int nt_th = 256;
            giv_bc25d_cast_d2f<<<(unsigned)((nq + nt_th - 1) / nt_th), nt_th, 0, ctx.s>>>(
                panel_bcast, d_panel_f, nq);
            giv_bc25d_cast_d2f<<<(unsigned)((nt + nt_th - 1) / nt_th), nt_th, 0, ctx.s>>>(
                d_trail, d_trail_f_off, nt);
            CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, (int)ncols_part, (int)locr,
                                      &one_f, d_panel_f, (int)locr,
                                              d_trail_f_off, (int)locr,
                                      &zero_f, d_W_part_f, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, ctx.s));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            FULL25D_NCCL_CHECK(ncclAllReduce(d_W_part_f, d_W_part_f, (size_t)sb * ncols_part, ncclFloat,
                                              ncclSum, S.nccl_col, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.s, e_ar_done, 0));
            CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)locr, (int)ncols_part, sb,
                                      &neg_one_f, d_panel_f, (int)locr,
                                                  d_W_part_f, b,
                                      &one_f, d_trail_f_off, (int)locr));
            giv_bc25d_cast_f2d<<<(unsigned)((nt + nt_th - 1) / nt_th), nt_th, 0, ctx.s>>>(
                d_trail_f_off, d_trail, nt);
        }
    };

    auto run_qr_serial = [&]() {
        for (int k = 0; k < N; k += b) {
            int sb = std::min(b, N - k);
            phase_q1_compute(pri, k, sb, S.nccl_col, s_comm);
            phase_q1_broadcast(pri, k, sb);
            if (k + sb < N) {
                std::int64_t trail_lcol = bc_first_trail_lcol(k + sb, b, Py, my_py, locc);
                std::int64_t ncols = locc - trail_lcol;
                phase_q2_part(pri, sb, pri.panel_bcast, trail_lcol, ncols, 0);
            }
        }
    };

    auto run_qr_la = [&]() {
        int sb_first = std::min(b, N);
        phase_q1_compute(pri, 0, sb_first, S.nccl_col, s_comm);
        phase_q1_broadcast(pri, 0, sb_first);

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
                phase_q2_part(pri, sb, pri.panel_bcast, trail_lcol, sb_next_local, 0);
                CUDA_CHECK(cudaEventRecord(e_q2_next_done, pri.s));
            }

            CUDA_CHECK(cudaStreamWaitEvent(la.s, e_q2_next_done, 0));
            phase_q1_compute(la, next_k, sb_next, S.nccl_col_la, s_comm_la);
            CUDA_CHECK(cudaEventRecord(e_la_q1_done, la.s));

            if (ncols_rest > 0) {
                phase_q2_part(pri, sb, pri.panel_bcast,
                               trail_lcol + sb_next_local, ncols_rest, sb_next_local);
            }

            CUDA_CHECK(cudaStreamWaitEvent(pri.s, e_la_q1_done, 0));
            std::swap(pri, la);
            std::swap(pri.s, la.s);
            std::swap(pri.cublas, la.cublas);

            phase_q1_broadcast(pri, next_k, sb_next);
        }
    };

    auto run_qr = [&]() { if (eff_la) run_qr_la(); else run_qr_serial(); };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, locsz * sizeof(double), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) { reset_A(); run_qr(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (eff_la) { CUDA_CHECK(cudaStreamSynchronize(s_la));
                  CUDA_CHECK(cudaStreamSynchronize(s_comm_la)); }
    MPI_Barrier(MPI_COMM_WORLD);

    {
        double* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)locc * locc * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(pri.cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  (int)locc, (int)locr, &one_d, d_A, (int)locr,
                                  &zero_d, d_GG, (int)locc));
        CUDA_CHECK(cudaStreamSynchronize(pri.s));
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
            printf("  Validation     variant=givens_bc25d_butterfly%s N=%d b=%d grid=[%d,%d,%d] log2(P_r)=%d  max|diag(Q'Q)-1| = %.2e\n",
                   eff_la ? "+LA" : "", N, b, Px, Py, Pz, log2_pr, max_global);
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
        printf("  givens_bc25d_butterfly%s  matrix=%s N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               eff_la ? "+LA" : "", matnm, N, b, Px, Py, Pz, times[0], tmed);
        printf("METRICS bench=givens_bc25d matrix=%s layout=bc_2p5d panel=butterfly_givens%s N=%d b=%d Px=%d Py=%d Pz=%d passes=1 ours_ms=%.4f\n",
               matnm, eff_la ? "+la" : "", N, b, Px, Py, Pz, tmed);
        fflush(stdout);
    }

    auto free_ctx = [&](GivBcPanelCtx& c) {
        cudaFree(c.cs0); cudaFree(c.R0); cudaFree(c.R_partner); cudaFree(c.stacked);
        cudaFree(c.cs_stage); cudaFree(c.Q_stage); cudaFree(c.Q_local);
        cudaFree(c.G_self); cudaFree(c.G_tmp); cudaFree(c.panel_bcast);
        if (c.cublas) cublasDestroy(c.cublas);
    };
    cudaFree(d_A); cudaFree(d_A_orig); cudaFree(d_W);
    free_ctx(pri);
    if (eff_la) free_ctx(la);
    if (use_mp_trail) { cudaFree(d_panel_bcast_f); cudaFree(d_A_trail_f); cudaFree(d_W_f); }
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm);
    if (eff_la) { cudaStreamDestroy(s_la); cudaStreamDestroy(s_comm_la); }
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_q2_next_done); cudaEventDestroy(e_la_q1_done);
    return 0;
}

#endif  // GIVENS_BC25D_INL
