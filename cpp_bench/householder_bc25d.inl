// householder_bc25d.inl — Path-h true small-tile BC 2.5D runner.
//
// Phase Q1 uses butterfly Tournament-TSQR (Demmel et al. 2012,
// Kwasniewski SC'21 §A.3 / qr_schur_xpartition.tex §Phase Q3_h).  See
// the earlier version of this file for the algorithm derivation; this
// version adds two scheduling optimizations:
//
//   * Item-4 (always on): ping-pong (g_cur, g_nxt) host-side pointer
//     swap in the G_self extract loop eliminates a per-stage memcpy.
//   * Item-1 + Item-2 (gated on eff_la):
//       - Phase Q2 is split column-wise into "Q2_next" (the first
//         sb_next cols of the local trail, which on owner ranks are
//         panel-(k+1)'s columns) and "Q2_rest" (everything after).
//       - Panel-(k+1) Q1 compute runs on a dedicated s_la stream while
//         Q2_rest(k) runs on s_comp.  The data dependency
//         "panel-(k+1)'s cols must be updated by Q2_next" is enforced
//         by a CUDA event handshake.
//       - LA owners (my_py == py_panel_next) duplicate all panel
//         scratch (V_stage_la, tau_stage_la, R0_la, …) so the two
//         streams don't contend for buffers.
//       - When eff_la is false the runner falls back to a single-stream
//         schedule that's bit-for-bit equivalent to the pre-LA path.

#ifndef HOUSEHOLDER_BC25D_INL
#define HOUSEHOLDER_BC25D_INL

#include "bc25d_helpers.cuh"
#include "full25d_grid.hpp"
#include "full25d_kernels.cuh"
#include "bench_vendor_metrics.hpp"
#include "nextla_mp_trail.hpp"
#include "tsqr_butterfly.cuh"

__global__ static void hh_bc25d_cast_d2f(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ static void hh_bc25d_cast_f2d(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}

// Group of per-panel scratch + handles bound to one compute stream.  In
// the LA path we maintain two of these (one for s_comp, one for s_la)
// so panel-k Q2_rest and panel-(k+1) Q1 compute run concurrently.
struct HhBcPanelCtx {
    cudaStream_t s = nullptr;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    double* tau0 = nullptr;
    double* R0 = nullptr;
    double* R_partner = nullptr;
    double* stacked = nullptr;
    double* V_stage = nullptr;
    double* tau_stage = nullptr;
    double* Q_stage_full = nullptr;
    double* G_self = nullptr;
    double* G_tmp = nullptr;
    double* panel_work = nullptr;
    double* panel_bcast = nullptr;
    int*    info = nullptr;
};

static int run_householder_bc25d_fp64(int N, int b, bool eff_la,
                                        bool use_mp_trail, bool use_tf32_trail,
                                        const Full25DGrid& G, const Full25DSubcomms& S) {
    int Px = G.Px, Py = G.Py, Pz = G.Pz;
    int my_py = G.my_py;
    int col_size = S.col_size;
    int col_rank = S.col_rank;
    int log2_pr  = tsqr_butterfly_log2_ceil(col_size);
    if ((1 << log2_pr) != col_size) {
        if (_rank == 0)
            fprintf(stderr, "householder_bc25d: butterfly requires col_size (=%d) to be a power of 2\n",
                    col_size);
        return 82;
    }
    std::int64_t locr = 0, locc = 0;
    bc_local_dims(N, b, Px, Py, G.my_px, G.my_py, &locr, &locc);
    if (locr <= 0 || locc <= 0) {
        if (_rank == 0) fprintf(stderr, "householder_bc25d: empty local buffer (locr=%lld, locc=%lld)\n",
                                 (long long)locr, (long long)locc);
        return 81;
    }

    cudaStream_t s_comp, s_comm, s_la = nullptr, s_comm_la = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    if (eff_la) {
        CUDA_CHECK(cudaStreamCreate(&s_la));
        // Separate comm stream for nccl_col_la so LA butterfly does not
        // serialize on s_comm against the main panel's Q2 AllReduce.
        CUDA_CHECK(cudaStreamCreate(&s_comm_la));
    }
    cudaEvent_t e_comp_done, e_ar_done, e_q2_next_done, e_la_q1_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done,    cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,      cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_q2_next_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_la_q1_done,   cudaEventDisableTiming));

    HhBcPanelCtx pri{}, la{};
    pri.s = s_comp;
    CUBLAS_CHECK(cublasCreate(&pri.cublas));    CUBLAS_CHECK(cublasSetStream(pri.cublas, pri.s));
    CUSOLVER_CHECK(cusolverDnCreate(&pri.cusolver)); CUSOLVER_CHECK(cusolverDnSetStream(pri.cusolver, pri.s));
    if (use_tf32_trail) CUBLAS_CHECK(cublasSetMathMode(pri.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    if (eff_la) {
        la.s = s_la;
        CUBLAS_CHECK(cublasCreate(&la.cublas));     CUBLAS_CHECK(cublasSetStream(la.cublas, la.s));
        CUSOLVER_CHECK(cusolverDnCreate(&la.cusolver)); CUSOLVER_CHECK(cusolverDnSetStream(la.cusolver, la.s));
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
    auto alloc_ctx_scratch = [&](HhBcPanelCtx& c) {
        CUDA_CHECK(cudaMalloc(&c.tau0,         (size_t)b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.R0,           (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.R_partner,    (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.stacked,      (size_t)2 * b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.V_stage,      (size_t)log2_max * 2 * b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.tau_stage,    (size_t)log2_max * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.Q_stage_full, (size_t)2 * b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.G_self,       (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.G_tmp,        (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.panel_bcast,  (size_t)locr * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.info,         sizeof(int)));
    };
    alloc_ctx_scratch(pri);
    if (eff_la) alloc_ctx_scratch(la);

    int lwork_geqrf_panel = 0, lwork_geqrf_stack = 0;
    int lwork_orgqr_panel = 0, lwork_orgqr_stack = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(pri.cusolver, (int)locr, b, d_A, (int)locr, &lwork_geqrf_panel));
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(pri.cusolver, 2 * b, b, pri.stacked, 2 * b, &lwork_geqrf_stack));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(pri.cusolver, (int)locr, b, b, d_A, (int)locr, pri.tau0, &lwork_orgqr_panel));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(pri.cusolver, 2 * b, b, b, pri.stacked, 2 * b, pri.tau0, &lwork_orgqr_stack));
    int lwork_panel = std::max({lwork_geqrf_panel, lwork_geqrf_stack,
                                 lwork_orgqr_panel, lwork_orgqr_stack});
    CUDA_CHECK(cudaMalloc(&pri.panel_work, (size_t)lwork_panel * sizeof(double)));
    if (eff_la) CUDA_CHECK(cudaMalloc(&la.panel_work, (size_t)lwork_panel * sizeof(double)));

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

    // --- phase_q1: butterfly TSQR on `ctx.s` using `ctx.*` scratch ---
    // Writes panel Q into d_A's panel slot (if owner) AND into ctx.panel_bcast.
    // Does NOT do the row_comm broadcast; the caller does that on s_comm.
    //
    // nccl_col_for_butterfly selects which col_comm to use for the butterfly
    // R-exchange.  Pass S.nccl_col for the primary stream, S.nccl_col_la for
    // the LA stream so the two streams' col_comm collectives don't serialize.
    auto phase_q1_compute = [&](HhBcPanelCtx& ctx, int k, int sb,
                                  ncclComm_t nccl_col_for_butterfly,
                                  cudaStream_t s_comm_for_butterfly) {
        int py_panel = (k / b) % Py;
        bool i_own_panel = (my_py == py_panel);
        if (!i_own_panel) return;
        std::int64_t panel_lcol = bc_panel_lcol(k, b, Py);
        double* d_panel_loc = d_A + panel_lcol * locr;

        CUSOLVER_CHECK(cusolverDnDgeqrf(ctx.cusolver, (int)locr, sb, d_panel_loc, (int)locr,
                                         ctx.tau0, ctx.panel_work, lwork_panel, ctx.info));
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
            CUSOLVER_CHECK(cusolverDnDgeqrf(ctx.cusolver, 2 * b, sb, ctx.stacked, 2 * b,
                                             ctx.tau_stage + (size_t)stage * b,
                                             ctx.panel_work, lwork_panel, ctx.info));
            tsqr_butterfly_copy_upper_R(ctx.stacked, 2 * b, ctx.R0, sb, ctx.s);
            CUDA_CHECK(cudaMemcpyAsync(ctx.V_stage + (size_t)stage * 2 * b * b,
                                        ctx.stacked,
                                        (size_t)2 * b * sb * sizeof(double),
                                        cudaMemcpyDeviceToDevice, ctx.s));
        }

        // G_self extraction with ping-pong (item 4).
        double* g_cur = ctx.G_self;
        double* g_nxt = ctx.G_tmp;
        tsqr_butterfly_eye(g_cur, sb, sb, ctx.s);
        for (int stage = log2_pr - 1; stage >= 0; --stage) {
            CUDA_CHECK(cudaMemcpyAsync(ctx.Q_stage_full,
                                        ctx.V_stage + (size_t)stage * 2 * b * b,
                                        (size_t)2 * b * sb * sizeof(double),
                                        cudaMemcpyDeviceToDevice, ctx.s));
            CUSOLVER_CHECK(cusolverDnDorgqr(ctx.cusolver, 2 * b, sb, sb,
                                             ctx.Q_stage_full, 2 * b,
                                             ctx.tau_stage + (size_t)stage * b,
                                             ctx.panel_work, lwork_panel, ctx.info));
            int half = tsqr_butterfly_half(col_rank, stage);
            CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      sb, sb, sb, &one_d,
                                      ctx.Q_stage_full + (size_t)half * b, 2 * b,
                                      g_cur, b, &zero_d, g_nxt, b));
            std::swap(g_cur, g_nxt);
        }

        CUSOLVER_CHECK(cusolverDnDorgqr(ctx.cusolver, (int)locr, sb, sb,
                                         d_panel_loc, (int)locr, ctx.tau0,
                                         ctx.panel_work, lwork_panel, ctx.info));
        CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  (int)locr, sb, sb,
                                  &one_d, d_panel_loc, (int)locr,
                                          g_cur,       b,
                                  &zero_d, ctx.panel_bcast, (int)locr));
        CUDA_CHECK(cudaMemcpyAsync(d_panel_loc, ctx.panel_bcast,
                                    (size_t)locr * sb * sizeof(double),
                                    cudaMemcpyDeviceToDevice, ctx.s));
    };

    auto phase_q1_broadcast = [&](HhBcPanelCtx& ctx, int k, int sb) {
        int py_panel = (k / b) % Py;
        CUDA_CHECK(cudaEventRecord(e_comp_done, ctx.s));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        FULL25D_NCCL_CHECK(ncclBroadcast(ctx.panel_bcast, ctx.panel_bcast,
                                          (size_t)locr * sb, ncclDouble,
                                          py_panel, S.nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.s, e_ar_done, 0));
    };

    // phase_q2_part calls below use s_comm for AllReduce(W) since it's bound
    // to the main panel pri.s = s_comp; the LA stream s_la only runs Q1
    // compute, never Q2.

    // phase_q2_part: trailing update over [trail_lcol, trail_lcol + ncols_part).
    // Q is taken from `panel_bcast` (which was filled by phase_q1_compute).
    // d_W_off is the column offset within d_W to write the partial W to.
    auto phase_q2_part = [&](HhBcPanelCtx& ctx, int sb,
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
            hh_bc25d_cast_d2f<<<(unsigned)((nq + nt_th - 1) / nt_th), nt_th, 0, ctx.s>>>(
                panel_bcast, d_panel_f, nq);
            hh_bc25d_cast_d2f<<<(unsigned)((nt + nt_th - 1) / nt_th), nt_th, 0, ctx.s>>>(
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
            hh_bc25d_cast_f2d<<<(unsigned)((nt + nt_th - 1) / nt_th), nt_th, 0, ctx.s>>>(
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

    // LA loop: split Q2 column-wise; pipeline panel-(k+1) Q1 on s_la with
    // panel-k Q2_rest on s_comp.  Buffer swap each iteration so the active
    // panel's scratch is always at `pri`.
    auto run_qr_la = [&]() {
        // Bootstrap: panel-0 Q1 on s_comp using main col_comm.
        int sb_first = std::min(b, N);
        phase_q1_compute(pri, 0, sb_first, S.nccl_col, s_comm);
        phase_q1_broadcast(pri, 0, sb_first);

        for (int k = 0; k < N; k += b) {
            int sb = std::min(b, N - k);
            int next_k = k + sb;
            if (next_k >= N) break;  // last panel; nothing to do.
            int sb_next = std::min(b, N - next_k);
            int py_panel_next = (next_k / b) % Py;
            bool i_own_next = (my_py == py_panel_next);

            std::int64_t trail_lcol = bc_first_trail_lcol(next_k, b, Py, my_py, locc);
            std::int64_t ncols = locc - trail_lcol;
            // sb_next_local: number of local cols that correspond to panel-(k+1)'s
            // global col range [next_k, next_k+sb_next).  For LA owners these are
            // the first sb_next cols of the local trail starting at trail_lcol.
            // For non-owners they're zero, and the split degenerates to monolithic.
            std::int64_t sb_next_local = i_own_next ? sb_next : 0;
            std::int64_t ncols_rest    = ncols - sb_next_local;

            // Q2_next: update panel-(k+1)'s local cols on s_comp (owners only).
            if (sb_next_local > 0) {
                phase_q2_part(pri, sb, pri.panel_bcast,
                               trail_lcol, sb_next_local, 0);
                CUDA_CHECK(cudaEventRecord(e_q2_next_done, pri.s));
            }

            // Launch panel-(k+1) Q1 on s_la using the la scratch + the
            // dedicated nccl_col_la communicator on s_comm_la, so the LA
            // butterfly Send/Recv runs concurrently with pri's Q2 AllReduce.
            CUDA_CHECK(cudaStreamWaitEvent(la.s, e_q2_next_done, 0));
            phase_q1_compute(la, next_k, sb_next, S.nccl_col_la, s_comm_la);
            CUDA_CHECK(cudaEventRecord(e_la_q1_done, la.s));

            // Q2_rest: trailing update over the remaining local cols, on s_comp.
            if (ncols_rest > 0) {
                std::int64_t trail_lcol_rest = trail_lcol + sb_next_local;
                phase_q2_part(pri, sb, pri.panel_bcast,
                               trail_lcol_rest, ncols_rest, sb_next_local);
            }

            // Wait for panel-(k+1) Q1 compute on s_la; then broadcast its
            // result on s_comm.  Swap pri <-> la so the next iteration's
            // "current panel" scratch is the just-completed one.
            CUDA_CHECK(cudaStreamWaitEvent(pri.s, e_la_q1_done, 0));
            std::swap(pri, la);
            // pri now points to the just-finished panel's scratch + handles;
            // la now points to the buffer-set that's free for the next LA.
            // But pri.s is still s_la under the hood — swap the stream/handle
            // back so panel-(k+1)'s broadcast still uses s_comm conventions.
            // (Simpler: keep pri.s = s_comp always; just swap the scratch
            // pointers, not s/cublas/cusolver.)
            // Undo the s/handle swap to keep the invariant:
            std::swap(pri.s, la.s);
            std::swap(pri.cublas, la.cublas);
            std::swap(pri.cusolver, la.cusolver);
            std::swap(pri.panel_work, la.panel_work);  // workspace is per-stream
            // Now pri = {s_comp, cublas_pri, cusolver_pri, panel_work_pri, but la-scratch (R0/V_stage/etc.)};
            // la  = {s_la,  cublas_la,  cusolver_la,  panel_work_la,  but pri-scratch}.
            // The actual data buffers (tau0, R0, R_partner, stacked, V_stage, …,
            // panel_bcast) follow the just-finished panel; this is what we want
            // so that the next broadcast / Q2 reads from the right panel_bcast.

            phase_q1_broadcast(pri, next_k, sb_next);
        }
    };

    auto run_qr = [&]() {
        if (eff_la) run_qr_la();
        else        run_qr_serial();
    };

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
            printf("  Validation     variant=hh_bc25d_tsqr%s N=%d b=%d grid=[%d,%d,%d] log2(P_r)=%d  max|diag(Q'Q)-1| = %.2e\n",
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
        printf("  hh_bc25d_tsqr%s  matrix=%s N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               eff_la ? "+LA" : "", matnm, N, b, Px, Py, Pz, times[0], tmed);
        printf("METRICS bench=householder_bc25d matrix=%s layout=bc_2p5d panel=tsqr_butterfly%s N=%d b=%d Px=%d Py=%d Pz=%d passes=1 ours_ms=%.4f\n",
               matnm, eff_la ? "+la" : "", N, b, Px, Py, Pz, tmed);
        fflush(stdout);
    }

    auto free_ctx = [&](HhBcPanelCtx& c) {
        cudaFree(c.tau0); cudaFree(c.R0); cudaFree(c.R_partner); cudaFree(c.stacked);
        cudaFree(c.V_stage); cudaFree(c.tau_stage); cudaFree(c.Q_stage_full);
        cudaFree(c.G_self); cudaFree(c.G_tmp); cudaFree(c.panel_bcast);
        cudaFree(c.panel_work); cudaFree(c.info);
        if (c.cublas)   cublasDestroy(c.cublas);
        if (c.cusolver) cusolverDnDestroy(c.cusolver);
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

#endif  // HOUSEHOLDER_BC25D_INL
