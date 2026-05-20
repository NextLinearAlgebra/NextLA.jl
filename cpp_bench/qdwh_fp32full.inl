// Path (q) fp32full: float storage for A, X, stacked S, inner CQR2, and QDWH Halley recurrence.

#ifndef QDWH_FP32FULL_INL
#define QDWH_FP32FULL_INL

#include "bench_vendor_metrics.hpp"

__global__ void fill_stacked_kernel_f(float* __restrict__ S, const float* __restrict__ Xk, int m_local, int N,
                                      int rank, float scale_X) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)2 * m_local * N;
    if (idx >= total) return;
    int i = (int)(idx % (2 * m_local));
    int j = (int)(idx / (2 * m_local));
    if (i < m_local) {
        S[idx] = scale_X * Xk[i + (long long)j * m_local];
    } else {
        int row_in_I = rank * m_local + (i - m_local);
        S[idx] = (row_in_I == j) ? 1.f : 0.f;
    }
}

__global__ void update_X_kernel_f(float* __restrict__ Xnew, const float* __restrict__ Xk, const float* __restrict__ P,
                                  int m_local, int N, float alpha_x, float alpha_p) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_local * N;
    if (idx >= total) return;
    Xnew[idx] = alpha_x * Xk[idx] + alpha_p * P[idx];
}

static int run_qdwh_fp32full_main(const Args& A, int c) {
    int N = A.N, b = A.b, m_local = N / c;
    int m_st_local = 2 * m_local;
    int ngpu;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    ncclUniqueId nccl_id;
    if (_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, c, nccl_id, _rank));

    cudaStream_t s_comp, s_comm;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    cudaEvent_t e_comp_done, e_ar_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done, cudaEventDisableTiming));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cusolverDnHandle_t cusolver;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s_comp));

    float *d_A = nullptr, *d_X = nullptr, *d_Xnew = nullptr, *d_S = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)m_local * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)m_local * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Xnew, (size_t)m_local * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)m_st_local * N * sizeof(float)));

    float *d_G = nullptr, *d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)b * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * N * sizeof(float)));
    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    float* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)potrf_lwork * sizeof(float)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    float* d_Q2_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q2_recv, (size_t)m_local * N * (size_t)c * sizeof(float)));
    float* d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, (size_t)m_local * N * sizeof(float)));
    float* d_Q2_pack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q2_pack, (size_t)m_local * N * sizeof(float)));

    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;

    auto phase_q1_cqr2_f = [&](int k, int sb) {
        for (int it = 0; it < 2; ++it) {
            float* d_S_panel = d_S + (size_t)k * m_st_local;
            CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, sb, m_st_local, &one_f, d_S_panel,
                                     m_st_local, &zero_f, d_G, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_G, d_G, (size_t)b * b, ncclFloat, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
            CUSOLVER_CHECK(
                cusolverDnSpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G, b, d_potrf_work, potrf_lwork, d_info));
            CUBLAS_CHECK(cublasStrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                     m_st_local, sb, &one_f, d_G, b, d_S_panel, m_st_local));
        }
    };

    auto phase_q2_cqr2_f = [&](int k, int sb, int col_start, int ncols) {
        float* d_S_panel = d_S + (size_t)k * m_st_local;
        float* d_S_tr = d_S + (size_t)col_start * m_st_local;
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, sb, ncols, m_st_local, &one_f, d_S_panel, m_st_local,
                                 d_S_tr, m_st_local, &zero_f, d_W, b));
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_comm, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, m_st_local, ncols, sb, &neg_one_f, d_S_panel,
                                 m_st_local, d_W, b, &one_f, d_S_tr, m_st_local));
    };

    auto inner_qr_f = [&]() {
        int k = 0;
        while (k < N) {
            int sb = std::min(b, N - k);
            int n_tr = N - (k + sb);
            phase_q1_cqr2_f(k, sb);
            if (n_tr > 0) phase_q2_cqr2_f(k, sb, k + sb, n_tr);
            k += sb;
        }
    };

    auto run_qdwh_f = [&]() -> float {
        float frob_local = 0.f, frob_global = 0.f;
        size_t na = (size_t)m_local * N;
        CUBLAS_CHECK(cublasSnrm2(cublas, (int)na, d_A, 1, &frob_local));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        frob_local *= frob_local;
        MPI_Allreduce(&frob_local, &frob_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        float alpha = std::sqrt(std::max(frob_global, 1e-30f));
        float inv_alpha = 1.f / alpha;
        {
            int threads = 256;
            long long total = (long long)m_local * N;
            long long blocks = (total + threads - 1) / threads;
            update_X_kernel_f<<<(unsigned)blocks, threads, 0, s_comp>>>(d_X, d_A, d_A, m_local, N, inv_alpha, 0.f);
        }
        float l = 1.f / std::sqrt((float)N);
        if (l < 1e-15f) l = 1e-15f;

        for (int kit = 0; kit < A.iters; ++kit) {
            float l2 = l * l;
            float dd = std::pow(4.f * (1.f - l2) / (l2 * l2), 1.f / 3.f);
            float sd = std::sqrt(1.f + dd);
            float inner = std::max(0.f, 8.f - 4.f * dd + 8.f * (2.f - l2) / (l2 * sd));
            float a_k = sd + 0.5f * std::sqrt(inner);
            float b_k = (a_k - 1.f) * (a_k - 1.f) / 4.f;
            float c_k = a_k + b_k - 1.f;
            float scale_X = std::sqrt(c_k);

            {
                int threads = 256;
                long long total = (long long)m_st_local * N;
                long long blocks = (total + threads - 1) / threads;
                fill_stacked_kernel_f<<<(unsigned)blocks, threads, 0, s_comp>>>(d_S, d_X, m_local, N, _rank, scale_X);
            }
            inner_qr_f();

            for (int j = 0; j < N; ++j) {
                float* dst = d_Q2_pack + (size_t)j * m_local;
                float* src = d_S + (size_t)m_local + (size_t)j * m_st_local;
                CUDA_CHECK(cudaMemcpy(dst, src, (size_t)m_local * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllGather(d_Q2_pack, d_Q2_recv, (size_t)m_local * N, ncclFloat, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));

            for (int r = 0; r < c; ++r) {
                float* Q_2_r = d_Q2_recv + (size_t)r * m_local * N;
                float* P_cols = d_P + (size_t)r * m_local * m_local;
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, m_local, m_local, N, &one_f, d_S, m_st_local,
                                        Q_2_r, m_local, &zero_f, P_cols, m_local));
            }

            float alpha_x = b_k / c_k;
            float alpha_p = (a_k - b_k / c_k) / std::sqrt(c_k);
            {
                int threads = 256;
                long long total = (long long)m_local * N;
                long long blocks = (total + threads - 1) / threads;
                update_X_kernel_f<<<(unsigned)blocks, threads, 0, s_comp>>>(d_Xnew, d_X, d_P, m_local, N, alpha_x,
                                                                               alpha_p);
            }
            std::swap(d_X, d_Xnew);

            float num = l * (a_k + b_k * l * l);
            float den = 1.f + c_k * l * l;
            l = num / den;
        }
        return l;
    };

    auto reset_A = [&]() {
        std::vector<float> host(m_local * (size_t)N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<float> nrm(0.f, 1.f);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    };

    reset_A();
    for (int i = 0; i < 2; ++i) {
        reset_A();
        run_qdwh_f();
    }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    MPI_Barrier(MPI_COMM_WORLD);

    const int nrun = 3;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset_A();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qdwh_f();
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  %-30s  N=%d b=%d c=%d  tmin=%9.2f ms  tmed=%9.2f ms\n", "qdwh_fp32full", N, b, c, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, c);
        printf("METRICS bench=qdwh_2p5d matrix=fp32full N=%d b=%d c=%d passes=1 ", N, b, c);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_Xnew);
    cudaFree(d_S);
    cudaFree(d_G);
    cudaFree(d_W);
    cudaFree(d_potrf_work);
    cudaFree(d_info);
    cudaFree(d_Q2_recv);
    cudaFree(d_P);
    cudaFree(d_Q2_pack);
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaEventDestroy(e_comp_done);
    cudaEventDestroy(e_ar_done);
    cudaStreamDestroy(s_comp);
    cudaStreamDestroy(s_comm);
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();
    return 0;
}

#endif
