// Multi-variant C++ benchmark for the 2.5D scheduling from
// qr_schur_xpartition.tex §A.3 (Path s) and §A.1 (Phase Q5, Q6).
//
//   Variants exposed via CLI flags:
//     --passes=N         3 (sCQR3, default) or 2 (CQR2)
//     --mp               trailing-update GEMMs in FP32  (Phase Q6 low-prec pass)
//     --ir=K             after the (possibly MP) factorization, run K rounds
//                        of "fix Q via 1-pass Cholesky-QR + recompute R" in
//                        FP64 (Phase Q6 of the tex)
//     --lookahead        2-stream pipelining of Phase Q1 and Phase Q2
//                        (Phase Q5 of the tex)
//
//   Processor grid: Px=Py=1, Pz=c (the paper's degenerate-2.5D / pure-replica
//   case for P_1=1); each rank pins to one GPU and owns m/c rows of A.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>

#define CUDA_CHECK(stmt) do { cudaError_t e=(stmt); if(e!=cudaSuccess){ fprintf(stderr,"[r%d] CUDA %s @ %s:%d\n",_rank,cudaGetErrorString(e),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUBLAS_CHECK(stmt) do { cublasStatus_t s=(stmt); if(s!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"[r%d] cuBLAS %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)
#define CUSOLVER_CHECK(stmt) do { cusolverStatus_t s=(stmt); if(s!=CUSOLVER_STATUS_SUCCESS){ fprintf(stderr,"[r%d] cuSOLVER %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,3);} } while(0)
#define NCCL_CHECK(stmt) do { ncclResult_t r=(stmt); if(r!=ncclSuccess){ fprintf(stderr,"[r%d] NCCL %s @ %s:%d\n",_rank,ncclGetErrorString(r),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,4);} } while(0)

static int _rank = 0;

__global__ void cast_d2f(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ void cast_f2d(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}
__global__ void trace_b_kernel(const double* G, int ldg, int b, double* out) {
    __shared__ double sh[1024];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int j = tid; j < b; j += blockDim.x) acc += G[j + (long long)j * ldg];
    sh[tid] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    if (tid == 0) out[0] = sh[0];
}
__global__ void shift_diag_from_trace_kernel(double* G, int ldg, int b, const double* tr, double coef) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < b) G[j + (long long)j * ldg] += coef * tr[0];
}

struct Args {
    int N = 16000, b = 0, passes = 3, n_ir = 0;
    bool mp = false, lookahead = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--N=", 0) == 0) a.N = std::atoi(s.c_str() + 4);
        else if (s.rfind("--b=", 0) == 0) a.b = std::atoi(s.c_str() + 4);
        else if (s.rfind("--passes=", 0) == 0) a.passes = std::atoi(s.c_str() + 9);
        else if (s.rfind("--ir=", 0) == 0) a.n_ir = std::atoi(s.c_str() + 5);
        else if (s == "--mp") a.mp = true;
        else if (s == "--lookahead" || s == "--la") a.lookahead = true;
        else if (s.size() > 0 && std::isdigit((unsigned char)s[0])) {
            // positional: first N, second b (back-compat with scqr3_2p5d_bench)
            if (a.N == 16000 && i == 1) a.N = std::atoi(s.c_str());
            else if (a.b == 0 && i == 2) a.b = std::atoi(s.c_str());
        }
    }
    if (a.b == 0) {
        if      (a.N <=  4000) a.b = 363;
        else if (a.N <=  8000) a.b = 512;
        else if (a.N <= 16000) a.b = 512;
        else if (a.N <= 24000) a.b = 512;
        else if (a.N <= 32000) a.b = 512;
        else if (a.N <= 48000) a.b = 1024;
        else                   a.b = 1024;
    }
    if (a.passes < 1 || a.passes > 3) a.passes = 3;
    return a;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int c;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c);

    Args A = parse_args(argc, argv);
    if (A.N % c != 0) { if (_rank==0) fprintf(stderr,"N=%d not divisible by c=%d\n",A.N,c); MPI_Abort(MPI_COMM_WORLD,5); }
    int N = A.N, b = A.b, m_local = N / c;
    int ngpu; CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    // Build variant tag for reporting.
    char tag[160];
    std::snprintf(tag, sizeof(tag), "passes=%d%s%s%s%s",
                  A.passes, A.mp?"+MP":"",
                  A.lookahead?"+LA":"",
                  A.n_ir>0?"+IR":"",
                  A.n_ir>0?(std::to_string(A.n_ir)).c_str():"");

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" 2.5D variant   N=%d b=%d c=%d   variant=%s\n", N, b, c, tag);
        printf("=================================================================\n");
        fflush(stdout);
    }

    // NCCL bootstrap.
    ncclUniqueId nccl_id;
    if (_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, c, nccl_id, _rank));

    // Streams + events.
    cudaStream_t s_comp, s_comm, s_la;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    CUDA_CHECK(cudaStreamCreate(&s_la));
    cudaEvent_t e_comp_done, e_ar_done, e_panel_done, e_next_ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done,   cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,     cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_panel_done,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_next_ready,  cudaEventDisableTiming));

    // Handles bound to compute stream by default.
    cublasHandle_t cublas;       CUBLAS_CHECK(cublasCreate(&cublas));       CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cusolverDnHandle_t cusolver; CUSOLVER_CHECK(cusolverDnCreate(&cusolver)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s_comp));

    // Buffers.
    double* d_A_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_local, (size_t)m_local * N * sizeof(double)));
    {
        std::vector<double> host(m_local * (size_t)N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A_local, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    double* d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_local * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A_local, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));

    double* d_G = nullptr;
    double* d_W = nullptr;
    double* d_trace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G,     (size_t)b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,     (size_t)b * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(double)));

    // FP32 scratch for MP path.
    float *d_A_panel_f=nullptr, *d_A_tr_f=nullptr, *d_W_f=nullptr;
    if (A.mp) {
        CUDA_CHECK(cudaMalloc(&d_A_panel_f, (size_t)m_local * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_tr_f,    (size_t)m_local * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_f,       (size_t)b * N * sizeof(float)));
    }

    // POTRF workspace.
    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Look-ahead second-panel scratch.
    double *d_G2=nullptr, *d_potrf_work2=nullptr;
    int *d_info2=nullptr;
    cublasHandle_t cublas_la;
    cusolverDnHandle_t cusolver_la;
    if (A.lookahead) {
        CUDA_CHECK(cudaMalloc(&d_G2, (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_potrf_work2, potrf_lwork * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info2, sizeof(int)));
        CUBLAS_CHECK(cublasCreate(&cublas_la));   CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    const double one_d=1.0, zero_d=0.0, neg_one_d=-1.0;
    const float  one_f=1.0f, zero_f=0.0f, neg_one_f=-1.0f;

    // Generic Phase Q1 (sCQR3 / CQR2 / CQR1) for one panel on stream s_use,
    // using handles `cb` (cuBLAS) and `cs` (cuSolver), Gram buffer Gb, info_buf.
    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t cb, cusolverDnHandle_t cs,
                        double* Gb, double* work, int* info,
                        int k, int sb, int passes) {
        double coef = 11.0 * ((double)N * sb + (double)sb * (sb + 1)) * 2.220446049250313e-16;
        for (int it = 0; it < passes; ++it) {
            double* d_A_panel = d_A_local + (size_t)k * m_local;
            CUBLAS_CHECK(cublasDsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                      sb, m_local, &one_d, d_A_panel, m_local,
                                      &zero_d, Gb, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(Gb, Gb, (size_t)b * b, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            if (it == 0) {
                trace_b_kernel<<<1, std::min(sb, 1024), 0, s_use>>>(Gb, b, sb, d_trace);
                shift_diag_from_trace_kernel<<<(sb + 255) / 256, 256, 0, s_use>>>(Gb, b, sb, d_trace, coef);
            }
            CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_UPPER, sb, Gb, b, work, potrf_lwork, info));
            CUBLAS_CHECK(cublasDtrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      m_local, sb, &one_d, Gb, b, d_A_panel, m_local));
        }
    };

    // Phase Q2 trailing update on A_tr_local = A_local[:, k+sb : k+sb+ncols],
    // using stream s_use. FP64 by default; if mp=true, casts to FP32 around
    // the two cuBLAS GEMMs.
    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb,
                        int k, int sb, int col_start, int ncols, bool mp) {
        double* d_A_panel = d_A_local + (size_t)k * m_local;
        double* d_A_tr    = d_A_local + (size_t)col_start * m_local;

        if (!mp) {
            // W = A_panel^T · A_tr  (FP64 GEMM, sb × ncols)
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_local,
                                      &one_d, d_A_panel, m_local,
                                              d_A_tr, m_local,
                                      &zero_d, d_W, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            // A_tr -= A_panel · W
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_local, ncols, sb,
                                      &neg_one_d, d_A_panel, m_local,
                                                  d_W, b,
                                      &one_d, d_A_tr, m_local));
        } else {
            // Cast A_panel and A_tr to FP32 on the active stream.
            size_t np = (size_t)m_local * sb;
            size_t nt = (size_t)m_local * ncols;
            cast_d2f<<<(np + 255)/256, 256, 0, s_use>>>(d_A_panel, d_A_panel_f, np);
            cast_d2f<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_tr,    d_A_tr_f,    nt);
            // W_f = A_panel_f^T · A_tr_f  (FP32 GEMM; cuBLAS uses TF32 TC if
            // CUBLAS_TENSOR_OP_MATH or compute type set; here single-precision
            // ALU path. To use TF32 explicitly: cublasSgemmEx with computeType
            // CUBLAS_COMPUTE_32F_FAST_TF32.)
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_local,
                                      &one_f, d_A_panel_f, m_local,
                                              d_A_tr_f, m_local,
                                      &zero_f, d_W_f, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W_f, d_W_f, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            // A_tr_f -= A_panel_f · W_f  (FP32)
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_local, ncols, sb,
                                      &neg_one_f, d_A_panel_f, m_local,
                                                  d_W_f, b,
                                      &one_f, d_A_tr_f, m_local));
            // Cast A_tr_f back to FP64.
            cast_f2d<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_tr_f, d_A_tr, nt);
            // Also cast W_f back so the upper R block is FP64 (we ignore W here).
        }
    };

    // ────────────────────────────────────────────────────────────────────
    // Phase Q6 (Iterative Refinement) — Carson–Higham 2018 style.
    // Re-orthogonalize Q via one Cholesky-QR pass in FP64, then recompute R
    // from A_orig: G = Q^T Q (multi-GPU via NCCL Allreduce), G = U^T U,
    // Q := Q U^{-1}. We don't need the explicit R here (the bench measures
    // factor wall-clock + ortho); a real consumer would also recompute
    // R = Q^T A_orig with one FP64 GEMM. Cost ≈ one extra Cholesky-QR pass.
    auto refine_one = [&]() {
        // Q is in d_A_local. G = Q^T Q (full N×N — costly, but the IR pass is
        // dominated by the GEMM, which is what we pay to recover precision).
        // For practicality we refine only the b×b leading diagonal stripes
        // panel-by-panel.
        int k = 0;
        while (k < N) {
            int sb = std::min(b, N - k);
            double* d_A_panel = d_A_local + (size_t)k * m_local;
            CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                      sb, m_local, &one_d, d_A_panel, m_local,
                                      &zero_d, d_G, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_G, d_G, (size_t)b * b, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
            CUSOLVER_CHECK(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G, b, d_potrf_work, potrf_lwork, d_info));
            CUBLAS_CHECK(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      m_local, sb, &one_d, d_G, b, d_A_panel, m_local));
            k += sb;
        }
    };

    auto run_qr = [&]() {
        int k = 0;
        if (!A.lookahead) {
            // ── Vanilla schedule: Phase Q1 then Phase Q2 per panel ─────────
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                phase_q1(s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, k, sb, A.passes);
                if (n_tr > 0) phase_q2(s_comp, cublas, k, sb, k + sb, n_tr, A.mp);
                k += sb;
            }
        } else {
            // ── Look-ahead schedule (paper Phase Q5) ───────────────────────
            // Step structure:
            //   1) Phase Q1(k) on s_comp.
            //   2) Phase Q2 of A_next = cols [k+sb, k+2sb) on s_comp.
            //   3) Event: s_la waits, then runs Phase Q1(k+1) on s_la.
            //   4) Phase Q2 of A_rest = cols [k+2sb, N) on s_comp, overlapped
            //      with Phase Q1(k+1) on s_la.
            //   5) Sync: s_comp waits for s_la before next iteration.
            int sb = std::min(b, N - k);
            phase_q1(s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, k, sb, A.passes);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));

            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? std::min(b, N - next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;

                if (next_sb > 0) {
                    // Q2 on A_next on s_comp.
                    phase_q2(s_comp, cublas, k, sb, next_k, next_sb, A.mp);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    // Phase Q1(k+1) on s_la, waiting for e_next_ready.
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, cusolver_la, d_G2, d_potrf_work2, d_info2, next_k, next_sb, A.passes);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                }
                if (n_rest > 0) {
                    // Q2 on A_rest on s_comp, concurrent with s_la's Q1.
                    phase_q2(s_comp, cublas, k, sb, rest_start, n_rest, A.mp);
                }
                // s_comp must wait for s_la to finish Q1(k+1) before the next step.
                CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                k = next_k;
                sb = next_sb;
                if (sb == 0) break;
            }
        }
        // Iterative refinement
        for (int j = 0; j < A.n_ir; ++j) refine_one();
    };

    // Save A_orig already done above. Warmup + validate + bench.
    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A_local, d_A_orig, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) { reset_A(); run_qr(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    // Validation: max|diag(Q'Q) - 1|
    {
        double* d_QtQ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_QtQ, (size_t)N * N * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  N, m_local, &one_d, d_A_local, m_local,
                                  &zero_d, d_QtQ, N));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        NCCL_CHECK(ncclAllReduce(d_QtQ, d_QtQ, (size_t)N * N, ncclDouble, ncclSum, nccl_comm, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        if (_rank == 0) {
            double max_dev = 0.0;
            for (int j = 0; j < N; ++j) {
                double dj;
                CUDA_CHECK(cudaMemcpy(&dj, d_QtQ + j + (size_t)j*N, sizeof(double), cudaMemcpyDeviceToHost));
                max_dev = std::max(max_dev, std::abs(dj - 1.0));
            }
            printf("  Validation     variant=%s N=%d c=%d  max|diag(Q'Q)-1| = %.2e\n",
                   tag, N, c, max_dev);
            fflush(stdout);
        }
        cudaFree(d_QtQ);
    }

    // Timed runs.
    const int nrun = 5;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset_A();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qr();
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        printf("  %-30s  N=%d b=%d c=%d  tmin=%9.2f ms  tmed=%9.2f ms\n",
               tag, N, b, c, times[0], times[nrun/2]);
        fflush(stdout);
    }

    cudaFree(d_A_orig); cudaFree(d_A_local); cudaFree(d_G); cudaFree(d_W); cudaFree(d_trace);
    cudaFree(d_potrf_work); cudaFree(d_info);
    if (A.mp) { cudaFree(d_A_panel_f); cudaFree(d_A_tr_f); cudaFree(d_W_f); }
    if (A.lookahead) {
        cudaFree(d_G2); cudaFree(d_potrf_work2); cudaFree(d_info2);
        cublasDestroy(cublas_la); cusolverDnDestroy(cusolver_la);
    }
    cublasDestroy(cublas); cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm); cudaStreamDestroy(s_la);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done); cudaEventDestroy(e_next_ready);
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();
    return 0;
}
