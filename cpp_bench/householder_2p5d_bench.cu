// Householder + WY 2.5D multi-GPU QR benchmark — Path h of qr_schur_xpartition.tex.
//
//   Variants exposed via CLI flags (same set as scqr3_2p5d_variants.cu):
//     --mp               trailing-update GEMMs in FP32  (Phase Q6 low-prec)
//     --ir=K             K rounds of post-factorization Cholesky-QR refinement
//                        (Phase Q6 of the tex)
//     --la / --lookahead 2-stream pipelining of Phase Q1 (panel) and Phase Q2
//                        (trailing update)  -- Phase Q5 of the tex.
//
//   Schedule per panel k:
//     Q1  AllGather A[:, k:k+b] across all ranks  → full m×b panel on every rank
//         cusolverDnDgeqrf on the replicated panel (parallel-but-redundant)
//         cusolverDnDorgqr to materialize the thin Q (m×b orthonormal columns)
//         memcpy2D Q's local rows back into A_local[:, k:k+b]
//     Q2  W_local = Q_local^T · A_trail_local           (local Dgemm)
//         AllReduce(W)                                  (NCCL)
//         A_trail_local -= Q_local · W                  (local Dgemm)
//
//   This implements the gather-and-redundant-factor flavor of Path h; the
//   TSQR/CAQR tournament-reduce flavor is more communication-efficient but is
//   not implemented here. The gather-form is mathematically equivalent and
//   keeps the implementation parallel to the Path s (sCQR3) variants.
//
//   Processor grid: 1D row partition  m_local = N / c  rows per rank.

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

// Rearrange NCCL AllGather output (rank-block layout, c · m_local · sb doubles)
// into a single column-major m_total × sb panel.
//   recv[r * m_local * sb + j * m_local + i_local]  →  full[(r * m_local + i_local) + j * m_total]
__global__ void rearrange_recv_to_panel(const double* __restrict__ recv,
                                         double* __restrict__ full,
                                         int m_local, int sb, int P, int m_total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_total * sb;
    if (idx >= total) return;
    int i = (int)(idx % m_total);     // global row
    int j = (int)(idx / m_total);     // column
    int r = i / m_local;              // rank
    int i_local = i - r * m_local;
    full[idx] = recv[(long long)r * m_local * sb + (long long)j * m_local + i_local];
}

struct Args {
    int N = 16000, b = 0, n_ir = 0;
    bool mp = false, lookahead = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--N=", 0) == 0) a.N = std::atoi(s.c_str() + 4);
        else if (s.rfind("--b=", 0) == 0) a.b = std::atoi(s.c_str() + 4);
        else if (s.rfind("--ir=", 0) == 0) a.n_ir = std::atoi(s.c_str() + 5);
        else if (s == "--mp") a.mp = true;
        else if (s == "--lookahead" || s == "--la") a.lookahead = true;
        else if (s.size() > 0 && std::isdigit((unsigned char)s[0])) {
            if (a.N == 16000 && i == 1) a.N = std::atoi(s.c_str());
            else if (a.b == 0 && i == 2) a.b = std::atoi(s.c_str());
        }
    }
    if (a.b == 0) {
        if      (a.N <=  4000) a.b = 363;
        else if (a.N <=  8000) a.b = 512;
        else if (a.N <= 16000) a.b = 512;
        else if (a.N <= 32000) a.b = 512;
        else if (a.N <= 48000) a.b = 1024;
        else if (a.N <= 64000) a.b = 1024;
        else if (a.N <= 96000) a.b = 1536;
        else                   a.b = 2048;
    }
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

    char tag[160];
    std::snprintf(tag, sizeof(tag), "householder%s%s%s%s",
                  A.mp?"+MP":"", A.lookahead?"+LA":"",
                  A.n_ir>0?"+IR":"", A.n_ir>0?(std::to_string(A.n_ir)).c_str():"");

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" Householder 2.5D  N=%d b=%d c=%d   variant=%s\n", N, b, c, tag);
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
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done,    cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_panel_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_next_ready, cudaEventDisableTiming));

    cublasHandle_t cublas;       CUBLAS_CHECK(cublasCreate(&cublas));       CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cusolverDnHandle_t cusolver; CUSOLVER_CHECK(cusolverDnCreate(&cusolver)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s_comp));

    // Distributed A (m_local rows × N cols, column-major).
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

    // Replicated panel buffers (full m × b column-major).
    double* d_panel_recv = nullptr;   // c · m_local · b doubles (NCCL AllGather output)
    double* d_panel_full = nullptr;   // m × b column-major (after rearrangement; also holds Q after orgqr)
    double* d_tau        = nullptr;
    int*    d_info       = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel_recv, (size_t)m_local * b * (size_t)c * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel_full, (size_t)N * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau,        (size_t)b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info,       sizeof(int)));

    // Trailing-update workspace.
    double* d_W     = nullptr;
    double* d_G_ref = nullptr;     // for IR pass (Cholesky-QR refinement)
    double* d_potrf_work_ref = nullptr;
    int     potrf_lwork_ref = 0;
    CUDA_CHECK(cudaMalloc(&d_W,     (size_t)b * N * sizeof(double)));
    if (A.n_ir > 0) {
        CUDA_CHECK(cudaMalloc(&d_G_ref, (size_t)b * b * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G_ref, b, &potrf_lwork_ref));
        CUDA_CHECK(cudaMalloc(&d_potrf_work_ref, potrf_lwork_ref * sizeof(double)));
    }

    // Workspace sizes (query for the largest panel we'll use).
    int lwork_geqrf = 0, lwork_orgqr = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolver, N, b, d_panel_full, N, &lwork_geqrf));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolver, N, b, b, d_panel_full, N, d_tau, &lwork_orgqr));
    int lwork_panel = std::max(lwork_geqrf, lwork_orgqr);
    double* d_panel_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel_work, lwork_panel * sizeof(double)));

    // MP scratch (FP32 trailing).
    float *d_A_panel_f=nullptr, *d_A_tr_f=nullptr, *d_W_f=nullptr;
    if (A.mp) {
        CUDA_CHECK(cudaMalloc(&d_A_panel_f, (size_t)m_local * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_tr_f,    (size_t)m_local * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_f,       (size_t)b * N * sizeof(float)));
    }

    // Look-ahead second-panel scratch.
    double *d_panel_recv2=nullptr, *d_panel_full2=nullptr, *d_tau2=nullptr, *d_panel_work2=nullptr;
    int    *d_info2 = nullptr;
    cublasHandle_t cublas_la;
    cusolverDnHandle_t cusolver_la;
    if (A.lookahead) {
        CUDA_CHECK(cudaMalloc(&d_panel_recv2, (size_t)m_local * b * (size_t)c * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_full2, (size_t)N * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_tau2,        (size_t)b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_work2, lwork_panel * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info2,       sizeof(int)));
        CUBLAS_CHECK(cublasCreate(&cublas_la));       CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    const double one_d=1.0, zero_d=0.0, neg_one_d=-1.0;
    const float  one_f=1.0f, zero_f=0.0f, neg_one_f=-1.0f;

    // Phase Q1 (Householder panel): AllGather → rearrange → geqrf → orgqr → memcpy2D Q back.
    //   - Uses stream s_use (and s_comm for the AllGather, sync via event)
    //   - Operates on the supplied scratch buffers (so look-ahead can use a 2nd set)
    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t /*cb*/, cusolverDnHandle_t cs,
                        double* p_recv, double* p_full, double* tau, double* work, int* info,
                        int k, int sb) {
        // 1) AllGather A_local[:, k:k+sb] (m_local × sb column-major, contiguous in d_A_local)
        //    on s_comm. Synchronize back to s_use afterward.
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllGather(d_A_local + (size_t)k * m_local,
                                 p_recv,
                                 (size_t)m_local * sb, ncclDouble,
                                 nccl_comm, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));

        // 2) Rearrange rank-block → column-major m × sb panel.
        long long total = (long long)N * sb;
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        rearrange_recv_to_panel<<<(unsigned)blocks, threads, 0, s_use>>>(p_recv, p_full, m_local, sb, c, N);

        // 3) cuSolver dgeqrf on m × sb panel (replicated on every rank).
        CUSOLVER_CHECK(cusolverDnDgeqrf(cs, N, sb, p_full, N, tau, work, lwork_panel, info));

        // 4) cuSolver dorgqr → Q (m × sb orthonormal) in place.
        CUSOLVER_CHECK(cusolverDnDorgqr(cs, N, sb, sb, p_full, N, tau, work, lwork_panel, info));

        // 5) memcpy2D rank's Q rows back into A_local[:, k:k+sb].
        //    src: p_full + rank * m_local  (column-major, ld = N)
        //    dst: d_A_local + k * m_local  (column-major, ld = m_local)
        CUDA_CHECK(cudaMemcpy2DAsync(d_A_local + (size_t)k * m_local, (size_t)m_local * sizeof(double),
                                     p_full + (size_t)_rank * m_local, (size_t)N * sizeof(double),
                                     (size_t)m_local * sizeof(double), (size_t)sb,
                                     cudaMemcpyDeviceToDevice, s_use));
    };

    // Phase Q2 trailing update — identical to the sCQR3 variants bench.
    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb,
                        int k, int sb, int col_start, int ncols, bool mp) {
        double* d_A_panel = d_A_local + (size_t)k * m_local;
        double* d_A_tr    = d_A_local + (size_t)col_start * m_local;

        if (!mp) {
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
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_local, ncols, sb,
                                      &neg_one_d, d_A_panel, m_local,
                                                  d_W, b,
                                      &one_d, d_A_tr, m_local));
        } else {
            size_t np = (size_t)m_local * sb;
            size_t nt = (size_t)m_local * ncols;
            cast_d2f<<<(np + 255)/256, 256, 0, s_use>>>(d_A_panel, d_A_panel_f, np);
            cast_d2f<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_tr,    d_A_tr_f,    nt);
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
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_local, ncols, sb,
                                      &neg_one_f, d_A_panel_f, m_local,
                                                  d_W_f, b,
                                      &one_f, d_A_tr_f, m_local));
            cast_f2d<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_tr_f, d_A_tr, nt);
        }
    };

    // Phase Q6 (Iterative Refinement) — same as scqr3 variants: re-orthogonalize
    // each diagonal-block panel of Q via one Cholesky-QR pass.
    auto refine_one = [&]() {
        int k = 0;
        while (k < N) {
            int sb = std::min(b, N - k);
            double* d_A_panel = d_A_local + (size_t)k * m_local;
            CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                      sb, m_local, &one_d, d_A_panel, m_local,
                                      &zero_d, d_G_ref, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_G_ref, d_G_ref, (size_t)b * b, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
            CUSOLVER_CHECK(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G_ref, b, d_potrf_work_ref, potrf_lwork_ref, d_info));
            CUBLAS_CHECK(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      m_local, sb, &one_d, d_G_ref, b, d_A_panel, m_local));
            k += sb;
        }
    };

    auto run_qr = [&]() {
        int k = 0;
        if (!A.lookahead) {
            // Vanilla schedule: Phase Q1 then Phase Q2 per panel.
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                phase_q1(s_comp, cublas, cusolver,
                         d_panel_recv, d_panel_full, d_tau, d_panel_work, d_info,
                         k, sb);
                if (n_tr > 0) phase_q2(s_comp, cublas, k, sb, k + sb, n_tr, A.mp);
                k += sb;
            }
        } else {
            // Look-ahead (Phase Q5): pipeline panel(k+1) with trailing-update(k).
            int sb = std::min(b, N - k);
            phase_q1(s_comp, cublas, cusolver,
                     d_panel_recv, d_panel_full, d_tau, d_panel_work, d_info,
                     k, sb);
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
                    // Q1(k+1) on s_la, waiting for e_next_ready.
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, cusolver_la,
                             d_panel_recv2, d_panel_full2, d_tau2, d_panel_work2, d_info2,
                             next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                }
                if (n_rest > 0) {
                    // Q2 on A_rest on s_comp, concurrent with s_la's Q1.
                    phase_q2(s_comp, cublas, k, sb, rest_start, n_rest, A.mp);
                }
                // s_comp must wait for s_la to finish Q1(k+1) before next step.
                CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                k = next_k;
                sb = next_sb;
                if (sb == 0) break;
            }
        }
        // Iterative refinement (Phase Q6)
        for (int j = 0; j < A.n_ir; ++j) refine_one();
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A_local, d_A_orig, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));
    };

    // Warmup
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

    cudaFree(d_A_orig); cudaFree(d_A_local);
    cudaFree(d_panel_recv); cudaFree(d_panel_full); cudaFree(d_tau); cudaFree(d_info);
    cudaFree(d_panel_work); cudaFree(d_W);
    if (A.n_ir > 0) { cudaFree(d_G_ref); cudaFree(d_potrf_work_ref); }
    if (A.mp) { cudaFree(d_A_panel_f); cudaFree(d_A_tr_f); cudaFree(d_W_f); }
    if (A.lookahead) {
        cudaFree(d_panel_recv2); cudaFree(d_panel_full2); cudaFree(d_tau2);
        cudaFree(d_panel_work2); cudaFree(d_info2);
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
