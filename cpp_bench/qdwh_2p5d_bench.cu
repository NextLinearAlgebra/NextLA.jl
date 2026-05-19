// QDWH-QR 2.5D benchmark — Path q of qr_schur_xpartition.tex.
//
//   QR via the QR-based QDWH polar decomposition (Nakatsukasa-Higham 2013):
//     X_0 = A / alpha    where  alpha >= ||A||_2 (Frobenius bound used)
//     for k = 0, 1, 2, ...:
//        weights (a_k, b_k, c_k) from the Halley recurrence on l_k
//        [sqrt(c_k) X_k; I_n] = [Q_1; Q_2] R     (2n x n QR, 2.5D inner)
//        X_{k+1} = (b_k/c_k) X_k + (1/sqrt(c_k)) (a_k - b_k/c_k) Q_1 Q_2^T
//        l_{k+1} = l_k (a_k + b_k l_k^2) / (1 + c_k l_k^2)
//        if |l_{k+1} - 1| < tol: break
//   After convergence X_inf = U (orthogonal polar factor of A).
//   R = U^T A (upper triangular).
//
//   Inner QR (stacked 2n×n CQR2 / Path s style): each panel uses local SYRK
//     into G, NCCL AllReduce(G) over the 1D communicator, replicated POTRF
//     and TRSM — same Phase Q3 pattern as scqr3_2p5d_variants / Path (s).
//     Trailing update: W = Q^T A_trail, AllReduce(W), A_trail -= Q W.
//     Later Halley steps AllGather Q_2 row blocks for the polar recurrence.
//
//   The 2n x n stacked input is row-distributed across c ranks with an
//   interleaved layout: rank r holds X_k's m_local = n/c local rows then
//   I_n's m_local rows starting at global row r*m_local. The QR is invariant
//   under row permutation of the input so this interleaving is harmless.
//
//   Path (q) vs Path (s) slab parity: same --matrix= fp64|fp64mp|fp64mp_tf32|fp32full and --la/--no-la
//   vocabulary as scqr3_full25d_bench.cu (passes=2|3 remain s-only).
//   Inner-QR lookahead for fp32full is not implemented: default-on LA is auto-off unless argv sets explicit --la (abort).
//   Flags:
//     --matrix=fp64|fp64mp|fp64mp_tf32|fp32full   (fp32full: float A/X/S; inner-QR --la not implemented — see main())
//     --mp            alias for fp64mp when matrix was fp64
//     --la / --lookahead and --no-la / --no-lookahead  (default LA on for fp64 family, parity with Path (s))
//     --iters=K       fixed Halley iteration count (default 6)
//     --layout=blockcyclic --px= --py= --pz=1  BC layout; see qdwh_block_cyclic.inl (same four matrix modes as slab).
//
//   Processor grid: 1D row partition, m_local = N / c, with the stacked
//   m_stacked_local = 2 * m_local.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <cstdint>

#include "derived_schedule.hpp"
#include "matrix_mode.hpp"
#include "full25d_grid.hpp"
#include "nextla_mp_trail.hpp"
#include "nextla_fast_memory.hpp"
#include "bench_vendor_metrics.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>

#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
#define NEXTLA_HAVE_CUBLAS_TF32 1
#else
#define NEXTLA_HAVE_CUBLAS_TF32 0
#endif

#define CUDA_CHECK(stmt) do { cudaError_t e=(stmt); if(e!=cudaSuccess){ fprintf(stderr,"[r%d] CUDA %s @ %s:%d\n",_rank,cudaGetErrorString(e),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUBLAS_CHECK(stmt) do { cublasStatus_t s=(stmt); if(s!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"[r%d] cuBLAS %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)
#define CUSOLVER_CHECK(stmt) do { cusolverStatus_t s=(stmt); if(s!=CUSOLVER_STATUS_SUCCESS){ fprintf(stderr,"[r%d] cuSOLVER %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,3);} } while(0)
#define NCCL_CHECK(stmt) do { ncclResult_t r=(stmt); if(r!=ncclSuccess){ fprintf(stderr,"[r%d] NCCL %s @ %s:%d\n",_rank,ncclGetErrorString(r),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,4);} } while(0)

static int _rank = 0;

__global__ void cast_d2f_q(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ void cast_f2d_q(const float* f, double* d, size_t n) {
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

// Build the stacked S (2n × n) interleaved per rank:
//   rank r's local m_stacked_local = 2*m_local rows are
//     [sqrt(c_k) X_k rows of rank r ; identity rows global indices r*m_local .. (r+1)*m_local - 1]
// d_S has shape (2*m_local) x N column-major, leading dim = 2*m_local.
__global__ void fill_stacked_kernel(double* __restrict__ S,
                                     const double* __restrict__ Xk,
                                     int m_local, int N, int rank,
                                     double scale_X) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)2 * m_local * N;
    if (idx >= total) return;
    int i = (int)(idx % (2 * m_local));
    int j = (int)(idx / (2 * m_local));
    if (i < m_local) {
        // Top half: scale_X * X_k_local[i, j]
        S[idx] = scale_X * Xk[i + (long long)j * m_local];
    } else {
        // Bottom half: identity row global index = rank * m_local + (i - m_local)
        int row_in_I = rank * m_local + (i - m_local);
        S[idx] = (row_in_I == j) ? 1.0 : 0.0;
    }
}

// Update step: X_{k+1} = alpha_x * X_k + alpha_p * P
//   X_k_local: m_local × N column-major
//   P_local:   m_local × N column-major (output of Q_1 Q_2^T extracted per rank)
__global__ void update_X_kernel(double* __restrict__ Xnew,
                                 const double* __restrict__ Xk,
                                 const double* __restrict__ P,
                                 int m_local, int N,
                                 double alpha_x, double alpha_p) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_local * N;
    if (idx >= total) return;
    Xnew[idx] = alpha_x * Xk[idx] + alpha_p * P[idx];
}

struct Args {
    int N = 8000, b = 0, iters = 6;
    bool lookahead = true;
    bool lookahead_cli_set = false;
    std::int64_t M_fp64_words = 0;
    bool strict_b = false;
    MatrixMode matrix = MatrixMode::FP64;
    bool block_cyclic_layout = false;
    int px = 0, py = 0, pz = 0;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s.rfind("--N=", 0) == 0)     a.N = std::atoi(s.c_str() + 4);
        else if (s.rfind("--b=", 0) == 0)     a.b = std::atoi(s.c_str() + 4);
        else if (s.rfind("--iters=", 0) == 0) a.iters = std::atoi(s.c_str() + 8);
        else if (s.rfind("--M=", 0) == 0)     a.M_fp64_words = std::atoll(s.c_str() + 4);
        else if (s.rfind("--matrix=", 0) == 0) a.matrix = parse_matrix_mode(s.c_str() + 9);
        else if (s == "--mp") { if (a.matrix == MatrixMode::FP64) a.matrix = MatrixMode::FP64_MP; }
        else if (s == "--strict-b")          a.strict_b = true;
        else if (s == "--lookahead" || s == "--la") {
            a.lookahead = true;
            a.lookahead_cli_set = true;
        } else if (s == "--no-la" || s == "--no-lookahead") {
            a.lookahead = false;
            a.lookahead_cli_set = true;
        } else if (s.rfind("--layout=", 0) == 0) {
            const char* v = s.c_str() + 9;
            if (std::strcmp(v, "blockcyclic") == 0) a.block_cyclic_layout = true;
            else if (std::strcmp(v, "slab") == 0) a.block_cyclic_layout = false;
        } else if (s.rfind("--px=", 0) == 0) a.px = std::atoi(s.c_str() + 5);
        else if (s.rfind("--py=", 0) == 0) a.py = std::atoi(s.c_str() + 5);
        else if (s.rfind("--pz=", 0) == 0) a.pz = std::atoi(s.c_str() + 5);
    }
    return a;
}

#include "qdwh_fp32full.inl"

#include "qdwh_block_cyclic.inl"
#include "qdwh_full25d.inl"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int c;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c);

    Args A = parse_args(argc, argv);
    if (A.matrix == MatrixMode::FP32_FULL && A.lookahead) {
        if (!A.lookahead_cli_set) {
            if (_rank == 0) {
                fprintf(stdout,
                        "qdwh: fp32full has no inner-QR lookahead; running without LA (use explicit --la to probe; "
                        "pass --no-la to silence this line).\n");
                fflush(stdout);
            }
            A.lookahead = false;
        } else {
            if (_rank == 0)
                fprintf(stderr, "qdwh fp32full: --la not implemented; omit --la\n");
            MPI_Abort(MPI_COMM_WORLD, 93);
        }
    }
    const int P = c;
    int ngpu_q = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu_q));
    if (ngpu_q <= 0) {
        if (_rank == 0) fprintf(stderr, "qdwh: no CUDA devices\n");
        MPI_Abort(MPI_COMM_WORLD, 99);
    }
    const int bench_dev_q = _rank % ngpu_q;
    CUDA_CHECK(cudaSetDevice(bench_dev_q));
    if (A.M_fp64_words <= 0) {
        A.M_fp64_words = nextla_device_fast_memory_budget_elements(bench_dev_q, A.matrix);
        if (_rank == 0) {
            fprintf(stdout, "auto M (TeX fast memory): %lld matrix elements (σ=%zu B)\n",
                    (long long)A.M_fp64_words, nextla_matrix_element_bytes(A.matrix));
            fflush(stdout);
        }
    }
    if (A.block_cyclic_layout) {
        if (A.px <= 0 || A.py <= 0 || A.pz <= 0) {
            if (_rank == 0)
                fprintf(stderr,
                        "qdwh: --layout=blockcyclic requires --px= --py= --pz= (use Pz=1; P=Px*Py*Pz)\n");
            MPI_Abort(MPI_COMM_WORLD, 60);
        }
        if (A.pz != 1) {
            if (_rank == 0) fprintf(stderr, "qdwh: blockcyclic requires Pz=1\n");
            MPI_Abort(MPI_COMM_WORLD, 61);
        }
        if (P != A.px * A.py * A.pz) {
            if (_rank == 0)
                fprintf(stderr, "qdwh: MPI size P=%d != Px*Py*Pz=%d*%d*%d\n", P, A.px, A.py, A.pz);
            MPI_Abort(MPI_COMM_WORLD, 62);
        }
        if (A.b == 0 && A.M_fp64_words > 0) {
            int bb = default_block_b(A.M_fp64_words, (std::int64_t)A.N, A.px, A.py, A.pz);
            if (bb > 0) A.b = bb;
        }
        if (A.b == 0) {
            if (A.N <= 4000) A.b = 256;
            else if (A.N <= 8000) A.b = 384;
            else if (A.N <= 16000) A.b = 512;
            else if (A.N <= 32000) A.b = 768;
            else A.b = 1024;
        }
        int my_pz = _rank / (A.px * A.py);
        int my_px = (_rank / A.py) % A.px;
        int my_py = _rank % A.py;
        int rc = run_qdwh_bc_main(A, P, A.px, A.py, my_px, my_py, my_pz);
        MPI_Finalize();
        return rc;
    }

    // ── Full-2.5D Pz>1 dispatch (all 4 matrix modes) ──────────────────────
    {
        bool grid_cli = (A.px > 0 && A.py > 0 && A.pz > 0);
        Full25DGrid G_pre;
        if (grid_cli) {
            G_pre = resolve_full25d_grid(P, _rank, A.N, A.px, A.py, A.pz, A.M_fp64_words);
        } else if (A.M_fp64_words > 0) {
            G_pre = resolve_full25d_grid(P, _rank, A.N, 0, 0, 0, A.M_fp64_words);
        } else {
            G_pre.Px = 1; G_pre.Py = P; G_pre.Pz = 1;
        }
        bool use_full25d = (G_pre.Pz > 1 || (G_pre.Px > 1 && G_pre.Py > 1));
        if (use_full25d) {
            if (A.b == 0 && A.M_fp64_words > 0) {
                int bb = default_block_b(A.M_fp64_words, (std::int64_t)A.N, G_pre.Px, G_pre.Py, G_pre.Pz);
                if (bb > 0) A.b = bb;
            }
            if (A.b == 0) {
                if      (A.N <=  4000) A.b = 256;
                else if (A.N <=  8000) A.b = 384;
                else if (A.N <= 16000) A.b = 512;
                else if (A.N <= 32000) A.b = 768;
                else                   A.b = 1024;
            }
            A.px = G_pre.Px; A.py = G_pre.Py; A.pz = G_pre.Pz;
            Full25DGrid G = resolve_full25d_grid(P, _rank, A.N, A.px, A.py, A.pz, A.M_fp64_words);
            Full25DSubcomms S = build_full25d_subcomms(G);
            const bool use_mp_trail   = nextla_is_mp_trail_matrix(A.matrix);
            const bool use_tf32_trail = nextla_requests_tf32_matrix(A.matrix);
            if (_rank == 0) {
                const char* tag = (A.matrix == MatrixMode::FP32_FULL) ? "qdwh_full25d/fp32full"
                                 : (A.matrix == MatrixMode::FP64_MP_TF32) ? "qdwh_full25d/fp64mp_tf32"
                                 : (A.matrix == MatrixMode::FP64_MP)    ? "qdwh_full25d/fp64mp"
                                 : "qdwh_full25d/fp64";
                print_full25d_grid(G, tag, A.M_fp64_words, A.b);
            }
            int rc;
            if (A.matrix == MatrixMode::FP32_FULL) {
                rc = run_qdwh_full25d_fp32(A, G, S);
            } else {
                rc = run_qdwh_full25d_fp64(A, G, S, use_mp_trail, use_tf32_trail);
            }
            destroy_full25d_subcomms(S);
            MPI_Finalize();
            return rc;
        }
    }

    if (A.b == 0 && A.M_fp64_words > 0) {
        int bb = default_block_b(A.M_fp64_words, (std::int64_t)A.N, 1, 1, P);
        if (bb > 0) A.b = bb;
    }
    if (A.b == 0) {
        if (A.N <= 4000) A.b = 256;
        else if (A.N <= 8000) A.b = 384;
        else if (A.N <= 16000) A.b = 512;
        else if (A.N <= 32000) A.b = 768;
        else A.b = 1024;
    }
    if (A.matrix == MatrixMode::FP32_FULL) {
        return run_qdwh_fp32full_main(A, c);
    }
    const bool use_mp_inner = nextla_is_mp_trail_matrix(A.matrix);
    const bool use_tf32_trail = nextla_requests_tf32_matrix(A.matrix) && (NEXTLA_HAVE_CUBLAS_TF32 != 0);
    if (nextla_requests_tf32_matrix(A.matrix) && !use_tf32_trail) {
        if (_rank == 0) {
            fprintf(stderr,
                    "qdwh: --matrix=fp64mp_tf32 requires CUDA 11+ cuBLAS (CUBLAS_COMPUTE_32F_FAST_TF32).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 91);
    }
    nextla_maybe_print_tf32_trailing_banner_rank0(_rank, use_tf32_trail);
    if (A.N % c != 0) { if (_rank==0) fprintf(stderr,"N=%d not divisible by c=%d\n",A.N,c); MPI_Abort(MPI_COMM_WORLD,5); }
    int N = A.N, b = A.b, m_local = N / c;
    if (A.M_fp64_words > 0 && _rank == 0) {
        DerivedSchedule D1 = compute_degenerate_1d_schedule(c, N, A.M_fp64_words);
        fprintf(stdout, "%s\n", format_derived_schedule(D1).c_str());
        fflush(stdout);
    }
    if (A.strict_b && !b_in_window(b, c, N, 1, 1)) {
        if (_rank == 0) fprintf(stderr, "qdwh: b=%d violates §A3b window for c=%d, N=%d\n", b, c, N);
        MPI_Abort(MPI_COMM_WORLD, 55);
    }
    int m_st_local = 2 * m_local;
    int M_st = 2 * N;

    char tag[160];
    std::snprintf(tag, sizeof(tag), "qdwh_%s%s_it=%d", matrix_mode_tag(A.matrix), A.lookahead ? "+LA" : "", A.iters);

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" QDWH 2.5D         N=%d b=%d c=%d   variant=%s\n", N, b, c, tag);
        printf("=================================================================\n");
        fflush(stdout);
    }

    // NCCL bootstrap.
    ncclUniqueId nccl_id;
    if (_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, c, nccl_id, _rank));

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
    cublasHandle_t cublas_la;
    cusolverDnHandle_t cusolver_la;
    if (A.lookahead) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));       CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    // A_orig, X_k, X_new, stacked S, Q2 gather buffer, P_local.
    double *d_A = nullptr, *d_X = nullptr, *d_Xnew = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)m_local    * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,    (size_t)m_local    * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Xnew, (size_t)m_local    * N * sizeof(double)));
    {
        std::vector<double> host(m_local * (size_t)N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Stacked matrix S (2n × n), row-distributed: each rank m_st_local rows.
    double* d_S = nullptr;
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)m_st_local * N * sizeof(double)));

    // Inner-QR scratch (CQR2 style): Gram G (b×b), W (b × N), POTRF work.
    double *d_G = nullptr, *d_W = nullptr, *d_trace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G,     (size_t)b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,     (size_t)b * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(double)));
    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Look-ahead second-panel scratch.
    double *d_G2=nullptr, *d_potrf_work2=nullptr;
    int *d_info2=nullptr;
    if (A.lookahead) {
        CUDA_CHECK(cudaMalloc(&d_G2,           (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_potrf_work2,  potrf_lwork * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info2,        sizeof(int)));
    }

    // Q_2 AllGather scratch (we'll AllGather Q_2 rows from all ranks).
    double *d_Q2_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q2_recv, (size_t)m_local * N * (size_t)c * sizeof(double)));
    // P_local = Q_1_local @ Q_2_full^T  (m_local × N)
    double* d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, (size_t)m_local * N * sizeof(double)));
    // Packed contiguous Q_2 buffer (m_local × N) for the AllGather send.
    double* d_Q2_pack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q2_pack, (size_t)m_local * N * sizeof(double)));

    float* d_Wf = nullptr;
    float* d_pf_panel = nullptr;
    float* d_pf_tr = nullptr;
    if (use_mp_inner) {
        CUDA_CHECK(cudaMalloc(&d_Wf, (size_t)b * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pf_panel, (size_t)m_st_local * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pf_tr, (size_t)m_st_local * N * sizeof(float)));
    }

    const double one_d=1.0, zero_d=0.0, neg_one_d=-1.0;
    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;

    // CQR2 (passes=2) inner-QR Phase Q1 on the stacked S (m_st_local × N).
    auto phase_q1_cqr2 = [&](cudaStream_t s_use, cublasHandle_t cb, cusolverDnHandle_t cs,
                              double* Gb, double* work, int* info,
                              int k, int sb) {
        for (int it = 0; it < 2; ++it) {
            double* d_S_panel = d_S + (size_t)k * m_st_local;
            CUBLAS_CHECK(cublasDsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                      sb, m_st_local, &one_d, d_S_panel, m_st_local,
                                      &zero_d, Gb, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(Gb, Gb, (size_t)b * b, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_UPPER, sb, Gb, b, work, potrf_lwork, info));
            CUBLAS_CHECK(cublasDtrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      m_st_local, sb, &one_d, Gb, b, d_S_panel, m_st_local));
        }
    };

    auto phase_q2_cqr2 = [&](cudaStream_t s_use, cublasHandle_t cb,
                              int k, int sb, int col_start, int ncols) {
        double* d_S_panel = d_S + (size_t)k * m_st_local;
        double* d_S_tr    = d_S + (size_t)col_start * m_st_local;
        if (!use_mp_inner) {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_st_local,
                                      &one_d, d_S_panel, m_st_local, d_S_tr, m_st_local,
                                      &zero_d, d_W, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_st_local, ncols, sb,
                                      &neg_one_d, d_S_panel, m_st_local, d_W, b,
                                      &one_d, d_S_tr, m_st_local));
        } else {
            const size_t np = (size_t)m_st_local * sb;
            const size_t nt = (size_t)m_st_local * ncols;
            const int ntiles = 256;
            cast_d2f_q<<<(unsigned)((np + ntiles - 1) / ntiles), ntiles, 0, s_use>>>(d_S_panel, d_pf_panel, np);
            cast_d2f_q<<<(unsigned)((nt + ntiles - 1) / ntiles), ntiles, 0, s_use>>>(d_S_tr, d_pf_tr, nt);
            if (use_tf32_trail) {
#if NEXTLA_HAVE_CUBLAS_TF32
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                         sb, ncols, m_st_local,
                                         &one_f, d_pf_panel, CUDA_R_32F, m_st_local,
                                         d_pf_tr, CUDA_R_32F, m_st_local,
                                         &zero_f, d_Wf, CUDA_R_32F, b,
                                         CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                         sb, ncols, m_st_local,
                                         &one_f, d_pf_panel, m_st_local, d_pf_tr, m_st_local,
                                         &zero_f, d_Wf, b));
#endif
            } else {
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                         sb, ncols, m_st_local,
                                         &one_f, d_pf_panel, m_st_local, d_pf_tr, m_st_local,
                                         &zero_f, d_Wf, b));
            }
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_Wf, d_Wf, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            cast_f2d_q<<<(unsigned)(((size_t)sb * ncols + ntiles - 1) / ntiles), ntiles, 0, s_use>>>(
                d_Wf, d_W, (size_t)sb * ncols);
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                     m_st_local, ncols, sb,
                                     &neg_one_d, d_S_panel, m_st_local, d_W, b,
                                     &one_d, d_S_tr, m_st_local));
        }
    };

    // Inner 2.5D QR (CQR2 + optional Look-Ahead) on the stacked d_S.
    auto inner_qr = [&]() {
        int k = 0;
        if (!A.lookahead) {
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                phase_q1_cqr2(s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, k, sb);
                if (n_tr > 0) phase_q2_cqr2(s_comp, cublas, k, sb, k + sb, n_tr);
                k += sb;
            }
        } else {
            int sb = std::min(b, N - k);
            phase_q1_cqr2(s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, k, sb);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? std::min(b, N - next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    phase_q2_cqr2(s_comp, cublas, k, sb, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1_cqr2(s_la, cublas_la, cusolver_la, d_G2, d_potrf_work2, d_info2, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                }
                if (n_rest > 0) phase_q2_cqr2(s_comp, cublas, k, sb, rest_start, n_rest);
                CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                k = next_k;
                sb = next_sb;
                if (sb == 0) break;
            }
        }
    };

    // QDWH outer Halley iteration.
    //   Initial: X_0 = A / alpha where alpha is an upper bound for ||A||_2.
    //   We use Frobenius norm as a cheap upper bound: alpha = ||A||_F.
    //   l_0 = sigma_min(A) / alpha — for random Gaussian we use l_0 = 1/sqrt(N) (heuristic).
    auto run_qdwh = [&]() -> double {
        // alpha = ||A||_F  (cheap upper bound on ||A||_2)
        double frob_local = 0.0, frob_global = 0.0;
        // Compute on host (one-time, post-init); cuBLAS Dnrm2 is per-vector,
        // we use Dgemm-based trace trick:  alpha² = tr(A^T A).
        // For simplicity, do this via single Dnrm2 on flattened A.
        size_t na = (size_t)m_local * N;
        CUBLAS_CHECK(cublasDnrm2(cublas, na, d_A, 1, &frob_local));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        frob_local *= frob_local;
        MPI_Allreduce(&frob_local, &frob_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = sqrt(frob_global);
        if (alpha <= 0) alpha = 1.0;
        // X_0 = A / alpha
        double inv_alpha = 1.0 / alpha;
        {
            int threads = 256;
            long long total = (long long)m_local * N;
            long long blocks = (total + threads - 1) / threads;
            update_X_kernel<<<(unsigned)blocks, threads, 0, s_comp>>>(
                d_X, d_A, d_A, m_local, N, inv_alpha, 0.0);
        }

        // l_0 heuristic for Gaussian: l_0 = 1 / sqrt(N) (a few iters absorb the slack)
        double l = 1.0 / sqrt((double)N);
        if (l < 1e-15) l = 1e-15;

        for (int kit = 0; kit < A.iters; ++kit) {
            // Halley weights.
            double l2 = l * l;
            double dd = pow(4.0 * (1.0 - l2) / (l2 * l2), 1.0 / 3.0);
            double sd = sqrt(1.0 + dd);
            double inner = std::max(0.0, 8.0 - 4.0 * dd + 8.0 * (2.0 - l2) / (l2 * sd));
            double a_k = sd + 0.5 * sqrt(inner);
            double b_k = (a_k - 1.0) * (a_k - 1.0) / 4.0;
            double c_k = a_k + b_k - 1.0;
            double scale_X = sqrt(c_k);

            // Build stacked S.
            {
                int threads = 256;
                long long total = (long long)m_st_local * N;
                long long blocks = (total + threads - 1) / threads;
                fill_stacked_kernel<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_S, d_X, m_local, N, _rank, scale_X);
            }

            // Inner QR on d_S → d_S holds Q (m_st_local × N orthonormal per CQR2 fixup).
            inner_qr();

            // Extract Q_1 (top m_local rows of each rank's local S) and Q_2 (bottom m_local rows).
            // Q_1 starts at d_S + 0   (m_local × N column-major, ld=m_st_local)
            // Q_2 starts at d_S + m_local (m_local × N column-major, ld=m_st_local)

            // AllGather Q_2 across ranks (size m_local × N per rank).
            // We need to layout the AllGather output as a full N × N matrix.
            // Use ncclAllGather + a rearrangement: but the columns are stride
            // m_st_local in d_S, not contiguous.  So we first PACK Q_2 into a
            // contiguous m_local × N buffer (d_W reused), then AllGather.
            CUDA_CHECK(cudaMemcpy2DAsync(d_Q2_pack, m_local * sizeof(double),
                                         d_S + m_local, m_st_local * sizeof(double),
                                         m_local * sizeof(double), N,
                                         cudaMemcpyDeviceToDevice, s_comp));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllGather(d_Q2_pack, d_Q2_recv,
                                     (size_t)m_local * N, ncclDouble,
                                     nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));

            // P_local = Q_1_local · Q_2_full^T
            // Q_1_local: m_local × N column-major with ld=m_st_local, starting at d_S
            // Q_2_full:  N × N column-major with ld=N — but our AllGather output
            //            is a stack of c blocks each m_local × N column-major.
            //            For the GEMM, we treat it as N × N column-major if all
            //            blocks pack as [block_0 (m_local × N), block_1, ..., block_{c-1}]
            //            concatenated in memory.  Each block in column-major has
            //            stride m_local within a column.  Concatenated, column j
            //            spans rows [r*m_local + i_local] for r = 0..c-1.
            //            That IS column-major with ld = N (= c * m_local).
            //   So d_Q2_recv interpreted as N × N column-major, ld = m_local,
            //   actually NO — the blocks are stored CONSECUTIVELY in memory:
            //   d_Q2_recv[r*m_local*N + j*m_local + i_local] = rank r's Q_2[i_local, j].
            //   To make a single column-major N × N with ld=N, we'd need to
            //   interleave.  Instead, treat d_Q2_recv as a sequence of
            //   m_local × N column-major slabs, and do c separate GEMMs:
            //     P_local = sum_r Q_1_local · Q_2_r^T
            //   where Q_1_local has shape m_local × N and each Q_2_r is m_local × N.
            //   Actually, Q_1 Q_2^T is a single GEMM of (m_local × N) · (N × N)^T,
            //   but the issue is the row-ordering of Q_2 across ranks.  In our
            //   stacked layout the ranks contain Q_2 rows in order: rank 0 has
            //   rows [0, m_local), rank 1 has rows [m_local, 2*m_local), ...
            //   So d_Q2_recv as a concatenated buffer IS N × N column-major with
            //   ld = N — because in column j, the elements at offsets [0, N) are
            //   the N rows of Q_2 in correct order.  Let me re-derive:
            //     d_Q2_recv[byte_offset] = recv buffer linear address
            //     The AllGather puts rank r's sendbuf at offset r * sendcount in
            //     recvbuf.  sendcount = m_local * N doubles.
            //     So d_Q2_recv[r * m_local * N + j * m_local + i_local] =
            //       rank r's d_W (i.e. Q_2_local of rank r)[i_local + j * m_local]
            //       = rank r's Q_2 row i_local, column j.
            //     The global row index of this Q_2 element is rank * m_local + i_local
            //     because Q_2 is N × N row-distributed.
            //   For d_Q2_recv to be column-major N × N with ld = N, we need:
            //     d_Q2_recv[i_global + j * N] = Q_2[i_global, j]
            //     i_global = r * m_local + i_local, so
            //     d_Q2_recv[r * m_local + i_local + j * N] = ...
            //   But AllGather gives  d_Q2_recv[r * m_local * N + j * m_local + i_local].
            //   These are different memory layouts.  The AllGather output is
            //   NOT directly column-major N × N.  We need to rearrange or do
            //   c separate GEMMs.
            //
            // Doing c separate GEMMs:  P_local = sum_r Q_1_local · (Q_2_r)^T
            //   where Q_2_r is rank r's m_local × N slab in AllGather output.
            //   But Q_1_local · Q_2_r^T has shape m_local × m_local (not what we want).
            //   That's wrong — we want P_local of shape m_local × N.
            //
            //  Hmm.  Let me think again.  Q_1 is N × N (row-distributed: m_local
            //  rows per rank).  Q_2 is N × N (row-distributed similarly).
            //  Q_1 Q_2^T is N × N.
            //  Q_1 Q_2^T [i, j] = sum_k Q_1[i, k] * Q_2[j, k].
            //  For rank r holding Q_1_local rows [r*m_local : (r+1)*m_local) of Q_1:
            //    P_local[i_local, j] = sum_k Q_1_local[i_local, k] * Q_2[j, k]
            //                        = sum_k Q_1_local[i_local, k] * Q_2_all[j, k]
            //  Each P_local row needs ALL of Q_2 (size N × N).  So we AllGather
            //  Q_2 fully.  Then GEMM:  P_local = Q_1_local * Q_2_full^T.
            //  Q_1_local is m_local × N (cols).  Q_2_full is N × N.
            //  Result P_local is m_local × N.
            //  GEMM:  P_local^T = Q_2_full * Q_1_local^T
            //          equiv:   P_local = Q_1_local * Q_2_full^T
            //  GEMM dims:  P_local [m_local × N] = Q_1_local [m_local × N] · Q_2_full^T [N × N]
            //
            //  To do this GEMM cleanly we need Q_2_full as N × N column-major.
            //  But our AllGather output is in rank-block form.  We'll do c
            //  small GEMMs instead:
            //    P_local = sum_{r=0..c-1} Q_1_local[:, ?] * Q_2_r^T
            //  where Q_2_r is rank r's Q_2 slab (m_local × N).
            //  Wait that doesn't reduce — Q_1_local doesn't index over rank.
            //
            //  Let me reconsider.  In the AllGather output, each rank r's
            //  Q_2 contribution is the Q_2 rows for global rows [r*m_local : (r+1)*m_local).
            //  So Q_2_full[r*m_local + i_local, :] = d_Q2_recv[r*m_local*N + i_local + j*m_local].
            //
            //  The Q_1_local · Q_2_full^T:
            //    P_local[i_local, j] = sum_{i_global=0..N-1} Q_1_local[i_local, i_global] * Q_2_full[j, i_global]
            //  But wait, Q_2_full[j, i_global] needs j as a row index — and j ranges over [0, N), which is the column index of P_local.  So we're indexing into ALL of Q_2.
            //
            //  OK so for each i_local, j:
            //    P_local[i_local, j] = sum_{k=0..N-1} Q_1_local[i_local, k] * Q_2_full[j, k]
            //  This is a standard m_local × N matrix times N × N matrix-transpose.
            //
            //  cublasDgemm with C = A * B^T:
            //    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m_local, N, N,
            //                &alpha, A=Q_1_local (m_local × N, ld=m_st_local), lda=m_st_local,
            //                        B=Q_2_full (N × N, ld=N), ldb=N,
            //                &beta, C=P_local (m_local × N, ld=m_local), ldc=m_local)
            //
            //  But ldb must be the leading dim of Q_2_full in column-major, which
            //  in our rank-block AllGather output is NOT N.  So we need to
            //  transpose / rearrange first.
            //
            //  Alternative: do c smaller GEMMs.  Per rank r:
            //    contribute = Q_1_local · Q_2_r^T where Q_2_r is m_local × N (rank r's rows of Q_2)
            //    NB: this is m_local × m_local, not m_local × N
            //  But P_local is m_local × N, with columns indexed by [0, N).
            //  Wait, my derivation above: P_local[i_local, j] = sum_k Q_1_local[i_local, k] * Q_2_full[j, k].
            //  j ranges over [0, N) (rows of Q_2_full).  Q_2_full has N rows distributed:
            //    rows in [r*m_local, (r+1)*m_local) live on rank r.
            //  Split sum by rank:
            //    P_local[i_local, j] = sum_r [if j ∈ [r*m_local, (r+1)*m_local)] sum_k Q_1_local[i_local, k] * Q_2_r[j - r*m_local, k]
            //  So for each j in [r*m_local, (r+1)*m_local) (some rank r),
            //    P_local[i_local, j] = sum_k Q_1_local[i_local, k] * Q_2_r[j - r*m_local, k]
            //                        = (Q_1_local · Q_2_r^T)[i_local, j - r*m_local]
            //  So P_local can be filled BLOCK-BY-BLOCK in columns:
            //    P_local[:, r*m_local : (r+1)*m_local] = Q_1_local · Q_2_r^T
            //  Each block GEMM:  (m_local × N) · (N × m_local) = m_local × m_local
            //  c such GEMMs per rank.  Total work per rank: c × m_local² × N = m_local × N × m_local × c = same as a single (m_local × N × N) GEMM.
            //
            //  Block GEMM:
            //    cublasDgemm(N, T, m_local, m_local, N,
            //                alpha, Q_1_local (m_local × N), m_st_local,
            //                       Q_2_r (m_local × N, ld=m_local in d_Q2_recv layout), m_local,
            //                beta, P_local + r*m_local*m_local cols (m_local × m_local in P_local layout), m_local)
            //
            //  Where d_Q2_recv[r*m_local*N + ...] is a m_local × N column-major matrix with ld=m_local.
            //  That works!  Let me code this up.

            for (int r = 0; r < c; ++r) {
                const double* Q_2_r = d_Q2_recv + (size_t)r * m_local * N;
                double*       P_block = d_P + (size_t)r * m_local * m_local; // mistake: P_local is m_local × N column-major
                // Actually P_local is m_local × N column-major, ld = m_local.
                // Block for cols [r*m_local : (r+1)*m_local) starts at offset
                //   d_P + (size_t)(r * m_local) * m_local   (since ld = m_local)
                double* P_cols = d_P + (size_t)(r * m_local) * m_local;
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                          m_local, m_local, N,
                                          &one_d, d_S, m_st_local,           // Q_1_local
                                                  Q_2_r, m_local,             // Q_2_r ^T
                                          &zero_d, P_cols, m_local));
            }

            // Update X_{k+1} = (b_k/c_k) X_k + (1/sqrt(c_k)) (a_k - b_k/c_k) P
            double alpha_x = b_k / c_k;
            double alpha_p = (a_k - b_k / c_k) / sqrt(c_k);
            {
                int threads = 256;
                long long total = (long long)m_local * N;
                long long blocks = (total + threads - 1) / threads;
                update_X_kernel<<<(unsigned)blocks, threads, 0, s_comp>>>(
                    d_Xnew, d_X, d_P, m_local, N, alpha_x, alpha_p);
            }
            std::swap(d_X, d_Xnew);

            // Update l for next iter: l_{k+1} = l (a + b l^2) / (1 + c l^2)
            double num = l * (a_k + b_k * l * l);
            double den = 1.0 + c_k * l * l;
            l = num / den;
            if (std::abs(l - 1.0) < 1e-15) {
                // Converged early; for the bench we still run full iters to
                // ensure deterministic timing.
            }
        }

        return l;
    };

    auto reset_A = [&]() {
        // Re-init A from same seed for deterministic re-runs.
        std::vector<double> host(m_local * (size_t)N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    };

    // Warmup + validation.
    for (int i = 0; i < 2; ++i) { reset_A(); run_qdwh(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    // Validation: max|diag(U'U) - 1| over the orthogonal factor U = X_final.
    {
        double* d_QtQ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_QtQ, (size_t)N * N * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  N, m_local, &one_d, d_X, m_local,
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
            printf("  Validation     variant=%s N=%d c=%d  max|diag(U'U)-1| = %.2e\n",
                   tag, N, c, max_dev);
            fflush(stdout);
        }
        cudaFree(d_QtQ);
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
        if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        printf("  %-30s  N=%d b=%d c=%d  tmin=%9.2f ms  tmed=%9.2f ms\n",
               tag, N, b, c, times[0], times[nrun/2]);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, c);
        printf("METRICS bench=qdwh_2p5d matrix=%s N=%d b=%d c=%d passes=1 ", matrix_mode_tag(A.matrix), N, b, c);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", times[nrun / 2]);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Xnew); cudaFree(d_S);
    cudaFree(d_G); cudaFree(d_W); cudaFree(d_trace);
    if (use_mp_inner) {
        cudaFree(d_Wf);
        cudaFree(d_pf_panel);
        cudaFree(d_pf_tr);
    }
    cudaFree(d_potrf_work); cudaFree(d_info);
    cudaFree(d_Q2_recv); cudaFree(d_P); cudaFree(d_Q2_pack);
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
