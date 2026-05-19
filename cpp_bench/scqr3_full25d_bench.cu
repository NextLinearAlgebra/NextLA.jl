// Full 2.5D sCQR3 / CQR2 — Conflux-style [P_x, P_y, P_z] decomposition.
//
//   PDF / literature pins (expand figure § when SC_factorization.pdf is available):
//     * Kwasniewski et al., SC'21 (Conflux) — block-cyclic assignment, replication.
//     * TeX: NextLA.jl/qr_schur_xpartition.tex §A3aprime, §A3derive, §A3b, §A1.
//
//   Faithful implementation of the X-partition / 2.5D scheduling derived in
//   qr_schur_xpartition.tex §A.3 and Kwasniewski et al. SC'21:
//
//     Processor grid:  [P_x, P_y, P_z] = [sqrt(P_1), sqrt(P_1), c]
//                       with  P_1 = P / c   and   c = floor(P^{1/3})
//                       (or user-specified --px --py --pz)
//     Block size:      b = max(c, a * c * (PM/N^2))  =  a * c    (small block;
//                       Conflux's "v = a * c" with a a small constant).
//                       For empirical reasons we let the user pick b via --b
//                       so the X-partition cube  b ~ sqrt(M)  remains an
//                       admissible upper bound.  Default is the H200
//                       sqrt(M) heuristic.
//     Data layout:     A is N x N row-and-column distributed:
//                       rank (px, py, pz) holds rows
//                         [ (pz*P_x + px) * m_loc, (pz*P_x + px + 1) * m_loc )
//                       and columns
//                         [ py * n_loc, (py + 1) * n_loc )
//                       with m_loc = N / (P_x * P_z), n_loc = N / P_y.
//     Phase Q1:        Panel column slab of width sb at column k.  py_panel
//                       = k / n_loc.  Within column group {(px, py_panel, pz)}
//                       (P_x * P_z = P_1 ranks):
//                         1. local SYRK on (m_loc x sb) slab,
//                         2. AllReduce(G) across the column group,
//                         3. POTRF(G) replicated,
//                         4. local TRSM giving Q in-place.
//                       Phase Q3 collective:  AllReduce(G) over P_1 ranks
//                       ($\le 6 b^2$ words per processor).
//     Broadcast Q:     Send Q's local rows from py_panel to all py' in the
//                       row group {(px, py', pz) : py'}  (P_y ranks).  After
//                       broadcast every rank has its local rows of Q.
//                       Phase Q3' collective:  AllGather (or NCCL bcast) of
//                       Q's row-block ($m b / (P_x P_z)$ words per processor).
//     Phase Q2:        Local W = Q_loc^T A_trail_loc on every rank;
//                       AllReduce(W) over the column group of that A_trail
//                       column-piece (P_1 ranks); A_trail -= Q W locally.
//                       Phase Q4 collective:  AllReduce(W) over P_1 ranks
//                       ($b \cdot n_{tr}$ words).
//
//   Flags:
//     --N, --b, --passes, --px, --py, --pz, --la / --no-la   (lookahead on by default, §A1)
//     --matrix=fp64|fp64mp|fp64mp_tf32|fp32full   tri-mode + TF32 trailing variant (CUDA 11+)
//     --layout=bc_2p5d_degenerate|blockcyclic   slab = row×column specialization (default); blockcyclic = 2D block-cyclic (Pz=1)
//     --oom-probe                 exit 77 if primary cudaMalloc fails (MPI_Allreduce across ranks)
//     --M=          optional override of TeX fast-memory budget (matrix elements, σ-sized; see nextla_fast_memory.hpp)
//     --smoke                     legacy cbrt grid + legacy b(N); skips auto M from device memory
//     --strict-derived-grid       abort if no (c,p_r,p_c) fits §A3b (no 1×P×1 fallback)
//
//   Comparison:
//     cuSOLVERMp with grid [P_x, P_y * P_z] (effective 2D, no replication)
//     on the same P GPUs is the apples-to-apples NVIDIA baseline.

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
#include <cstring>

#include "derived_schedule.hpp"
#include "layout_verify.hpp"
#include "block_cyclic.hpp"
#include "matrix_mode.hpp"
#include "nextla_mp_trail.hpp"
#include "nextla_fast_memory.hpp"
#include "comm_groups.cuh"
#include "bench_vendor_metrics.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>

// CUBLAS_COMPUTE_32F_FAST_TF32 is an enum value in cublas_api.h, not a
// preprocessor macro, so a #if defined() test on it always fails.  The
// API has been in cuBLAS since CUDA 11 / driver-side it's always present
// on cuBLAS 11+; gate on CUDART_VERSION instead.
#if (CUDART_VERSION >= 11000)
#define NEXTLA_HAVE_CUBLAS_TF32 1
#else
#define NEXTLA_HAVE_CUBLAS_TF32 0
#endif

#define CUDA_CHECK(stmt) do { cudaError_t e=(stmt); if(e!=cudaSuccess){ fprintf(stderr,"[r%d] CUDA %s @ %s:%d\n",_rank,cudaGetErrorString(e),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUBLAS_CHECK(stmt) do { cublasStatus_t s=(stmt); if(s!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"[r%d] cuBLAS %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)
#define CUSOLVER_CHECK(stmt) do { cusolverStatus_t s=(stmt); if(s!=CUSOLVER_STATUS_SUCCESS){ fprintf(stderr,"[r%d] cuSOLVER %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,3);} } while(0)
#define NCCL_CHECK(stmt) do { ncclResult_t r=(stmt); if(r!=ncclSuccess){ fprintf(stderr,"[r%d] NCCL %s @ %s:%d\n",_rank,ncclGetErrorString(r),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,4);} } while(0)

static int _rank = 0;

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

__global__ void trace_b_kernel_f(const float* G, int ldg, int b, float* out) {
    __shared__ float sh[1024];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int j = tid; j < b; j += blockDim.x) acc += G[j + (long long)j * ldg];
    sh[tid] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    if (tid == 0) out[0] = sh[0];
}
__global__ void shift_diag_from_trace_f_kernel(float* G, int ldg, int b, const float* tr, float coef) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < b) G[j + (long long)j * ldg] += coef * tr[0];
}

__global__ void cast_d2f_ker(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ void cast_f2d_ker(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}

struct Args {
    int N = 8000, b = 0, passes = 3;
    int px = 0, py = 0, pz = 0;
    bool lookahead = true;          // TeX §A1-lookahead default (ablation: --no-la)
    std::int64_t M_fp64_words = 0;
    bool strict_b = false;
    bool strict_grid = false;       // with --M= and manual grid, require match to derived
    bool strict_derived_grid = false;  // no 1×P×1 fallback when --M= derivation fails
    bool smoke_bench = false;       // legacy cbrt grid + legacy b(N); skips auto M from device
    MatrixMode matrix = MatrixMode::FP64;
    bool block_cyclic_layout = false;  // --layout=blockcyclic (else Conflux slab specialization)
    bool oom_probe = false;            // exit 77 if cudaMalloc fails on primary A alloc
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s.rfind("--N=", 0) == 0)      a.N = std::atoi(s.c_str() + 4);
        else if (s.rfind("--b=", 0) == 0)      a.b = std::atoi(s.c_str() + 4);
        else if (s.rfind("--passes=", 0) == 0) a.passes = std::atoi(s.c_str() + 9);
        else if (s.rfind("--px=", 0) == 0)     a.px = std::atoi(s.c_str() + 5);
        else if (s.rfind("--py=", 0) == 0)     a.py = std::atoi(s.c_str() + 5);
        else if (s.rfind("--pz=", 0) == 0)     a.pz = std::atoi(s.c_str() + 5);
        else if (s.rfind("--M=", 0) == 0)      a.M_fp64_words = std::atoll(s.c_str() + 4);
        else if (s == "--strict-b")          a.strict_b = true;
        else if (s == "--strict-grid")       a.strict_grid = true;
        else if (s == "--strict-derived-grid") a.strict_derived_grid = true;
        else if (s == "--smoke")            a.smoke_bench = true;
        else if (s.rfind("--matrix=", 0) == 0) a.matrix = parse_matrix_mode(s.c_str() + 9);
        else if (s.rfind("--layout=", 0) == 0) {
            const char* v = s.c_str() + 9;
            if (std::strcmp(v, "blockcyclic") == 0) a.block_cyclic_layout = true;
            else if (std::strcmp(v, "slab") == 0) a.block_cyclic_layout = false;
        }
        else if (s == "--oom-probe") a.oom_probe = true;
        else if (s == "--no-la" || s == "--no-lookahead") a.lookahead = false;
        else if (s == "--la" || s == "--lookahead") a.lookahead = true;
    }
    if (a.passes < 1 || a.passes > 3) a.passes = 3;
    return a;
}

// Resolve [px, py, pz] from P and CLI hints. If none specified:
//   * With --M= : TeX §A3derive replication search (see derived_schedule.hpp).
//   * Else      : legacy scan c from floor(cbrt(P))+1 downward (original bench).
static void resolve_grid(int P, Args& A) {
    if (A.px > 0 && A.py > 0 && A.pz > 0) {
        if (A.px * A.py * A.pz != P) {
            if (_rank == 0) fprintf(stderr, "px*py*pz = %d != P = %d\n", A.px*A.py*A.pz, P);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        return;
    }
    if (A.M_fp64_words > 0) {
        DerivedSchedule D = compute_derived_schedule(P, A.N, A.M_fp64_words, A.strict_derived_grid);
        if (_rank == 0) {
            fprintf(stdout, "%s\n", format_derived_schedule(D).c_str());
            fflush(stdout);
        }
        if (!D.ok) {
            if (_rank == 0) fprintf(stderr, "derived schedule: %s\n", D.error.c_str());
            MPI_Abort(MPI_COMM_WORLD, 51);
        }
        A.px = D.px;
        A.py = D.py;
        A.pz = D.pz;
        if (A.b == 0) A.b = D.b;
        return;
    }
    if (!A.smoke_bench) {
        if (_rank == 0) {
            fprintf(stderr,
                    "scqr3_full25d: omitting --px/--py/--pz requires positive auto M (CUDA device) or --M=, "
                    "or add --smoke for legacy heuristic (small runs only).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 52);
    }
    // Legacy default: pick (px, py, pz) with px*py*pz = P, px = py = sqrt(P/c) integer,
    // c = floor(P^{1/3})-style scan from above.
    for (int c = (int)std::cbrt((double)P) + 1; c >= 1; --c) {
        if (P % c != 0) continue;
        int p1 = P / c;
        int s = (int)std::sqrt((double)p1);
        if (s * s == p1) {
            A.px = A.py = s;
            A.pz = c;
            return;
        }
    }
    A.px = 1; A.py = P; A.pz = 1;
}

static void fill_default_b_from_N_heuristic(int N, int* b_io) {
    if (*b_io > 0) return;
    if (N <= 4000) *b_io = 363;
    else if (N <= 8000) *b_io = 512;
    else if (N <= 16000) *b_io = 512;
    else if (N <= 32000) *b_io = 512;
    else if (N <= 64000) *b_io = 1024;
    else *b_io = 1024;
}

// §A3b default b: min(floor(sqrt(M)), floor(N/sqrt(P1))) rounded to a multiple of c (derived_schedule.hpp).
static void finalize_b_from_M_and_geometry(std::int64_t M, int N, int px, int py, int pz, int* b_io) {
    if (*b_io > 0) return;
    if (M > 0) {
        int bb = default_block_b(M, (std::int64_t)N, px, py, pz);
        if (bb > 0) *b_io = bb;
    }
    fill_default_b_from_N_heuristic(N, b_io);
}

// FP32-full Path (s): same slab layout and comm groups as double; optional
// two-stream lookahead (--la default) matching the double path.
static int run_fp32_body(const Args& A, int N, int b, int Px, int Py, int Pz,
                         int my_px, int my_py, int my_pz, int m_loc, int n_loc,
                         int col_rank, ColRowNccl* cr) {
    (void)my_px;
    (void)my_pz;
    ncclComm_t nccl_col = cr->nccl_col;
    ncclComm_t nccl_row = cr->nccl_row;

    cudaStream_t s_comp, s_comm, s_la;
    CUDA_CHECK(cudaStreamCreate(&s_comp));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    const bool eff_la = A.lookahead;
    if (eff_la) CUDA_CHECK(cudaStreamCreate(&s_la));
    cudaEvent_t e_comp_done, e_ar_done, e_panel_done, e_next_ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_panel_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e_next_ready, cudaEventDisableTiming));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, s_comp));
    cusolverDnHandle_t cusolver;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s_comp));
    cublasHandle_t cublas_la{};
    cusolverDnHandle_t cusolver_la{};
    if (eff_la) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));
        CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    float *d_A = nullptr, *d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)m_loc * n_loc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_loc * n_loc * sizeof(float)));
    {
        std::vector<float> host((size_t)m_loc * n_loc);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<float> nrm(0.f, 1.f);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A, (size_t)m_loc * n_loc * sizeof(float), cudaMemcpyDeviceToDevice));

    float* d_G = nullptr;
    float* d_W = nullptr;
    float* d_panel_bcast = nullptr;
    float* d_trace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)b * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * n_loc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_panel_bcast, (size_t)m_loc * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(float)));

    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    float* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)potrf_lwork * sizeof(float)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    float *d_G2 = nullptr, *d_potrf_work2 = nullptr, *d_panel_bcast2 = nullptr;
    int* d_info2 = nullptr;
    if (eff_la) {
        CUDA_CHECK(cudaMalloc(&d_G2, (size_t)b * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_panel_bcast2, (size_t)m_loc * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_potrf_work2, (size_t)potrf_lwork * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_info2, sizeof(int)));
    }

    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;

    auto sb_clipped = [&](int k) {
        int py_panel = k / n_loc;
        int next_py_boundary = (py_panel + 1) * n_loc;
        int avail = std::min(N - k, next_py_boundary - k);
        return std::min(b, avail);
    };

    auto phase_q1_f = [&](int k, int sb, int passes, cudaStream_t s_use, cublasHandle_t cb, cusolverDnHandle_t cs,
                          float* Gb, float* potw, int* info, float* p_bcast) {
        int py_panel = k / n_loc;
        int local_k = k - py_panel * n_loc;
        if (my_py == py_panel) {
            float coef = 11.f * ((float)N * sb + (float)sb * (sb + 1)) * 1.1920929e-07f;
            float* A_panel = d_A + (size_t)local_k * m_loc;
            for (int it = 0; it < passes; ++it) {
                CUBLAS_CHECK(cublasSsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                         sb, m_loc, &one_f, A_panel, m_loc,
                                         &zero_f, Gb, b));
                CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
                CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
                NCCL_CHECK(ncclAllReduce(Gb, Gb, (size_t)b * b, ncclFloat, ncclSum, nccl_col, s_comm));
                CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
                CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
                if (it == 0) {
                    trace_b_kernel_f<<<1, std::min(sb, 1024), 0, s_use>>>(Gb, b, sb, d_trace);
                    shift_diag_from_trace_f_kernel<<<(sb + 255) / 256, 256, 0, s_use>>>(Gb, b, sb, d_trace, coef);
                }
                CUSOLVER_CHECK(cusolverDnSpotrf(cs, CUBLAS_FILL_MODE_UPPER, sb, Gb, b, potw, potrf_lwork, info));
                CUBLAS_CHECK(cublasStrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                         m_loc, sb, &one_f, Gb, b, A_panel, m_loc));
            }
            CUDA_CHECK(cudaMemcpyAsync(p_bcast, d_A + (size_t)local_k * m_loc,
                                       (size_t)m_loc * sb * sizeof(float), cudaMemcpyDeviceToDevice, s_use));
        }
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclBroadcast(p_bcast, p_bcast, (size_t)m_loc * sb, ncclFloat,
                                 py_panel, nccl_row, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
    };

    auto phase_q2_f = [&](int k, int sb, cudaStream_t s_use, cublasHandle_t cb, float* p_panel) {
        int my_col_start_global = my_py * n_loc;
        int trail_global_start = k + sb;
        int local_start = std::max(0, trail_global_start - my_col_start_global);
        int local_end = n_loc;
        int ncols = local_end - local_start;
        if (ncols <= 0) return;
        float* A_trail_loc = d_A + (size_t)local_start * m_loc;
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                 sb, ncols, m_loc,
                                 &one_f, p_panel, m_loc,
                                         A_trail_loc, m_loc,
                                 &zero_f, d_W, b));
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_col, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m_loc, ncols, sb,
                                 &neg_one_f, p_panel, m_loc,
                                             d_W, b,
                                 &one_f, A_trail_loc, m_loc));
    };

    auto run_qr_f = [&]() {
        int k = 0;
        if (!eff_la) {
            while (k < N) {
                int sb = sb_clipped(k);
                phase_q1_f(k, sb, A.passes, s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, d_panel_bcast);
                if (k + sb < N) phase_q2_f(k, sb, s_comp, cublas, d_panel_bcast);
                k += sb;
            }
        } else {
            int sb = sb_clipped(k);
            phase_q1_f(k, sb, A.passes, s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, d_panel_bcast);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? sb_clipped(next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest_total = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    phase_q2_f(k, sb, s_comp, cublas, d_panel_bcast);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1_f(next_k, next_sb, A.passes, s_la, cublas_la, cusolver_la,
                               d_G2, d_potrf_work2, d_info2, d_panel_bcast2);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                    CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                    std::swap(d_panel_bcast, d_panel_bcast2);
                    std::swap(d_G, d_G2);
                    std::swap(d_potrf_work, d_potrf_work2);
                    std::swap(d_info, d_info2);
                    k = next_k;
                    sb = next_sb;
                } else {
                    if (n_rest_total > 0) phase_q2_f(k, sb, s_comp, cublas, d_panel_bcast);
                    break;
                }
            }
        }
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, (size_t)m_loc * n_loc * sizeof(float), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) { reset_A(); run_qr_f(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    {
        float* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)n_loc * n_loc * sizeof(float)));
        CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                 n_loc, m_loc, &one_f, d_A, m_loc,
                                 &zero_f, d_GG, n_loc));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        NCCL_CHECK(ncclAllReduce(d_GG, d_GG, (size_t)n_loc * n_loc, ncclFloat, ncclSum, nccl_col, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        float max_local = 0.f;
        if (col_rank == 0) {
            for (int j = 0; j < n_loc; ++j) {
                float v;
                CUDA_CHECK(cudaMemcpy(&v, d_GG + j + (size_t)j * n_loc, sizeof(float), cudaMemcpyDeviceToHost));
                float dev = std::fabs(v - 1.f);
                if (dev > max_local) max_local = dev;
            }
        }
        float max_global = 0.f;
        MPI_Allreduce(&max_local, &max_global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (_rank == 0) {
            printf("  Validation(fp32) variant=p25d_fp32full passes=%d%s N=%d grid=[%d,%d,%d]  max|diag(Q'Q)-1| = %.2e\n",
                   A.passes, eff_la ? "+LA" : "", N, Px, Py, Pz, (double)max_global);
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
        run_qr_f();
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
        printf("  %-30s  N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               "p25d_fp32full", N, b, Px, Py, Pz, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, Px * Py * Pz);
        printf("METRICS bench=scqr3_full25d matrix=fp32full layout=bc_2p5d_degenerate N=%d b=%d Px=%d Py=%d Pz=%d passes=%d ",
               N, b, Px, Py, Pz, A.passes);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_A_orig); cudaFree(d_G); cudaFree(d_W); cudaFree(d_panel_bcast); cudaFree(d_trace);
    cudaFree(d_potrf_work); cudaFree(d_info);
    if (eff_la) {
        cudaFree(d_G2); cudaFree(d_potrf_work2); cudaFree(d_info2); cudaFree(d_panel_bcast2);
        cublasDestroy(cublas_la); cusolverDnDestroy(cusolver_la);
        cudaStreamDestroy(s_la);
    }
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp);
    cudaStreamDestroy(s_comm);
    cudaEventDestroy(e_comp_done);
    cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done);
    cudaEventDestroy(e_next_ready);
    return 0;
}

#include "scqr3_block_cyclic.inl"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int P;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    Args A = parse_args(argc, argv);

    int ngpu_devpick = 0;
    cudaError_t gce = cudaGetDeviceCount(&ngpu_devpick);
    if (gce != cudaSuccess || ngpu_devpick <= 0) {
        if (_rank == 0)
            fprintf(stderr, "scqr3_full25d: cudaGetDeviceCount failed or zero devices (%s)\n",
                    cudaGetErrorString(gce));
        MPI_Abort(MPI_COMM_WORLD, 99);
    }
    const int bench_dev = _rank % ngpu_devpick;
    CUDA_CHECK(cudaSetDevice(bench_dev));

    if (!A.smoke_bench && A.M_fp64_words <= 0) {
        A.M_fp64_words = nextla_device_fast_memory_budget_elements(bench_dev, A.matrix);
        if (_rank == 0) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, bench_dev);
            fprintf(stdout,
                    "auto M (TeX fast memory): %lld matrix elements (σ=%zu B, GPU mem %.2f GiB, frac=%.4g)\n",
                    (long long)A.M_fp64_words, nextla_matrix_element_bytes(A.matrix),
                    (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), nextla_fastmem_frac_from_env());
            fflush(stdout);
        }
    }

    resolve_grid(P, A);

    finalize_b_from_M_and_geometry(A.M_fp64_words, A.N, A.px, A.py, A.pz, &A.b);

    int N = A.N, b = A.b;
    int Px = A.px, Py = A.py, Pz = A.pz, P1 = Px * Py;

    if (A.block_cyclic_layout) {
        if (Pz != 1) {
            if (_rank == 0) fprintf(stderr, "blockcyclic layout requires Pz=1 (got Pz=%d)\n", Pz);
            MPI_Abort(MPI_COMM_WORLD, 61);
        }
        if (Px * Py * Pz != P) {
            if (_rank == 0) fprintf(stderr, "Px*Py*Pz = %d != P = %d\n", Px * Py * Pz, P);
            MPI_Abort(MPI_COMM_WORLD, 62);
        }
        if (!block_cyclic_valid(N, b, b, Px, Py)) {
            if (_rank == 0) fprintf(stderr, "blockcyclic: need N%%b==0 and positive Px,Py\n");
            MPI_Abort(MPI_COMM_WORLD, 63);
        }
        if (total_elements_bc(N, b, b, Px, Py) != (std::int64_t)N * N) {
            if (_rank == 0) fprintf(stderr, "blockcyclic: total local elements != N^2\n");
            MPI_Abort(MPI_COMM_WORLD, 64);
        }
        if (A.strict_b || A.M_fp64_words > 0) {
            if (!b_in_window(b, Pz, N, Px, Py)) {
                if (_rank == 0) fprintf(stderr, "§A3b window violated: c=%d b=%d N=%d pr=%d pc=%d\n",
                                        Pz, b, N, Px, Py);
                MPI_Abort(MPI_COMM_WORLD, 8);
            }
        }
        int my_pz = _rank / (Px * Py);
        int my_px = (_rank / Py) % Px;
        int my_py = _rank % Py;
        run_block_cyclic_scqr3_main(A, P, Px, Py, my_px, my_py, my_pz);
        MPI_Finalize();
        return 0;
    }

    int m_loc = N / (Px * Pz);
    int n_loc = N / Py;
    if (N % (Px * Pz) != 0 || N % Py != 0) {
        if (_rank == 0) fprintf(stderr, "N=%d not divisible by Px*Pz=%d or Py=%d\n", N, Px*Pz, Py);
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    if (!layout_slab_total_matches_global(N, P, Px, Py, Pz)) {
        if (_rank == 0) fprintf(stderr, "layout_slab_total_matches_global failed N=%d P=%d grid [%d,%d,%d]\n",
                                N, P, Px, Py, Pz);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }
    if (A.M_fp64_words > 0 && A.px > 0 && A.py > 0 && A.pz > 0) {
        if (!derived_grid_matches_manual(P, N, A.M_fp64_words, Px, Py, Pz)) {
            DerivedSchedule Dv = compute_derived_schedule(P, N, A.M_fp64_words, false);
            if (_rank == 0) {
                fprintf(stderr, "[warn] manual grid [%d,%d,%d] differs from --M=%lld derived [%d,%d,%d]\n",
                        Px, Py, Pz, (long long)A.M_fp64_words, Dv.px, Dv.py, Dv.pz);
                fprintf(stderr, "       %s\n", format_derived_schedule(Dv).c_str());
            }
            if (A.strict_grid) MPI_Abort(MPI_COMM_WORLD, 9);
        }
    }
    if (A.strict_b || A.M_fp64_words > 0) {
        if (!b_in_window(b, Pz, N, Px, Py)) {
            if (_rank == 0) fprintf(stderr, "§A3b window violated: c=%d b=%d N=%d pr=%d pc=%d\n",
                                    Pz, b, N, Px, Py);
            MPI_Abort(MPI_COMM_WORLD, 8);
        }
    }

    // Decode (px, py, pz) from rank.  Linear ordering: rank = pz*Px*Py + px*Py + py.
    int my_pz = _rank / (Px * Py);
    int my_px = (_rank / Py) % Px;
    int my_py = _rank % Py;

    char tag[200];
    const char* matnm = "fp64";
    if (A.matrix == MatrixMode::FP64_MP) matnm = "fp64mp";
    else if (A.matrix == MatrixMode::FP64_MP_TF32) matnm = "fp64mp_tf32";
    else if (A.matrix == MatrixMode::FP32_FULL) matnm = "fp32full";
    std::snprintf(tag, sizeof(tag), "p25d_%s_passes=%d%s", matnm, A.passes, A.lookahead ? "+LA" : "");

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" Full 2.5D  N=%d b=%d  grid=[Px=%d, Py=%d, Pz=%d]  variant=%s\n",
               N, b, Px, Py, Pz, tag);
        printf("   m_loc=%d  n_loc=%d  P1=%d  c=Pz=%d\n", m_loc, n_loc, P1, Pz);
        printf("=================================================================\n");
        fflush(stdout);
    }

    // ── Sub-communicators + NCCL (shared factory: comm_groups.cuh) ─────────
    ColRowNccl cr{};
    create_col_row_nccl_full(MPI_COMM_WORLD, _rank, Px, Py, Pz, my_px, my_py, my_pz, &cr);
    MPI_Comm mpi_col_comm = cr.mpi_col_comm;
    MPI_Comm mpi_row_comm = cr.mpi_row_comm;
    ncclComm_t nccl_col = cr.nccl_col;
    ncclComm_t nccl_row = cr.nccl_row;
    int col_size, col_rank, row_size, row_rank;
    MPI_Comm_size(mpi_col_comm, &col_size); MPI_Comm_rank(mpi_col_comm, &col_rank);
    MPI_Comm_size(mpi_row_comm, &row_size); MPI_Comm_rank(mpi_row_comm, &row_rank);

    if (A.matrix == MatrixMode::FP32_FULL) {
        run_fp32_body(A, N, b, Px, Py, Pz, my_px, my_py, my_pz, m_loc, n_loc, col_rank, &cr);
        destroy_col_row_nccl(&cr);
        MPI_Finalize();
        return 0;
    }

    const bool use_mp_trail = nextla_is_mp_trail_matrix(A.matrix);
    const bool use_tf32_trail = nextla_requests_tf32_matrix(A.matrix) && (NEXTLA_HAVE_CUBLAS_TF32 != 0);
    if (nextla_requests_tf32_matrix(A.matrix) && !use_tf32_trail) {
        if (_rank == 0) {
            fprintf(stderr,
                    "scqr3_full25d: --matrix=fp64mp_tf32 requires a CUDA toolkit whose cuBLAS "
                    "defines CUBLAS_COMPUTE_32F_FAST_TF32 (CUDA 11+).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 91);
    }
    nextla_maybe_print_tf32_trailing_banner_rank0(_rank, use_tf32_trail);

    const bool eff_la = A.lookahead;  // fp32 path returned above

    // CUDA streams + cuBLAS / cuSolver handles.
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
    cublasHandle_t cublas_la = nullptr;
    cusolverDnHandle_t cusolver_la = nullptr;
    if (eff_la) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));       CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    // Local data A (m_loc x n_loc column-major) and A_orig for reset.
    double *d_A = nullptr, *d_A_orig = nullptr;
    {
        size_t nod = (size_t)m_loc * n_loc * sizeof(double);
        if (A.oom_probe) {
            cudaError_t em = cudaMalloc((void**)&d_A, nod);
            int ok = 1;
            if (em == cudaErrorMemoryAllocation) ok = 0;
            else if (em != cudaSuccess) {
                fprintf(stderr, "[r%d] CUDA %s @ %s:%d\n", _rank, cudaGetErrorString(em), __FILE__, __LINE__);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            int allok = 0;
            MPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (allok == 0) {
                if (_rank == 0) fprintf(stderr, "OOM_PROBE: cudaMalloc primary A failed on at least one rank (exit 77)\n");
                MPI_Finalize();
                return 77;
            }
        } else {
            CUDA_CHECK(cudaMalloc(&d_A, nod));
        }
    }
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_loc * n_loc * sizeof(double)));
    {
        std::vector<double> host((size_t)m_loc * n_loc);
        // Seed unique per rank for non-zero off-diag structure.
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A, (size_t)m_loc * n_loc * sizeof(double), cudaMemcpyDeviceToDevice));

    // Buffers.
    double* d_G       = nullptr;
    double* d_W       = nullptr;       // sb x n_loc local trailing W
    double* d_panel_bcast = nullptr;   // m_loc x b: this rank's slice of Q after broadcast
    double* d_trace   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G,           (size_t)b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,           (size_t)b * n_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel_bcast, (size_t)m_loc * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trace,       sizeof(double)));

    float* d_pf_Q = nullptr;
    float* d_pf_A = nullptr;
    float* d_pf_W = nullptr;
    if (use_mp_trail) {
        CUDA_CHECK(cudaMalloc(&d_pf_Q, (size_t)m_loc * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pf_A, (size_t)m_loc * n_loc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pf_W, (size_t)b * n_loc * sizeof(float)));
    }

    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Look-ahead second-panel scratch.
    double *d_G2=nullptr, *d_potrf_work2=nullptr, *d_panel_bcast2=nullptr;
    int *d_info2=nullptr;
    if (eff_la) {
        CUDA_CHECK(cudaMalloc(&d_G2,           (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_bcast2, (size_t)m_loc * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_potrf_work2,  potrf_lwork * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info2,        sizeof(int)));
    }

    const double one_d=1.0, zero_d=0.0, neg_one_d=-1.0;

    // ──────────────────────────────────────────────────────────────────────
    // Phase Q1 sCQR3 panel for column k of width sb on column-group py_panel.
    // Followed by broadcast of Q's row-slab from py_panel to all py'.
    //
    // Steps (only ranks with my_py == py_panel run 1-5):
    //   1. local SYRK G_loc = A_panel_loc^T * A_panel_loc          (b x b)
    //   2. AllReduce(G, col_group)                                  (Phase Q3)
    //   3. (first pass only) Fukaya shift
    //   4. POTRF(G) replicated within column group
    //   5. local TRSM A_panel_loc <- A_panel_loc * R^{-1}
    //  (iterations 1-5 looped passes-times)
    //   6. After last iter, A_panel_loc has orthonormal columns (= Q rows
    //      for this (px, pz) row block).  Pack into d_panel_bcast (NOOP if
    //      already contiguous) and broadcast across row group:
    //      ncclBroadcast(d_panel_bcast, root=py_panel, row_group).
    //   For ranks with my_py != py_panel: receive d_panel_bcast.
    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t cb, cusolverDnHandle_t cs,
                        double* Gb, double* work, int* info, double* p_bcast,
                        ncclComm_t nccl_col_h, ncclComm_t nccl_row_h,
                        int k, int sb, int passes) {
        int py_panel = k / n_loc;
        int local_k  = k - py_panel * n_loc;

        if (my_py == py_panel) {
            double coef = 11.0 * ((double)N * sb + (double)sb * (sb + 1)) * 2.220446049250313e-16;
            for (int it = 0; it < passes; ++it) {
                double* A_panel = d_A + (size_t)local_k * m_loc;   // m_loc x sb col-major
                CUBLAS_CHECK(cublasDsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                          sb, m_loc, &one_d, A_panel, m_loc,
                                          &zero_d, Gb, b));
                CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
                CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
                NCCL_CHECK(ncclAllReduce(Gb, Gb, (size_t)b * b, ncclDouble, ncclSum,
                                          nccl_col_h, s_comm));
                CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
                CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
                if (it == 0) {
                    trace_b_kernel<<<1, std::min(sb, 1024), 0, s_use>>>(Gb, b, sb, d_trace);
                    shift_diag_from_trace_kernel<<<(sb + 255) / 256, 256, 0, s_use>>>(Gb, b, sb, d_trace, coef);
                }
                CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_UPPER, sb, Gb, b, work, potrf_lwork, info));
                CUBLAS_CHECK(cublasDtrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                          m_loc, sb, &one_d, Gb, b, A_panel, m_loc));
            }
            // Copy Q from A's panel slot into the broadcast buffer.
            CUDA_CHECK(cudaMemcpyAsync(p_bcast, d_A + (size_t)local_k * m_loc,
                                       (size_t)m_loc * sb * sizeof(double),
                                       cudaMemcpyDeviceToDevice, s_use));
        }
        // Broadcast Q across row group (P_y ranks; root = py_panel within row group).
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclBroadcast(p_bcast, p_bcast, (size_t)m_loc * sb, ncclDouble,
                                  py_panel, nccl_row_h, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
    };

    // ──────────────────────────────────────────────────────────────────────
    // Phase Q2 trailing update on A's columns [k+sb, N).
    //   For each rank: local A_trail_loc = A's local cols intersected with
    //   [k+sb, N).  In our column-block layout, rank with my_py owns columns
    //   [my_py * n_loc, (my_py+1) * n_loc).  A_trail columns intersect this
    //   range as [max(k+sb, my_py*n_loc), (my_py+1)*n_loc).  In local
    //   coords: start = max(0, k+sb - my_py*n_loc), end = n_loc.  If start
    //   >= end, no trailing work on this rank for this panel.
    //
    //   1. local W_loc = Q_loc^T A_trail_loc  (sb x ncols_loc)
    //   2. AllReduce(W, col_group)                                  (Phase Q4)
    //   3. local A_trail_loc -= Q_loc * W_loc
    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb, ncclComm_t nccl_col_h,
                        double* p_bcast,
                        int k, int sb) {
        int my_col_start_global = my_py * n_loc;
        int trail_global_start  = k + sb;
        int local_start = std::max(0, trail_global_start - my_col_start_global);
        int local_end   = n_loc;
        int ncols = local_end - local_start;
        if (ncols <= 0) {
            return;
        }
        double* A_trail_loc = d_A + (size_t)local_start * m_loc;
        if (use_mp_trail) {
            const int nt = 256;
            const size_t nq = (size_t)m_loc * sb;
            cast_d2f_ker<<<((nq + nt - 1) / nt), nt, 0, s_use>>>(p_bcast, d_pf_Q, nq);
            const size_t na = (size_t)m_loc * ncols;
            cast_d2f_ker<<<((na + nt - 1) / nt), nt, 0, s_use>>>(A_trail_loc, d_pf_A, na);
            const float one_f = 1.f, zero_f = 0.f;
            if (use_tf32_trail) {
#if NEXTLA_HAVE_CUBLAS_TF32
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                          sb, ncols, m_loc,
                                          &one_f, d_pf_Q, CUDA_R_32F, m_loc,
                                          d_pf_A, CUDA_R_32F, m_loc,
                                          &zero_f, d_pf_W, CUDA_R_32F, b,
                                          CUBLAS_COMPUTE_32F_FAST_TF32,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                        sb, ncols, m_loc,
                                        &one_f, d_pf_Q, m_loc, d_pf_A, m_loc,
                                        &zero_f, d_pf_W, b));
#endif
            } else {
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                        sb, ncols, m_loc,
                                        &one_f, d_pf_Q, m_loc,
                                        d_pf_A, m_loc,
                                        &zero_f, d_pf_W, b));
            }
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_pf_W, d_pf_W, (size_t)sb * ncols, ncclFloat, ncclSum,
                                      nccl_col_h, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            const size_t nw = (size_t)sb * ncols;
            cast_f2d_ker<<<((nw + nt - 1) / nt), nt, 0, s_use>>>(d_pf_W, d_W, nw);
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_loc, ncols, sb,
                                      &neg_one_d, p_bcast, m_loc,
                                                  d_W, b,
                                      &one_d, A_trail_loc, m_loc));
        } else {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_loc,
                                      &one_d, p_bcast, m_loc,
                                              A_trail_loc, m_loc,
                                      &zero_d, d_W, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum,
                                      nccl_col_h, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_loc, ncols, sb,
                                      &neg_one_d, p_bcast, m_loc,
                                                  d_W, b,
                                      &one_d, A_trail_loc, m_loc));
        }
    };

    // Clip sb so the panel stays entirely within the column-block owned by one py.
    auto sb_clipped = [&](int k) {
        int py_panel = k / n_loc;
        int next_py_boundary = (py_panel + 1) * n_loc;
        int avail = std::min(N - k, next_py_boundary - k);
        return std::min(b, avail);
    };

    auto run_qr = [&]() {
        int k = 0;
        if (!eff_la) {
            while (k < N) {
                int sb = sb_clipped(k);
                phase_q1(s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, d_panel_bcast,
                         nccl_col, nccl_row, k, sb, A.passes);
                if (k + sb < N) phase_q2(s_comp, cublas, nccl_col, d_panel_bcast, k, sb);
                k += sb;
            }
        } else {
            int sb = sb_clipped(k);
            phase_q1(s_comp, cublas, cusolver, d_G, d_potrf_work, d_info, d_panel_bcast,
                     nccl_col, nccl_row, k, sb, A.passes);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? sb_clipped(next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest_total = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    // Q2 of next-panel columns on s_comp
                    // (we update A_trail in [next_k, next_k+next_sb) so next-panel SYRK
                    //  sees the post-trailing-update data).
                    // We split phase_q2 into two calls: cols [next_k, next_k+next_sb)
                    // on s_comp now; cols [next_k+next_sb, N) on s_comp concurrently
                    // with phase_q1(k+1) on s_la.
                    // Helper: phase_q2_range(s_use, cb, comm, p_bcast, k, sb, gstart, gend)
                    // For brevity, we reuse phase_q2 with a temporary N-truncation —
                    // but our phase_q2 always trails to N.  To keep this short, we
                    // emit the whole trailing update on s_comp before launching the
                    // look-ahead Q1 on s_la.  (Full overlap is left to future work.)
                    phase_q2(s_comp, cublas, nccl_col, d_panel_bcast, k, sb);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, cusolver_la, d_G2, d_potrf_work2, d_info2,
                             d_panel_bcast2, nccl_col, nccl_row, next_k, next_sb, A.passes);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                    CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                    // Swap panel buffers for the next iteration
                    std::swap(d_panel_bcast, d_panel_bcast2);
                    std::swap(d_G, d_G2);
                    std::swap(d_potrf_work, d_potrf_work2);
                    std::swap(d_info, d_info2);
                    k = next_k;
                    sb = next_sb;
                } else {
                    if (n_rest_total > 0) phase_q2(s_comp, cublas, nccl_col, d_panel_bcast, k, sb);
                    break;
                }
            }
        }
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, (size_t)m_loc * n_loc * sizeof(double), cudaMemcpyDeviceToDevice));
    };

    // Warmup
    for (int i = 0; i < 2; ++i) { reset_A(); run_qr(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    // Validation: max|diag(Q'Q) - 1| over the full N×N reconstruction.
    //   Each rank has its m_loc x n_loc block of Q.  Compute its contribution
    //   to G_full = Q^T Q via a SYRK over the LOCAL rows: g_block = (Q_loc)^T (Q_loc),
    //   which gives an n_loc x n_loc block for *this rank's columns* but only
    //   the contribution from this rank's row range.  AllReduce(g_block) over
    //   col_group(my_py) gives the per-column-group block.
    //   For the bench we just check the diagonals on my_py = 0 column group's
    //   first few entries — but to keep things uniform we compute the global
    //   max over all column groups via two AllReduces.
    {
        double* d_GG = nullptr;
        CUDA_CHECK(cudaMalloc(&d_GG, (size_t)n_loc * n_loc * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  n_loc, m_loc, &one_d, d_A, m_loc,
                                  &zero_d, d_GG, n_loc));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        NCCL_CHECK(ncclAllReduce(d_GG, d_GG, (size_t)n_loc * n_loc, ncclDouble, ncclSum,
                                  nccl_col, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        // Now each rank has the full n_loc x n_loc block of Q^T Q for its
        // column range.  Diagonal entries are |q_j|^2; we want max|diag - 1|.
        // Only the col_rank == 0 ranks (one per py-group) have unique data.
        double max_local = 0.0;
        if (col_rank == 0) {
            std::vector<double> diag(n_loc);
            for (int j = 0; j < n_loc; ++j) {
                CUDA_CHECK(cudaMemcpy(&diag[j], d_GG + j + (size_t)j * n_loc, sizeof(double), cudaMemcpyDeviceToHost));
                double dev = std::abs(diag[j] - 1.0);
                if (dev > max_local) max_local = dev;
            }
        }
        double max_global = 0.0;
        MPI_Allreduce(&max_local, &max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (_rank == 0) {
            printf("  Validation     variant=%s N=%d grid=[%d,%d,%d]  max|diag(Q'Q)-1| = %.2e\n",
                   tag, N, Px, Py, Pz, max_global);
            fflush(stdout);
        }
        cudaFree(d_GG);
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
        if (eff_la) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        printf("  %-30s  N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               tag, N, b, Px, Py, Pz, times[0], times[nrun/2]);
        double tmed = times[nrun / 2];
        const char* mcsv = (A.matrix == MatrixMode::FP64_MP)          ? "fp64mp"
                           : (A.matrix == MatrixMode::FP64_MP_TF32) ? "fp64mp_tf32"
                                                                     : "fp64";
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, Px * Py * Pz);
        printf("METRICS bench=scqr3_full25d matrix=%s layout=bc_2p5d_degenerate N=%d b=%d Px=%d Py=%d Pz=%d passes=%d ",
               mcsv, N, b, Px, Py, Pz, A.passes);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_A_orig); cudaFree(d_G); cudaFree(d_W); cudaFree(d_panel_bcast); cudaFree(d_trace);
    cudaFree(d_potrf_work); cudaFree(d_info);
    if (use_mp_trail) {
        cudaFree(d_pf_Q);
        cudaFree(d_pf_A);
        cudaFree(d_pf_W);
    }
    if (eff_la) {
        cudaFree(d_G2); cudaFree(d_potrf_work2); cudaFree(d_info2); cudaFree(d_panel_bcast2);
        cublasDestroy(cublas_la); cusolverDnDestroy(cusolver_la);
    }
    cublasDestroy(cublas); cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm); cudaStreamDestroy(s_la);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done); cudaEventDestroy(e_next_ready);
    destroy_col_row_nccl(&cr);
    MPI_Finalize();
    return 0;
}
