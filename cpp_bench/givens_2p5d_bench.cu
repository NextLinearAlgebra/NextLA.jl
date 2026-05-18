// Givens-rotation 2.5D QR benchmark — Path g of qr_schur_xpartition.tex.
//
//   Panel factorization: classical sequential Givens rotations applied
//   column-by-column to zero subdiagonal elements. The custom CUDA kernel
//   `givens_panel_kernel` runs as ONE BLOCK per panel and serializes the
//   (j, i) rotation loop while parallelising the trailing-column update
//   across threads. Rotations are saved as (c, s) pairs in row-major form
//   indexed by (column j, row i), giving b * (m - 1) entries per panel.
//
//   Explicit thin Q (m × b orthonormal) is reconstructed from the rotation
//   list by applying them to I_m[:, 0:b] in `form_Q_from_givens_kernel`.
//   The trailing update is then identical to Path h (Householder):
//     W = Q^T · A_trail  →  AllReduce  →  A_trail -= Q · W.
//
//   Same flag surface as scqr3_2p5d_variants.cu (Path s) and
//   householder_2p5d_bench.cu (Path h):
//     --matrix=fp64mp_tf32  mixed-precision trailing GEMMs via TF32 tensor ops (CUDA 11+)
//     --mp           trailing-update GEMMs in FP32  (Phase Q6 low-prec)
//     --ir=K         K rounds of Cholesky-QR refinement after factor
//                     (Phase Q6)
//     --la / --lookahead (default on, parity with scqr3_full25d) and --no-la / --no-lookahead (Phase Q5).
//     Slab parity (Path g vs s): same four --matrix= modes + LA flags as scqr3_full25d_bench.cu (passes=2|3 remain s-only).
//
//   Processor grid: default 1D row partition (m_local = N / c). Optional:
//     --layout=blockcyclic --px= --py= --pz=1  (P = Px·Py·Pz) for Path (g) block-cyclic tiles.

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

__global__ void cast_d2f(const double* d, float* f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)d[i];
}
__global__ void cast_f2d(const float* f, double* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)f[i];
}

__global__ void rearrange_recv_to_panel(const double* __restrict__ recv,
                                         double* __restrict__ full,
                                         int m_local, int sb, int P, int m_total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_total * sb;
    if (idx >= total) return;
    int i = (int)(idx % m_total);
    int j = (int)(idx / m_total);
    int r = i / m_local;
    int i_local = i - r * m_local;
    full[idx] = recv[(long long)r * m_local * sb + (long long)j * m_local + i_local];
}

__global__ void rearrange_recv_to_panel_f(const float* __restrict__ recv,
                                          float* __restrict__ full,
                                          int m_local, int sb, int P, int m_total) {
    (void)P;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_total * sb;
    if (idx >= total) return;
    int i = (int)(idx % m_total);
    int j = (int)(idx / m_total);
    int r = i / m_local;
    int i_local = i - r * m_local;
    full[idx] = recv[(long long)r * m_local * sb + (long long)j * m_local + i_local];
}

// Tournament-Givens parallel panel factorization (tex §A1-DAAP-givens
// "parallel-tournament").  Per column j, we eliminate the (m-j-1) subdiagonal
// entries through ⌈log2(m-j)⌉ levels of parallel pair rotations:
//   level l: stride = 2^l, pair p ∈ [0, num_pairs(j, l))
//     pair p rotates rows (j + p*2*stride, j + p*2*stride + stride) zeroing
//     A[i_bot, j].  All pairs at a given level are independent.
// Depth becomes Θ(b log m) instead of Θ(b m).  Storage for (c, s) keeps
// the same 2 * b * (m - 1) layout: per column j we use slot positions
// [j*(m-1) + 0, j*(m-1) + (m-j-1)) in the order we generate them.  The
// inverse kernel replays in reverse using identical indexing.
//
// Block size: blockDim.x is the parallel-pair width; the kernel launches as
// a single 1D block per panel.  No inter-block sync needed.
__device__ static inline int num_pairs_at_level(int active_m, int stride) {
    if (active_m <= stride) return 0;
    int double_stride = stride << 1;
    return (active_m - stride + double_stride - 1) / double_stride;
}

__global__ void givens_panel_kernel(double* __restrict__ A, int m, int b, int lda,
                                     double* __restrict__ cs_save) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    __shared__ int level_off[40];
    __shared__ int num_levels;

    for (int j = 0; j < b; ++j) {
        int active_m = m - j;
        long long slot_base = (long long)j * (m - 1);

        // Thread 0 precomputes the offset of each level within column j's slot range.
        if (tid == 0) {
            int cum = 0, l = 0;
            for (int stride = 1; stride < active_m; stride <<= 1) {
                level_off[l++] = cum;
                cum += num_pairs_at_level(active_m, stride);
            }
            num_levels = l;
        }
        __syncthreads();

        int nl = num_levels;
        for (int lvl = 0; lvl < nl; ++lvl) {
            int stride = 1 << lvl;
            int double_stride = stride << 1;
            int n_pairs = num_pairs_at_level(active_m, stride);
            int off = level_off[lvl];
            for (int p = tid; p < n_pairs; p += bs) {
                int i_top = j + p * double_stride;
                int i_bot = i_top + stride;
                // i_bot < m by construction of num_pairs_at_level.
                double a  = A[(long long)i_top + (long long)j * lda];
                double bv = A[(long long)i_bot + (long long)j * lda];
                double r  = hypot(a, bv);
                double c  = (r != 0.0) ? a / r  : 1.0;
                double s  = (r != 0.0) ? bv / r : 0.0;
                A[(long long)i_top + (long long)j * lda] = r;
                A[(long long)i_bot + (long long)j * lda] = 0.0;
                for (int kc = j + 1; kc < b; ++kc) {
                    double at = A[(long long)i_top + (long long)kc * lda];
                    double ab = A[(long long)i_bot + (long long)kc * lda];
                    A[(long long)i_top + (long long)kc * lda] =  c * at + s * ab;
                    A[(long long)i_bot + (long long)kc * lda] = -s * at + c * ab;
                }
                long long cs_idx = 2 * (slot_base + off + p);
                cs_save[cs_idx]     = c;
                cs_save[cs_idx + 1] = s;
            }
            __syncthreads();
        }
    }
}

// Form explicit thin Q (m × b orthonormal columns) by applying the saved
// Givens rotations to the first b columns of I_m, in reverse order.
//   - Q is allocated as m × b column-major (ld = m).
//   - We first zero-fill Q, then write 1's on the diagonal Q[k, k] = 1 for k = 0..b-1.
//   - Then we apply rotations in reverse order: for j from b-1 down to 0, for i
//     from j+1 to m-1 (reverse of the panel direction).  Each rotation acts on
//     rows (i-1, i) of Q with the same (c, s) pair (no transpose: we are
//     reconstructing Q such that Q^T A_orig is upper-triangular).
__global__ void init_thinQ_identity(double* Q, int m, int b) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m * b;
    if (idx >= total) return;
    int i = (int)(idx % m);
    int j = (int)(idx / m);
    Q[idx] = (i == j) ? 1.0 : 0.0;
}
// Tournament inverse: replay the rotations in REVERSE time order.  For
// each column j (from b-1 down to 0), iterate levels from highest stride
// down to 1, then within a level the pairs (level pairs are independent
// in both directions).
__global__ void form_Q_from_givens_kernel(double* __restrict__ Q, int m, int b, int ldq,
                                          const double* __restrict__ cs_save) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    __shared__ int level_off[40];
    __shared__ int num_levels;
    for (int j = b - 1; j >= 0; --j) {
        int active_m = m - j;
        long long slot_base = (long long)j * (m - 1);

        if (tid == 0) {
            int cum = 0, l = 0;
            for (int stride = 1; stride < active_m; stride <<= 1) {
                level_off[l++] = cum;
                cum += num_pairs_at_level(active_m, stride);
            }
            num_levels = l;
        }
        __syncthreads();

        int nl = num_levels;
        for (int lvl = nl - 1; lvl >= 0; --lvl) {
            int stride = 1 << lvl;
            int double_stride = stride << 1;
            int n_pairs = num_pairs_at_level(active_m, stride);
            int off = level_off[lvl];
            for (int p = tid; p < n_pairs; p += bs) {
                int i_top = j + p * double_stride;
                int i_bot = i_top + stride;
                long long cs_idx = 2 * (slot_base + off + p);
                double c = cs_save[cs_idx];
                double s = cs_save[cs_idx + 1];
                // Forward-apply rotation G(i_top, i_bot; c, s) to Q's rows.
                for (int kc = 0; kc < b; ++kc) {
                    double at = Q[(long long)i_top + (long long)kc * ldq];
                    double ab = Q[(long long)i_bot + (long long)kc * ldq];
                    Q[(long long)i_top + (long long)kc * ldq] =  c * at + s * ab;
                    Q[(long long)i_bot + (long long)kc * ldq] = -s * at + c * ab;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void givens_panel_kernel_f(float* __restrict__ A, int m, int b, int lda,
                                      float* __restrict__ cs_save) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    __shared__ int level_off[40];
    __shared__ int num_levels;

    for (int j = 0; j < b; ++j) {
        int active_m = m - j;
        long long slot_base = (long long)j * (m - 1);
        if (tid == 0) {
            int cum = 0, l = 0;
            for (int stride = 1; stride < active_m; stride <<= 1) {
                level_off[l++] = cum;
                cum += num_pairs_at_level(active_m, stride);
            }
            num_levels = l;
        }
        __syncthreads();
        int nl = num_levels;
        for (int lvl = 0; lvl < nl; ++lvl) {
            int stride = 1 << lvl;
            int double_stride = stride << 1;
            int n_pairs = num_pairs_at_level(active_m, stride);
            int off = level_off[lvl];
            for (int p = tid; p < n_pairs; p += bs) {
                int i_top = j + p * double_stride;
                int i_bot = i_top + stride;
                float a  = A[(long long)i_top + (long long)j * lda];
                float bv = A[(long long)i_bot + (long long)j * lda];
                float r  = hypotf(a, bv);
                float c  = (r != 0.f) ? a / r  : 1.f;
                float s  = (r != 0.f) ? bv / r : 0.f;
                A[(long long)i_top + (long long)j * lda] = r;
                A[(long long)i_bot + (long long)j * lda] = 0.f;
                for (int kc = j + 1; kc < b; ++kc) {
                    float at = A[(long long)i_top + (long long)kc * lda];
                    float ab = A[(long long)i_bot + (long long)kc * lda];
                    A[(long long)i_top + (long long)kc * lda] =  c * at + s * ab;
                    A[(long long)i_bot + (long long)kc * lda] = -s * at + c * ab;
                }
                long long cs_idx = 2 * (slot_base + off + p);
                cs_save[cs_idx]     = c;
                cs_save[cs_idx + 1] = s;
            }
            __syncthreads();
        }
    }
}

__global__ void init_thinQ_identity_f(float* Q, int m, int b) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m * b;
    if (idx >= total) return;
    int i = (int)(idx % m);
    int j = (int)(idx / m);
    Q[idx] = (i == j) ? 1.f : 0.f;
}

__global__ void form_Q_from_givens_kernel_f(float* __restrict__ Q, int m, int b, int ldq,
                                            const float* __restrict__ cs_save) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    __shared__ int level_off[40];
    __shared__ int num_levels;
    for (int j = b - 1; j >= 0; --j) {
        int active_m = m - j;
        long long slot_base = (long long)j * (m - 1);
        if (tid == 0) {
            int cum = 0, l = 0;
            for (int stride = 1; stride < active_m; stride <<= 1) {
                level_off[l++] = cum;
                cum += num_pairs_at_level(active_m, stride);
            }
            num_levels = l;
        }
        __syncthreads();
        int nl = num_levels;
        for (int lvl = nl - 1; lvl >= 0; --lvl) {
            int stride = 1 << lvl;
            int double_stride = stride << 1;
            int n_pairs = num_pairs_at_level(active_m, stride);
            int off = level_off[lvl];
            for (int p = tid; p < n_pairs; p += bs) {
                int i_top = j + p * double_stride;
                int i_bot = i_top + stride;
                long long cs_idx = 2 * (slot_base + off + p);
                float c = cs_save[cs_idx];
                float s = cs_save[cs_idx + 1];
                for (int kc = 0; kc < b; ++kc) {
                    float at = Q[(long long)i_top + (long long)kc * ldq];
                    float ab = Q[(long long)i_bot + (long long)kc * ldq];
                    Q[(long long)i_top + (long long)kc * ldq] =  c * at + s * ab;
                    Q[(long long)i_bot + (long long)kc * ldq] = -s * at + c * ab;
                }
            }
            __syncthreads();
        }
    }
}

struct Args {
    int N = 16000, b = 0, n_ir = 0;
    bool mp = false, lookahead = true;
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
        if      (s.rfind("--N=", 0) == 0) a.N = std::atoi(s.c_str() + 4);
        else if (s.rfind("--b=", 0) == 0) a.b = std::atoi(s.c_str() + 4);
        else if (s.rfind("--ir=", 0) == 0) a.n_ir = std::atoi(s.c_str() + 5);
        else if (s == "--mp") a.mp = true;
        else if (s.rfind("--matrix=", 0) == 0) a.matrix = parse_matrix_mode(s.c_str() + 9);
        else if (s.rfind("--M=", 0) == 0) a.M_fp64_words = std::atoll(s.c_str() + 4);
        else if (s == "--strict-b") a.strict_b = true;
        else if (s.rfind("--layout=", 0) == 0) {
            const char* v = s.c_str() + 9;
            if (std::strcmp(v, "blockcyclic") == 0) a.block_cyclic_layout = true;
            else if (std::strcmp(v, "slab") == 0) a.block_cyclic_layout = false;
        } else if (s.rfind("--px=", 0) == 0) a.px = std::atoi(s.c_str() + 5);
        else if (s.rfind("--py=", 0) == 0) a.py = std::atoi(s.c_str() + 5);
        else if (s.rfind("--pz=", 0) == 0) a.pz = std::atoi(s.c_str() + 5);
        else if (s == "--lookahead" || s == "--la") {
            a.lookahead = true;
            a.lookahead_cli_set = true;
        } else if (s == "--no-la" || s == "--no-lookahead") {
            a.lookahead = false;
            a.lookahead_cli_set = true;
        }
    }
    if (a.mp && a.matrix == MatrixMode::FP64) a.matrix = MatrixMode::FP64_MP;
    return a;
}

// FP32-full Path (g): float Givens panel + explicit Q + float trailing; supports --la and --ir.
static int givens_fp32full_run(
    Args A, int c, ncclComm_t nccl_comm,
    cudaStream_t s_comp, cudaStream_t s_comm, cudaStream_t s_la,
    cudaEvent_t e_comp_done, cudaEvent_t e_ar_done,
    cudaEvent_t e_panel_done, cudaEvent_t e_next_ready,
    cublasHandle_t cublas, cusolverDnHandle_t cusolver) {
    int N = A.N, b = A.b;
    int m_local = N / c;

    float* d_A_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_local, (size_t)m_local * N * sizeof(float)));
    {
        std::vector<float> host(m_local * (size_t)N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<float> nrm(0.f, 1.f);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A_local, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    float* d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_local * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A_local, (size_t)m_local * N * sizeof(float), cudaMemcpyDeviceToDevice));

    float *d_panel_recv = nullptr, *d_panel_full = nullptr, *d_Q_full = nullptr;
    float *d_cs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel_recv, (size_t)m_local * b * (size_t)c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_panel_full, (size_t)N * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_full, (size_t)N * b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cs, (size_t)2 * b * (size_t)(N - 1) * sizeof(float)));

    float* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * N * sizeof(float)));

    float* d_G_ref = nullptr;
    float* d_potrf_work_ref = nullptr;
    int potrf_lwork_ref = 0;
    if (A.n_ir > 0) {
        CUDA_CHECK(cudaMalloc(&d_G_ref, (size_t)b * b * sizeof(float)));
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G_ref, b, &potrf_lwork_ref));
        CUDA_CHECK(cudaMalloc(&d_potrf_work_ref, (size_t)potrf_lwork_ref * sizeof(float)));
    }

    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    float *d_panel_recv2 = nullptr, *d_panel_full2 = nullptr, *d_Q_full2 = nullptr, *d_cs2 = nullptr;
    cublasHandle_t cublas_la{};
    cusolverDnHandle_t cusolver_la{};
    if (A.lookahead) {
        CUDA_CHECK(cudaMalloc(&d_panel_recv2, (size_t)m_local * b * (size_t)c * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_panel_full2, (size_t)N * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q_full2, (size_t)N * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cs2, (size_t)2 * b * (size_t)(N - 1) * sizeof(float)));
        CUBLAS_CHECK(cublasCreate(&cublas_la));
        CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;

    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t /*cb*/, cusolverDnHandle_t /*cs*/,
                        float* p_recv, float* p_full, float* p_Q, float* cs_buf,
                        int k, int sb) {
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllGather(d_A_local + (size_t)k * m_local,
                                 p_recv,
                                 (size_t)m_local * sb, ncclFloat,
                                 nccl_comm, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));

        {
            long long total = (long long)N * sb;
            int threads = 256;
            long long blocks = (total + threads - 1) / threads;
            rearrange_recv_to_panel_f<<<(unsigned)blocks, threads, 0, s_use>>>(p_recv, p_full, m_local, sb, c, N);
        }

        givens_panel_kernel_f<<<1, 256, 0, s_use>>>(p_full, N, sb, N, cs_buf);

        {
            long long total = (long long)N * sb;
            int threads = 256;
            long long blocks = (total + threads - 1) / threads;
            init_thinQ_identity_f<<<(unsigned)blocks, threads, 0, s_use>>>(p_Q, N, sb);
            form_Q_from_givens_kernel_f<<<1, 256, 0, s_use>>>(p_Q, N, sb, N, cs_buf);
        }

        CUDA_CHECK(cudaMemcpy2DAsync(d_A_local + (size_t)k * m_local, (size_t)m_local * sizeof(float),
                                     p_Q + (size_t)_rank * m_local, (size_t)N * sizeof(float),
                                     (size_t)m_local * sizeof(float), (size_t)sb,
                                     cudaMemcpyDeviceToDevice, s_use));
    };

    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb,
                        int k, int sb, int col_start, int ncols) {
        float* d_A_panel = d_A_local + (size_t)k * m_local;
        float* d_A_tr = d_A_local + (size_t)col_start * m_local;
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                 sb, ncols, m_local,
                                 &one_f, d_A_panel, m_local, d_A_tr, m_local,
                                 &zero_f, d_W, b));
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_comm, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m_local, ncols, sb,
                                 &neg_one_f, d_A_panel, m_local, d_W, b,
                                 &one_f, d_A_tr, m_local));
    };

    auto refine_one = [&]() {
        if (A.n_ir <= 0) return;
        int k = 0;
        while (k < N) {
            int sb = std::min(b, N - k);
            float* d_A_panel = d_A_local + (size_t)k * m_local;
            CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     sb, m_local, &one_f, d_A_panel, m_local,
                                     &zero_f, d_G_ref, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_G_ref, d_G_ref, (size_t)b * b, ncclFloat, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_ar_done, 0));
            CUSOLVER_CHECK(cusolverDnSpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G_ref, b, d_potrf_work_ref,
                                            potrf_lwork_ref, d_info));
            CUBLAS_CHECK(cublasStrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                     m_local, sb, &one_f, d_G_ref, b, d_A_panel, m_local));
            k += sb;
        }
    };

    auto run_qr = [&]() {
        int k = 0;
        if (!A.lookahead) {
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                phase_q1(s_comp, cublas, cusolver,
                         d_panel_recv, d_panel_full, d_Q_full, d_cs, k, sb);
                if (n_tr > 0) phase_q2(s_comp, cublas, k, sb, k + sb, n_tr);
                k += sb;
            }
        } else {
            int sb = std::min(b, N - k);
            phase_q1(s_comp, cublas, cusolver,
                     d_panel_recv, d_panel_full, d_Q_full, d_cs, k, sb);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? std::min(b, N - next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    phase_q2(s_comp, cublas, k, sb, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, cusolver_la,
                             d_panel_recv2, d_panel_full2, d_Q_full2, d_cs2,
                             next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                }
                if (n_rest > 0) phase_q2(s_comp, cublas, k, sb, rest_start, n_rest);
                CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                k = next_k;
                sb = next_sb;
                if (sb == 0) break;
            }
        }
        for (int j = 0; j < A.n_ir; ++j) refine_one();
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A_local, d_A_orig, (size_t)m_local * N * sizeof(float), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) {
        reset_A();
        run_qr();
    }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

    {
        float* d_QtQ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_QtQ, (size_t)N * N * sizeof(float)));
        CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                 N, m_local, &one_f, d_A_local, m_local,
                                 &zero_f, d_QtQ, N));
        CUDA_CHECK(cudaStreamSynchronize(s_comp));
        NCCL_CHECK(ncclAllReduce(d_QtQ, d_QtQ, (size_t)N * N, ncclFloat, ncclSum, nccl_comm, s_comm));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        if (_rank == 0) {
            float max_dev = 0.f;
            for (int j = 0; j < N; ++j) {
                float dj;
                CUDA_CHECK(cudaMemcpy(&dj, d_QtQ + j + (size_t)j * N, sizeof(float), cudaMemcpyDeviceToHost));
                max_dev = std::max(max_dev, std::fabs(dj - 1.f));
            }
            printf("  Validation(fp32) variant=givens_fp32full N=%d c=%d  max|diag(Q'Q)-1| = %.2e\n",
                   N, c, (double)max_dev);
            fflush(stdout);
        }
        cudaFree(d_QtQ);
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
        if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  %-30s  N=%d b=%d c=%d  tmin=%9.2f ms  tmed=%9.2f ms\n",
               "givens_fp32full", N, b, c, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, c);
        printf("METRICS bench=givens_2p5d matrix=fp32full N=%d b=%d c=%d passes=1 ", N, b, c);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    cudaFree(d_A_local);
    cudaFree(d_A_orig);
    cudaFree(d_panel_recv);
    cudaFree(d_panel_full);
    cudaFree(d_Q_full);
    cudaFree(d_cs);
    cudaFree(d_W);
    cudaFree(d_info);
    if (A.n_ir > 0) {
        cudaFree(d_G_ref);
        cudaFree(d_potrf_work_ref);
    }
    if (A.lookahead) {
        cudaFree(d_panel_recv2);
        cudaFree(d_panel_full2);
        cudaFree(d_Q_full2);
        cudaFree(d_cs2);
        cublasDestroy(cublas_la);
        cusolverDnDestroy(cusolver_la);
    }
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp);
    cudaStreamDestroy(s_comm);
    cudaStreamDestroy(s_la);
    cudaEventDestroy(e_comp_done);
    cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done);
    cudaEventDestroy(e_next_ready);
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();
    return 0;
}

#include "givens_block_cyclic.inl"
#include "givens_full25d.inl"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int c;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c);

    Args A = parse_args(argc, argv);
    const int P = c;
    int ngpu_g = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu_g));
    if (ngpu_g <= 0) {
        if (_rank == 0) fprintf(stderr, "givens: no CUDA devices\n");
        MPI_Abort(MPI_COMM_WORLD, 99);
    }
    const int bench_dev_g = _rank % ngpu_g;
    CUDA_CHECK(cudaSetDevice(bench_dev_g));
    if (A.M_fp64_words <= 0) {
        A.M_fp64_words = nextla_device_fast_memory_budget_elements(bench_dev_g, A.matrix);
        if (_rank == 0) {
            fprintf(stdout, "auto M (TeX fast memory): %lld matrix elements (σ=%zu B)\n",
                    (long long)A.M_fp64_words, nextla_matrix_element_bytes(A.matrix));
            fflush(stdout);
        }
    }
    if (A.block_cyclic_layout && A.matrix == MatrixMode::FP32_FULL && A.lookahead && !A.lookahead_cli_set) {
        if (_rank == 0) {
            fprintf(stdout,
                    "givens: fp32full block-cyclic has no inner lookahead wiring; disabling LA for this run (pass "
                    "--no-la to silence).\n");
            fflush(stdout);
        }
        A.lookahead = false;
    }
    if (A.block_cyclic_layout) {
        if (A.px <= 0 || A.py <= 0 || A.pz <= 0) {
            if (_rank == 0)
                fprintf(stderr,
                        "givens: --layout=blockcyclic requires --px= --py= --pz= (Pz=1; P=Px*Py*Pz)\n");
            MPI_Abort(MPI_COMM_WORLD, 60);
        }
        if (A.pz != 1) {
            if (_rank == 0) fprintf(stderr, "givens: blockcyclic requires Pz=1\n");
            MPI_Abort(MPI_COMM_WORLD, 61);
        }
        if (P != A.px * A.py * A.pz) {
            if (_rank == 0)
                fprintf(stderr, "givens: MPI size P=%d != Px*Py*Pz=%d*%d*%d\n", P, A.px, A.py, A.pz);
            MPI_Abort(MPI_COMM_WORLD, 62);
        }
        if (A.b == 0 && A.M_fp64_words > 0) {
            int bb = default_block_b(A.M_fp64_words, (std::int64_t)A.N, A.px, A.py, A.pz);
            if (bb > 0) A.b = bb;
        }
        if (A.b == 0) {
            if (A.N <= 4000) A.b = 64;
            else if (A.N <= 8000) A.b = 96;
            else if (A.N <= 16000) A.b = 128;
            else if (A.N <= 32000) A.b = 192;
            else if (A.N <= 48000) A.b = 256;
            else A.b = 256;
        }
        int my_pz = _rank / (A.px * A.py);
        int my_px = (_rank / A.py) % A.px;
        int my_py = _rank % A.py;
        int rc = run_givens_bc_main(A, P, A.px, A.py, my_px, my_py, my_pz);
        MPI_Finalize();
        return rc;
    }

    // ── Full-2.5D Pz>1 dispatch (tournament-Givens panel; all 4 matrix modes) ──
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
                if      (A.N <=  4000) A.b = 64;
                else if (A.N <=  8000) A.b = 96;
                else if (A.N <= 16000) A.b = 128;
                else if (A.N <= 32000) A.b = 192;
                else                   A.b = 256;
            }
            A.px = G_pre.Px; A.py = G_pre.Py; A.pz = G_pre.Pz;
            Full25DGrid G = resolve_full25d_grid(P, _rank, A.N, A.px, A.py, A.pz, A.M_fp64_words);
            Full25DSubcomms S = build_full25d_subcomms(G);
            const bool use_mp_trail   = nextla_is_mp_trail_matrix(A.matrix);
            const bool use_tf32_trail = nextla_requests_tf32_matrix(A.matrix);
#if !defined(CUBLAS_COMPUTE_32F_FAST_TF32)
            if (use_tf32_trail) {
                if (_rank == 0) fprintf(stderr,
                    "givens_full25d: --matrix=fp64mp_tf32 requires CUDA 11+ cuBLAS TF32.\n");
                MPI_Abort(MPI_COMM_WORLD, 91);
            }
#endif
            if (_rank == 0) {
                const char* tag = (A.matrix == MatrixMode::FP32_FULL) ? "givens_full25d/fp32full"
                                 : (A.matrix == MatrixMode::FP64_MP_TF32) ? "givens_full25d/fp64mp_tf32"
                                 : (A.matrix == MatrixMode::FP64_MP) ? "givens_full25d/fp64mp"
                                 : "givens_full25d/fp64";
                print_full25d_grid(G, tag, A.M_fp64_words, A.b);
            }
            int rc;
            if (A.matrix == MatrixMode::FP32_FULL) {
                rc = run_givens_full25d_fp32(A, G, S);
            } else {
                rc = run_givens_full25d_fp64(A, G, S, use_mp_trail, use_tf32_trail);
            }
            destroy_full25d_subcomms(S);
            MPI_Finalize();
            return rc;
        }
    }

    if (A.N % c != 0) { if (_rank==0) fprintf(stderr,"N=%d not divisible by c=%d\n",A.N,c); MPI_Abort(MPI_COMM_WORLD,5); }
    if (A.b == 0 && A.M_fp64_words > 0) {
        int bb = default_block_b(A.M_fp64_words, (std::int64_t)A.N, 1, 1, P);
        if (bb > 0) A.b = bb;
    }
    if (A.b == 0) {
        if (A.N <= 4000) A.b = 64;
        else if (A.N <= 8000) A.b = 96;
        else if (A.N <= 16000) A.b = 128;
        else if (A.N <= 32000) A.b = 192;
        else if (A.N <= 48000) A.b = 256;
        else A.b = 256;
    }
    int N = A.N, b = A.b, m_local = N / c;
    const bool use_mp =
        nextla_is_mp_trail_matrix(A.matrix) || A.mp;
    const bool use_tf32_trail =
        nextla_requests_tf32_matrix(A.matrix) && (NEXTLA_HAVE_CUBLAS_TF32 != 0);
    if (nextla_requests_tf32_matrix(A.matrix) && !use_tf32_trail) {
        if (_rank == 0) {
            fprintf(stderr,
                    "givens: --matrix=fp64mp_tf32 requires CUDA 11+ cuBLAS (CUBLAS_COMPUTE_32F_FAST_TF32).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 91);
    }
    nextla_maybe_print_tf32_trailing_banner_rank0(_rank, use_tf32_trail);
    if (A.M_fp64_words > 0 && _rank == 0) {
        DerivedSchedule D1 = compute_degenerate_1d_schedule(c, N, A.M_fp64_words);
        fprintf(stdout, "%s\n", format_derived_schedule(D1).c_str());
        fflush(stdout);
    }
    if (A.strict_b && !b_in_window(b, c, N, 1, 1)) {
        if (_rank == 0) fprintf(stderr, "givens: b=%d violates §A3b window for c=%d, N=%d\n", b, c, N);
        MPI_Abort(MPI_COMM_WORLD, 55);
    }

    char tag[160];
    std::snprintf(tag, sizeof(tag), "givens_%s%s%s%s%s",
                  matrix_mode_tag(A.matrix),
                  A.mp ? "+MP" : "",
                  A.lookahead ? "+LA" : "",
                  A.n_ir > 0 ? "+IR" : "",
                  A.n_ir > 0 ? std::to_string(A.n_ir).c_str() : "");

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" Givens 2.5D       N=%d b=%d c=%d   variant=%s\n", N, b, c, tag);
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

    if (A.matrix == MatrixMode::FP32_FULL) {
        return givens_fp32full_run(A, c, nccl_comm, s_comp, s_comm, s_la,
                                   e_comp_done, e_ar_done, e_panel_done, e_next_ready,
                                   cublas, cusolver);
    }

    // Distributed A.
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

    // Replicated panel buffers (m × b).
    double *d_panel_recv = nullptr, *d_panel_full = nullptr, *d_Q_full = nullptr;
    double *d_cs = nullptr;        // 2 * b * (m-1) doubles for Givens (c, s)
    CUDA_CHECK(cudaMalloc(&d_panel_recv, (size_t)m_local * b * (size_t)c * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel_full, (size_t)N * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q_full,     (size_t)N * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cs,         (size_t)2 * b * (size_t)(N - 1) * sizeof(double)));

    double* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * N * sizeof(double)));

    double* d_G_ref = nullptr; double* d_potrf_work_ref = nullptr; int potrf_lwork_ref = 0;
    if (A.n_ir > 0) {
        CUDA_CHECK(cudaMalloc(&d_G_ref, (size_t)b * b * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G_ref, b, &potrf_lwork_ref));
        CUDA_CHECK(cudaMalloc(&d_potrf_work_ref, potrf_lwork_ref * sizeof(double)));
    }
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // MP scratch.
    float *d_A_panel_f=nullptr, *d_A_tr_f=nullptr, *d_W_f=nullptr;
    if (use_mp) {
        CUDA_CHECK(cudaMalloc(&d_A_panel_f, (size_t)m_local * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_tr_f,    (size_t)m_local * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_f,       (size_t)b * N * sizeof(float)));
    }

    // Look-ahead scratch.
    double *d_panel_recv2=nullptr, *d_panel_full2=nullptr, *d_Q_full2=nullptr, *d_cs2=nullptr;
    cublasHandle_t cublas_la;
    cusolverDnHandle_t cusolver_la;
    if (A.lookahead) {
        CUDA_CHECK(cudaMalloc(&d_panel_recv2, (size_t)m_local * b * (size_t)c * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_full2, (size_t)N * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Q_full2,     (size_t)N * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cs2,         (size_t)2 * b * (size_t)(N - 1) * sizeof(double)));
        CUBLAS_CHECK(cublasCreate(&cublas_la));       CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    const double one_d=1.0, zero_d=0.0, neg_one_d=-1.0;
    const float  one_f=1.0f, zero_f=0.0f, neg_one_f=-1.0f;

    // Phase Q1: AllGather → rearrange → Givens panel → form Q → memcpy2D back.
    auto phase_q1 = [&](cudaStream_t s_use, cublasHandle_t /*cb*/, cusolverDnHandle_t /*cs*/,
                        double* p_recv, double* p_full, double* p_Q, double* cs_buf,
                        int k, int sb) {
        // 1) AllGather panel rows.
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllGather(d_A_local + (size_t)k * m_local,
                                 p_recv,
                                 (size_t)m_local * sb, ncclDouble,
                                 nccl_comm, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));

        // 2) Rearrange rank-block → column-major m × sb panel.
        {
            long long total = (long long)N * sb;
            int threads = 256;
            long long blocks = (total + threads - 1) / threads;
            rearrange_recv_to_panel<<<(unsigned)blocks, threads, 0, s_use>>>(p_recv, p_full, m_local, sb, c, N);
        }

        // 3) Givens panel factorization (one-block kernel; replicated).
        givens_panel_kernel<<<1, 256, 0, s_use>>>(p_full, N, sb, N, cs_buf);

        // 4) Form thin Q from rotation list.
        {
            long long total = (long long)N * sb;
            int threads = 256;
            long long blocks = (total + threads - 1) / threads;
            init_thinQ_identity<<<(unsigned)blocks, threads, 0, s_use>>>(p_Q, N, sb);
            form_Q_from_givens_kernel<<<1, 256, 0, s_use>>>(p_Q, N, sb, N, cs_buf);
        }

        // 5) memcpy2D rank's Q rows back into A_local[:, k:k+sb].
        CUDA_CHECK(cudaMemcpy2DAsync(d_A_local + (size_t)k * m_local, (size_t)m_local * sizeof(double),
                                     p_Q + (size_t)_rank * m_local, (size_t)N * sizeof(double),
                                     (size_t)m_local * sizeof(double), (size_t)sb,
                                     cudaMemcpyDeviceToDevice, s_use));
    };

    // Phase Q2 trailing update — identical to other variants.
    auto phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb,
                        int k, int sb, int col_start, int ncols, bool mp) {
        double* d_A_panel = d_A_local + (size_t)k * m_local;
        double* d_A_tr    = d_A_local + (size_t)col_start * m_local;
        if (!mp) {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_local,
                                      &one_d, d_A_panel, m_local, d_A_tr, m_local,
                                      &zero_d, d_W, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_local, ncols, sb,
                                      &neg_one_d, d_A_panel, m_local, d_W, b,
                                      &one_d, d_A_tr, m_local));
        } else {
            size_t np = (size_t)m_local * sb;
            size_t nt = (size_t)m_local * ncols;
            cast_d2f<<<(np + 255)/256, 256, 0, s_use>>>(d_A_panel, d_A_panel_f, np);
            cast_d2f<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_tr,    d_A_tr_f,    nt);
            if (use_tf32_trail) {
#if NEXTLA_HAVE_CUBLAS_TF32
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                          sb, ncols, m_local,
                                          &one_f, d_A_panel_f, CUDA_R_32F, m_local,
                                          d_A_tr_f, CUDA_R_32F, m_local,
                                          &zero_f, d_W_f, CUDA_R_32F, b,
                                          CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                         sb, ncols, m_local,
                                         &one_f, d_A_panel_f, m_local, d_A_tr_f, m_local,
                                         &zero_f, d_W_f, b));
#endif
            } else {
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                         sb, ncols, m_local,
                                         &one_f, d_A_panel_f, m_local,
                                         d_A_tr_f, m_local,
                                         &zero_f, d_W_f, b));
            }
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W_f, d_W_f, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            if (use_tf32_trail) {
#if NEXTLA_HAVE_CUBLAS_TF32
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                          m_local, ncols, sb,
                                          &neg_one_f, d_A_panel_f, CUDA_R_32F, m_local,
                                          d_W_f, CUDA_R_32F, b,
                                          &one_f, d_A_tr_f, CUDA_R_32F, m_local,
                                          CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m_local, ncols, sb,
                                         &neg_one_f, d_A_panel_f, m_local,
                                                     d_W_f, b,
                                         &one_f, d_A_tr_f, m_local));
#endif
            } else {
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m_local, ncols, sb,
                                         &neg_one_f, d_A_panel_f, m_local,
                                                     d_W_f, b,
                                         &one_f, d_A_tr_f, m_local));
            }
            cast_f2d<<<(nt + 255)/256, 256, 0, s_use>>>(d_A_tr_f, d_A_tr, nt);
        }
    };

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
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                phase_q1(s_comp, cublas, cusolver,
                         d_panel_recv, d_panel_full, d_Q_full, d_cs, k, sb);
                if (n_tr > 0) phase_q2(s_comp, cublas, k, sb, k + sb, n_tr, use_mp);
                k += sb;
            }
        } else {
            int sb = std::min(b, N - k);
            phase_q1(s_comp, cublas, cusolver,
                     d_panel_recv, d_panel_full, d_Q_full, d_cs, k, sb);
            CUDA_CHECK(cudaEventRecord(e_panel_done, s_comp));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? std::min(b, N - next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    phase_q2(s_comp, cublas, k, sb, next_k, next_sb, use_mp);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, cusolver_la,
                             d_panel_recv2, d_panel_full2, d_Q_full2, d_cs2,
                             next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                }
                if (n_rest > 0) phase_q2(s_comp, cublas, k, sb, rest_start, n_rest, use_mp);
                CUDA_CHECK(cudaStreamWaitEvent(s_comp, e_panel_done, 0));
                k = next_k;
                sb = next_sb;
                if (sb == 0) break;
            }
        }
        for (int j = 0; j < A.n_ir; ++j) refine_one();
    };

    auto reset_A = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A_local, d_A_orig, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) { reset_A(); run_qr(); }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
    MPI_Barrier(MPI_COMM_WORLD);

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
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, c);
        printf("METRICS bench=givens_2p5d matrix=%s N=%d b=%d c=%d passes=1 ",
               matrix_mode_tag(A.matrix), N, b, c);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", times[nrun / 2]);
        fflush(stdout);
    }

    cudaFree(d_A_orig); cudaFree(d_A_local);
    cudaFree(d_panel_recv); cudaFree(d_panel_full); cudaFree(d_Q_full); cudaFree(d_cs);
    cudaFree(d_W); cudaFree(d_info);
    if (A.n_ir > 0) { cudaFree(d_G_ref); cudaFree(d_potrf_work_ref); }
    if (use_mp) { cudaFree(d_A_panel_f); cudaFree(d_A_tr_f); cudaFree(d_W_f); }
    if (A.lookahead) {
        cudaFree(d_panel_recv2); cudaFree(d_panel_full2); cudaFree(d_Q_full2); cudaFree(d_cs2);
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
