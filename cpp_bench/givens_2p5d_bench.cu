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
//     --mp           trailing-update GEMMs in FP32  (Phase Q6 low-prec)
//     --ir=K         K rounds of Cholesky-QR refinement after factor
//                     (Phase Q6)
//     --la           2-stream pipelining of Phase Q1 / Phase Q2 (Phase Q5)
//
//   Processor grid: 1D row partition, m_local = N / c rows per rank.

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

// One-block-per-panel Givens panel factorization. Block size: threads = blockDim.x.
//   - A is m × b column-major, ld = lda (= m for our gather-form panel)
//   - cs[j*(m-1)+(m-1-i)] stores (c, s) for the rotation that zeroed A[i, j]
//     stored as cs[2*idx], cs[2*idx+1]
// Sequential over (j, i). Parallel over the trailing-column update (across
// threads) which dominates the inner work.
__global__ void givens_panel_kernel(double* __restrict__ A, int m, int b, int lda,
                                     double* __restrict__ cs_save) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    __shared__ double c_sh, s_sh;
    for (int j = 0; j < b; ++j) {
        for (int i = m - 1; i > j; --i) {
            if (tid == 0) {
                double a = A[(long long)(i - 1) + (long long)j * lda];
                double bv = A[(long long)i      + (long long)j * lda];
                double r = hypot(a, bv);
                double c = (r != 0.0) ? a / r : 1.0;
                double s = (r != 0.0) ? bv / r : 0.0;
                A[(long long)(i - 1) + (long long)j * lda] = r;
                A[(long long)i       + (long long)j * lda] = 0.0;
                c_sh = c;
                s_sh = s;
                long long idx = (long long)j * (m - 1) + (long long)(m - 1 - i);
                cs_save[2 * idx]     = c;
                cs_save[2 * idx + 1] = s;
            }
            __syncthreads();
            double c = c_sh, s = s_sh;
            for (int k = j + 1 + tid; k < b; k += bs) {
                double a  = A[(long long)(i - 1) + (long long)k * lda];
                double bv = A[(long long)i       + (long long)k * lda];
                A[(long long)(i - 1) + (long long)k * lda] =  c * a + s * bv;
                A[(long long)i       + (long long)k * lda] = -s * a + c * bv;
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
__global__ void form_Q_from_givens_kernel(double* __restrict__ Q, int m, int b, int ldq,
                                          const double* __restrict__ cs_save) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    // Apply rotations in reverse of the panel factorization order: undoing
    // the elimination in reverse gives Q (since the factor produced R from A
    // by applying rotations in the forward order; Q is the accumulation in
    // reverse).
    for (int j = b - 1; j >= 0; --j) {
        for (int i = j + 1; i < m; ++i) {
            long long idx = (long long)j * (m - 1) + (long long)(m - 1 - i);
            double c = cs_save[2 * idx];
            double s = cs_save[2 * idx + 1];
            // The rotation that ZEROED A[i, j] used (c, s); to reconstruct
            // Q's contribution we apply the same rotation to Q's rows (i-1, i)
            // with sign that matches: Q ← G(i-1, i; c, s) · Q   (forward apply).
            for (int k = tid; k < b; k += bs) {
                double a  = Q[(long long)(i - 1) + (long long)k * ldq];
                double bv = Q[(long long)i       + (long long)k * ldq];
                Q[(long long)(i - 1) + (long long)k * ldq] =  c * a + s * bv;
                Q[(long long)i       + (long long)k * ldq] = -s * a + c * bv;
            }
            __syncthreads();
        }
    }
}

struct Args {
    int N = 16000, b = 0, n_ir = 0;
    bool mp = false, lookahead = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s.rfind("--N=", 0) == 0) a.N = std::atoi(s.c_str() + 4);
        else if (s.rfind("--b=", 0) == 0) a.b = std::atoi(s.c_str() + 4);
        else if (s.rfind("--ir=", 0) == 0) a.n_ir = std::atoi(s.c_str() + 5);
        else if (s == "--mp") a.mp = true;
        else if (s == "--lookahead" || s == "--la") a.lookahead = true;
    }
    if (a.b == 0) {
        // Givens panel kernel is sequential in (j, i); use a smaller b to
        // keep per-panel work tractable while still amortizing the gather.
        if      (a.N <=  4000) a.b = 64;
        else if (a.N <=  8000) a.b = 96;
        else if (a.N <= 16000) a.b = 128;
        else if (a.N <= 32000) a.b = 192;
        else if (a.N <= 48000) a.b = 256;
        else                   a.b = 256;
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
    std::snprintf(tag, sizeof(tag), "givens%s%s%s%s",
                  A.mp?"+MP":"", A.lookahead?"+LA":"",
                  A.n_ir>0?"+IR":"", A.n_ir>0?(std::to_string(A.n_ir)).c_str():"");

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
    if (A.mp) {
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
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                      sb, ncols, m_local,
                                      &one_f, d_A_panel_f, m_local, d_A_tr_f, m_local,
                                      &zero_f, d_W_f, b));
            CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
            NCCL_CHECK(ncclAllReduce(d_W_f, d_W_f, (size_t)sb * ncols, ncclFloat, ncclSum, nccl_comm, s_comm));
            CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_local, ncols, sb,
                                      &neg_one_f, d_A_panel_f, m_local, d_W_f, b,
                                      &one_f, d_A_tr_f, m_local));
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
                if (n_tr > 0) phase_q2(s_comp, cublas, k, sb, k + sb, n_tr, A.mp);
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
                    phase_q2(s_comp, cublas, k, sb, next_k, next_sb, A.mp);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, s_comp));
                    CUDA_CHECK(cudaStreamWaitEvent(s_la, e_next_ready, 0));
                    phase_q1(s_la, cublas_la, cusolver_la,
                             d_panel_recv2, d_panel_full2, d_Q_full2, d_cs2,
                             next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, s_la));
                }
                if (n_rest > 0) phase_q2(s_comp, cublas, k, sb, rest_start, n_rest, A.mp);
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
        fflush(stdout);
    }

    cudaFree(d_A_orig); cudaFree(d_A_local);
    cudaFree(d_panel_recv); cudaFree(d_panel_full); cudaFree(d_Q_full); cudaFree(d_cs);
    cudaFree(d_W); cudaFree(d_info);
    if (A.n_ir > 0) { cudaFree(d_G_ref); cudaFree(d_potrf_work_ref); }
    if (A.mp) { cudaFree(d_A_panel_f); cudaFree(d_A_tr_f); cudaFree(d_W_f); }
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
