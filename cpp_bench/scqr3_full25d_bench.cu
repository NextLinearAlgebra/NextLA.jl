// Full 2.5D sCQR3 / CQR2 — Conflux-style [P_x, P_y, P_z] decomposition.
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
//     --N, --b, --passes, --px, --py, --pz, --la
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
    int N = 8000, b = 0, passes = 3;
    int px = 0, py = 0, pz = 0;
    bool lookahead = false;
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
        else if (s == "--la" || s == "--lookahead") a.lookahead = true;
    }
    if (a.b == 0) {
        if      (a.N <=  4000) a.b = 363;
        else if (a.N <=  8000) a.b = 512;
        else if (a.N <= 16000) a.b = 512;
        else if (a.N <= 32000) a.b = 512;
        else if (a.N <= 64000) a.b = 1024;
        else                   a.b = 1024;
    }
    if (a.passes < 1 || a.passes > 3) a.passes = 3;
    return a;
}

// Resolve [px, py, pz] from P and CLI hints. If none specified, pick
// (px, py, pz) = (sqrt(P/c), sqrt(P/c), c) with c = floor(P^{1/3}).
static void resolve_grid(int P, Args& A) {
    if (A.px > 0 && A.py > 0 && A.pz > 0) {
        if (A.px * A.py * A.pz != P) {
            if (_rank == 0) fprintf(stderr, "px*py*pz = %d != P = %d\n", A.px*A.py*A.pz, P);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        return;
    }
    // Default: pick (px, py, pz) with px*py*pz = P, px = py = sqrt(P/c) integer,
    // c = floor(P^{1/3}).
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
    // Fallback: 1 x P x 1
    A.px = 1; A.py = P; A.pz = 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int P;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    Args A = parse_args(argc, argv);
    resolve_grid(P, A);

    int N = A.N, b = A.b;
    int Px = A.px, Py = A.py, Pz = A.pz, P1 = Px * Py;
    int m_loc = N / (Px * Pz);
    int n_loc = N / Py;
    if (N % (Px * Pz) != 0 || N % Py != 0) {
        if (_rank == 0) fprintf(stderr, "N=%d not divisible by Px*Pz=%d or Py=%d\n", N, Px*Pz, Py);
        MPI_Abort(MPI_COMM_WORLD, 6);
    }

    // Decode (px, py, pz) from rank.  Linear ordering: rank = pz*Px*Py + px*Py + py.
    int my_pz = _rank / (Px * Py);
    int my_px = (_rank / Py) % Px;
    int my_py = _rank % Py;

    int ngpu; CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    char tag[160];
    std::snprintf(tag, sizeof(tag), "p25d_passes=%d%s", A.passes, A.lookahead?"+LA":"");

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" Full 2.5D  N=%d b=%d  grid=[Px=%d, Py=%d, Pz=%d]  variant=%s\n",
               N, b, Px, Py, Pz, tag);
        printf("   m_loc=%d  n_loc=%d  P1=%d  c=Pz=%d\n", m_loc, n_loc, P1, Pz);
        printf("=================================================================\n");
        fflush(stdout);
    }

    // ── Sub-communicators ──────────────────────────────────────────────────
    // col_comm: ranks sharing the same py.  Used to AllReduce G (Phase Q3)
    //           and W (Phase Q4); P_x * P_z ranks per group.
    // row_comm: ranks sharing the same (px, pz).  Used to broadcast Q from
    //           py_panel to all py'; P_y ranks per group.
    MPI_Comm mpi_col_comm, mpi_row_comm;
    {
        int col_color = my_py;
        int col_key   = my_pz * Px + my_px;
        MPI_Comm_split(MPI_COMM_WORLD, col_color, col_key, &mpi_col_comm);
        int row_color = my_pz * Px + my_px;
        int row_key   = my_py;
        MPI_Comm_split(MPI_COMM_WORLD, row_color, row_key, &mpi_row_comm);
    }
    int col_size, col_rank, row_size, row_rank;
    MPI_Comm_size(mpi_col_comm, &col_size); MPI_Comm_rank(mpi_col_comm, &col_rank);
    MPI_Comm_size(mpi_row_comm, &row_size); MPI_Comm_rank(mpi_row_comm, &row_rank);

    // NCCL sub-comms.
    ncclComm_t nccl_col, nccl_row;
    {
        ncclUniqueId id;
        if (col_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_col_comm);
        NCCL_CHECK(ncclCommInitRank(&nccl_col, col_size, id, col_rank));
    }
    {
        ncclUniqueId id;
        if (row_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_row_comm);
        NCCL_CHECK(ncclCommInitRank(&nccl_row, row_size, id, row_rank));
    }

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
    if (A.lookahead) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));       CUBLAS_CHECK(cublasSetStream(cublas_la, s_la));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la)); CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, s_la));
    }

    // Local data A (m_loc x n_loc column-major) and A_orig for reset.
    double *d_A = nullptr, *d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,      (size_t)m_loc * n_loc * sizeof(double)));
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

    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Look-ahead second-panel scratch.
    double *d_G2=nullptr, *d_potrf_work2=nullptr, *d_panel_bcast2=nullptr;
    int *d_info2=nullptr;
    if (A.lookahead) {
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
        int my_col_end_global   = my_col_start_global + n_loc;
        int trail_global_start  = k + sb;
        int local_start = std::max(0, trail_global_start - my_col_start_global);
        int local_end   = n_loc;
        int ncols = local_end - local_start;
        if (ncols <= 0) {
            // Even if no trailing cols, must participate in the AllReduce since
            // col_group has shared semantics.  Issue a zero-size AllReduce by
            // using ncols=0? NCCL doesn't support 0-size AllReduce reliably,
            // so we skip the AllReduce entirely on this rank but ensure the
            // group is balanced (every rank with same my_py must skip together).
            // Since trail_global_start is the same for every rank, and the
            // condition depends only on my_py, all ranks in col_group(my_py)
            // skip together.  Safe.
            return;
        }
        double* A_trail_loc = d_A + (size_t)local_start * m_loc;
        // 1. W_loc = Q_loc^T A_trail_loc
        CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                                  sb, ncols, m_loc,
                                  &one_d, p_bcast, m_loc,
                                          A_trail_loc, m_loc,
                                  &zero_d, d_W, b));
        // 2. AllReduce W across col_group(my_py)
        CUDA_CHECK(cudaEventRecord(e_comp_done, s_use));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, e_comp_done, 0));
        NCCL_CHECK(ncclAllReduce(d_W, d_W, (size_t)sb * ncols, ncclDouble, ncclSum,
                                  nccl_col_h, s_comm));
        CUDA_CHECK(cudaEventRecord(e_ar_done, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_use, e_ar_done, 0));
        // 3. A_trail_loc -= Q_loc * W_loc
        CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m_loc, ncols, sb,
                                  &neg_one_d, p_bcast, m_loc,
                                              d_W, b,
                                  &one_d, A_trail_loc, m_loc));
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
        if (!A.lookahead) {
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
    if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
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
        if (A.lookahead) CUDA_CHECK(cudaStreamSynchronize(s_la));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        printf("  %-30s  N=%d b=%d grid=[%d,%d,%d]  tmin=%9.2f ms  tmed=%9.2f ms\n",
               tag, N, b, Px, Py, Pz, times[0], times[nrun/2]);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_A_orig); cudaFree(d_G); cudaFree(d_W); cudaFree(d_panel_bcast); cudaFree(d_trace);
    cudaFree(d_potrf_work); cudaFree(d_info);
    if (A.lookahead) {
        cudaFree(d_G2); cudaFree(d_potrf_work2); cudaFree(d_info2); cudaFree(d_panel_bcast2);
        cublasDestroy(cublas_la); cusolverDnDestroy(cusolver_la);
    }
    cublasDestroy(cublas); cusolverDnDestroy(cusolver);
    cudaStreamDestroy(s_comp); cudaStreamDestroy(s_comm); cudaStreamDestroy(s_la);
    cudaEventDestroy(e_comp_done); cudaEventDestroy(e_ar_done);
    cudaEventDestroy(e_panel_done); cudaEventDestroy(e_next_ready);
    ncclCommDestroy(nccl_col);
    ncclCommDestroy(nccl_row);
    MPI_Comm_free(&mpi_col_comm);
    MPI_Comm_free(&mpi_row_comm);
    MPI_Finalize();
    return 0;
}
