// Multi-GPU sCQR3-2.5D benchmark — faithful to qr_schur_xpartition.tex §A.3.
//
// Implements the paper's distributed schedule:
//   - Px=Py=1, Pz=c (pure replica / row-strip): each rank owns m/c rows of A.
//   - Phase Q1: 3 iterations of (Gram, POTRF, TRSM). Gram is per-replica SYRK
//     into a partial b×b, then NCCL Allreduce(+) across replicas → full Gram.
//     POTRF + TRSM run locally on each replica (identical inputs ⇒ identical
//     outputs modulo FP determinism).
//   - Phase Q2: trailing update. Each rank computes W_local = Q^T A_tr on its
//     m/c rows, NCCL Allreduce(W), then A_tr_local -= Q_local · W.
//   - Phase Q3 (cross-replica reduce): folded into the AllReduces of Q1+Q2.
//
// All cuBLAS/cuSOLVER handles created once. All buffers preallocated. Each
// GPU has TWO streams (compute_stream, comm_stream) with CUDA events for
// cross-stream sync — zero host-side blocking in the inner loop.
//
// Build:  mpic++ -O3 -std=c++17 scqr3_2p5d_bench.cpp ... -o scqr3_2p5d_bench
// Run:    mpirun -np c ./scqr3_2p5d_bench N [b]
//
// Compares against single-GPU cuSOLVER.geqrf! on rank 0 as the "what NVIDIA
// ships you per device" reference, and against the cuSOLVERMp companion bench.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>

#define CUDA_CHECK(stmt) do { cudaError_t e=(stmt); if(e!=cudaSuccess){ fprintf(stderr,"[rank %d] CUDA %s @ %s:%d\n",_rank,cudaGetErrorString(e),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUBLAS_CHECK(stmt) do { cublasStatus_t s=(stmt); if(s!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"[rank %d] cuBLAS %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)
#define CUSOLVER_CHECK(stmt) do { cusolverStatus_t s=(stmt); if(s!=CUSOLVER_STATUS_SUCCESS){ fprintf(stderr,"[rank %d] cuSOLVER %d @ %s:%d\n",_rank,(int)s,__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,3);} } while(0)
#define NCCL_CHECK(stmt) do { ncclResult_t r=(stmt); if(r!=ncclSuccess){ fprintf(stderr,"[rank %d] NCCL %s @ %s:%d\n",_rank,ncclGetErrorString(r),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,4);} } while(0)

static int _rank = 0;

// ────────────────────────────────────────────────────────────────────────────
// Small kernel: add scalar `s` to the leading b diagonal entries of G.
__global__ void add_diag_kernel(double* G, int ldg, int b, double s) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < b) G[j + (long long)j * ldg] += s;
}

// Trace of leading b×b block (sum of diagonal entries).
__global__ void trace_b_kernel(const double* G, int ldg, int b, double* out) {
    __shared__ double sh[1024];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int j = tid; j < b; j += blockDim.x) acc += G[j + (long long)j * ldg];
    sh[tid] = acc;
    __syncthreads();
    // log-reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sh[0];
}

// Apply the Fukaya shift on-device: read trace, compute s = coef * tr, add to diag.
__global__ void shift_diag_from_trace_kernel(double* G, int ldg, int b,
                                              const double* trace_dev, double coef) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < b) {
        double tr = trace_dev[0];
        double s = coef * tr;
        G[j + (long long)j * ldg] += s;
    }
}

// ────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int c;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c);

    int N = (argc >= 2) ? std::atoi(argv[1]) : 16000;
    int b = (argc >= 3) ? std::atoi(argv[2]) : 0;

    // Heuristic b = N / sqrt(P_1) per Step 7. For pure replica scheme P_1 = 1
    // so b can be as large as fits in HBM; using N/(c+10) gives ~11 panels.
    if (b <= 0) {
        if      (N <=  4000) b = 363;
        else if (N <=  8000) b = 727;
        else if (N <= 16000) b = 1454;
        else if (N <= 24000) b = 2181;
        else if (N <= 32000) b = 2666;
        else if (N <= 48000) b = 4000;
        else                 b = 5333;
    }
    if (N % c != 0) {
        if (_rank == 0) fprintf(stderr, "N=%d not divisible by c=%d\n", N, c);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    int m_local = N / c;
    int ngpu;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    if (_rank == 0) {
        printf("==========================================================\n");
        printf(" sCQR3-2.5D C++ benchmark  N=%d  b=%d  c=%d (row-strip; Px=Py=1)\n", N, b, c);
        printf("==========================================================\n");
        fflush(stdout);
    }

    // NCCL bootstrap.
    ncclUniqueId nccl_id;
    if (_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, c, nccl_id, _rank));

    // CUDA streams + events: one compute, one comm.
    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_comm));
    cudaEvent_t evt_compute_done, evt_allreduce_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&evt_compute_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evt_allreduce_done, cudaEventDisableTiming));

    // cuBLAS + cuSOLVER handles, both bound to stream_compute.
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream_compute));
    cusolverDnHandle_t cusolver;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream_compute));

    // Local matrix A_local of size m_local × N (row-distributed).
    double* d_A_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_local, (size_t)m_local * N * sizeof(double)));

    // Random fill on host then copy to device.
    {
        std::vector<double> host(m_local * (size_t)N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A_local, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Persistent scratch.
    double* d_G_partial = nullptr;     // b×b Gram (column-major, ld=b)
    double* d_W_partial = nullptr;     // b×n  W
    double* d_trace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G_partial, (size_t)b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W_partial, (size_t)b * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(double)));

    // cusolverDn POTRF workspace.
    int potrf_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G_partial, b, &potrf_lwork));
    double* d_potrf_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potrf_work, potrf_lwork * sizeof(double)));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    const double one = 1.0, zero = 0.0, neg_one = -1.0;

    auto run_qr = [&]() {
        int k = 0;
        while (k < N) {
            int sb = std::min(b, N - k);
            int n_tr = N - (k + sb);
            double coef = 11.0 * ((double)(N) * sb + (double)sb * (sb + 1)) * (2.220446049250313e-16);  // Fukaya shift coef × FP64 unit roundoff

            // ─── Phase Q1: 3 iterations of (Gram, POTRF, TRSM). ───────────
            for (int it = 0; it < 3; ++it) {
                // Step 1: local SYRK G_partial = A_panel^T · A_panel
                // A_panel column-major: d_A_local[k*m_local : (k+sb)*m_local]
                // Actually for column-major Julia/Fortran style, A is m_local × N
                // with column-major layout. So A_panel starts at d_A_local + k*m_local
                // and is m_local × sb with stride m_local.
                double* d_A_panel = d_A_local + (size_t)k * m_local;
                CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                          sb, m_local,
                                          &one, d_A_panel, m_local,
                                          &zero, d_G_partial, b));

                // Step 2: stream_compute → evt_compute_done → stream_comm waits
                CUDA_CHECK(cudaEventRecord(evt_compute_done, stream_compute));
                CUDA_CHECK(cudaStreamWaitEvent(stream_comm, evt_compute_done, 0));

                // Step 3: NCCL Allreduce on stream_comm
                NCCL_CHECK(ncclAllReduce(d_G_partial, d_G_partial, (size_t)b * b,
                                          ncclDouble, ncclSum, nccl_comm, stream_comm));

                // Step 4: comm done → compute waits
                CUDA_CHECK(cudaEventRecord(evt_allreduce_done, stream_comm));
                CUDA_CHECK(cudaStreamWaitEvent(stream_compute, evt_allreduce_done, 0));

                // Step 5: Fukaya shift (iter 0 only) — on stream_compute
                if (it == 0) {
                    trace_b_kernel<<<1, std::min(b, 1024), 0, stream_compute>>>(d_G_partial, b, sb, d_trace);
                    shift_diag_from_trace_kernel<<<(sb + 255) / 256, 256, 0, stream_compute>>>(d_G_partial, b, sb, d_trace, coef);
                }

                // Step 6: POTRF
                CUSOLVER_CHECK(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G_partial, b, d_potrf_work, potrf_lwork, d_info));

                // Step 7: TRSM A_panel := A_panel · R^-1
                CUBLAS_CHECK(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                          m_local, sb, &one, d_G_partial, b, d_A_panel, m_local));
            }

            // ─── Phase Q2: trailing update on A[:, k+sb : N] ──────────────
            if (n_tr > 0) {
                double* d_A_panel = d_A_local + (size_t)k * m_local;
                double* d_A_tr    = d_A_local + (size_t)(k + sb) * m_local;

                // W_local = A_panel^T · A_tr   (sb × n_tr GEMM)
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                          sb, n_tr, m_local,
                                          &one, d_A_panel, m_local,
                                                d_A_tr, m_local,
                                          &zero, d_W_partial, b));

                // Allreduce W
                CUDA_CHECK(cudaEventRecord(evt_compute_done, stream_compute));
                CUDA_CHECK(cudaStreamWaitEvent(stream_comm, evt_compute_done, 0));
                NCCL_CHECK(ncclAllReduce(d_W_partial, d_W_partial, (size_t)sb * n_tr,
                                          ncclDouble, ncclSum, nccl_comm, stream_comm));
                CUDA_CHECK(cudaEventRecord(evt_allreduce_done, stream_comm));
                CUDA_CHECK(cudaStreamWaitEvent(stream_compute, evt_allreduce_done, 0));

                // A_tr -= A_panel · W
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                          m_local, n_tr, sb,
                                          &neg_one, d_A_panel, m_local,
                                                    d_W_partial, b,
                                          &one,    d_A_tr, m_local));
            }
            k += sb;
        }
    };

    // Save a copy of A_orig for correctness re-runs.
    double* d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_orig, (size_t)m_local * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A_orig, d_A_local, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));

    // Warmup (restore A_local from A_orig each time so the bench is reproducible).
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A_local, d_A_orig, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));
        run_qr();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_comm));
    MPI_Barrier(MPI_COMM_WORLD);

    // ─── Correctness check (after warmup): max|diag(Q'Q)-1| across replicas ──
    // The c replicas each hold a row-strip of Q (length m_local). The local
    // SYRK A^T A computes the contribution; AllReduce sums across replicas to
    // give the full Q'Q.
    {
        double* d_QtQ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_QtQ, (size_t)N * N * sizeof(double)));
        CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  N, m_local, &one, d_A_local, m_local,
                                  &zero, d_QtQ, N));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        NCCL_CHECK(ncclAllReduce(d_QtQ, d_QtQ, (size_t)N * N, ncclDouble, ncclSum, nccl_comm, stream_comm));
        CUDA_CHECK(cudaStreamSynchronize(stream_comm));
        if (_rank == 0) {
            std::vector<double> hd(N);
            for (int j = 0; j < N; ++j)
                CUDA_CHECK(cudaMemcpy(&hd[j], d_QtQ + j + (size_t)j*N, sizeof(double), cudaMemcpyDeviceToHost));
            double max_dev = 0.0;
            for (auto v : hd) max_dev = std::max(max_dev, std::abs(v - 1.0));
            printf("  Validation     N=%d c=%d  max|diag(Q'Q)-1| = %.2e\n", N, c, max_dev);
            fflush(stdout);
        }
        cudaFree(d_QtQ);
    }

    // Timed runs (restore A_local before each so each run measures a real factorization).
    const int nrun = 5;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A_local, d_A_orig, (size_t)m_local * N * sizeof(double), cudaMemcpyDeviceToDevice));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qr();
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_comm));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        printf("  sCQR3-2.5D       N=%d  b=%d  c=%d   tmin=%9.2f ms  tmed=%9.2f ms\n",
               N, b, c, times[0], times[nrun/2]);
        fflush(stdout);
    }

    cudaFree(d_A_orig); cudaFree(d_A_local); cudaFree(d_G_partial); cudaFree(d_W_partial);
    cudaFree(d_trace); cudaFree(d_potrf_work); cudaFree(d_info);
    cublasDestroy(cublas); cusolverDnDestroy(cusolver);
    cudaStreamDestroy(stream_compute); cudaStreamDestroy(stream_comm);
    cudaEventDestroy(evt_compute_done); cudaEventDestroy(evt_allreduce_done);
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();
    return 0;
}
