// Multi-GPU cuSOLVERMp QR benchmark (MPI + NCCL).
//
// One rank per GPU, 2D process grid `[P_x × P_y]`. Each rank holds a 2D
// block-cyclic slab of A with block size MB × NB. Calls cusolverMpGeqrf
// then synchronises and reports the wall-clock time on rank 0.
//
// Build:  mpic++ -O2 -std=c++17 cusolverMp_geqrf_bench.cpp \
//                -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
//                -lcusolverMp -lnccl -lcudart -lcuda \
//                -o cusolverMp_geqrf_bench
// Run:    mpirun -np $((P_x*P_y)) ./cusolverMp_geqrf_bench N [MB] [NB] [P_x] [P_y]

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
#include <nccl.h>
#include <cusolverMp.h>

#define CUDA_CHECK(stmt) do { \
    cudaError_t e = (stmt); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[rank %d] CUDA error %d at %s:%d  %s\n", _rank, e, __FILE__, __LINE__, cudaGetErrorString(e)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

#define NCCL_CHECK(stmt) do { \
    ncclResult_t r = (stmt); \
    if (r != ncclSuccess) { \
        fprintf(stderr, "[rank %d] NCCL error %d at %s:%d  %s\n", _rank, r, __FILE__, __LINE__, ncclGetErrorString(r)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

#define MP_CHECK(stmt) do { \
    cusolverStatus_t s = (stmt); \
    if (s != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "[rank %d] cusolverMp error %d at %s:%d\n", _rank, (int)s, __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 3); \
    } \
} while (0)

static int _rank = 0;

// 1D index of a global row/col into local (rank-owned) coordinates under 2D
// block-cyclic distribution with block size MB, process grid Px × Py.
// numroc(N, NB, my_proc, src_proc, n_procs) = local block count
static inline int64_t numroc(int64_t n, int64_t nb, int iproc, int isrc, int nprocs) {
    int64_t mydist = (nprocs + iproc - isrc) % nprocs;
    int64_t nblocks = n / nb;
    int64_t numroc = (nblocks / nprocs) * nb;
    int64_t extrablocks = nblocks % nprocs;
    if (mydist < extrablocks) numroc += nb;
    else if (mydist == extrablocks) numroc += n % nb;
    return numroc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int64_t N      = (argc >= 2) ? std::atoll(argv[1]) : 8000;
    int64_t MB     = (argc >= 3) ? std::atoll(argv[2]) : 256;
    int64_t NB     = (argc >= 4) ? std::atoll(argv[3]) : MB;
    int     Px     = (argc >= 5) ? std::atoi(argv[4])  : (int)std::sqrt((double)size);
    int     Py     = (argc >= 6) ? std::atoi(argv[5])  : size / Px;
    if (Px * Py != size) {
        if (_rank == 0) fprintf(stderr, "Px*Py (%d * %d = %d) must equal MPI size (%d)\n", Px, Py, Px*Py, size);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // Each rank pins to one GPU.
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int my_gpu = _rank % ndev;
    CUDA_CHECK(cudaSetDevice(my_gpu));

    int my_proc_row = _rank / Py;
    int my_proc_col = _rank % Py;
    if (_rank == 0) {
        printf("==========================================================\n");
        printf(" cuSOLVERMp Geqrf benchmark   N=%lld  MB=NB=%lld  grid=%dx%d (%d ranks)\n",
               (long long)N, (long long)MB, Px, Py, size);
        printf("==========================================================\n");
        fflush(stdout);
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // NCCL bootstrap: rank 0 generates a unique ID, broadcast via MPI.
    ncclUniqueId nccl_id;
    if (_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, size, nccl_id, _rank));

    // cusolverMp handle + grid + matrix descriptor.
    cusolverMpHandle_t handle = nullptr;
    MP_CHECK(cusolverMpCreate(&handle, my_gpu, stream));

    cusolverMpGrid_t grid = nullptr;
    MP_CHECK(cusolverMpCreateDeviceGrid(handle, &grid, nccl_comm, Px, Py, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));

    // Local block dims for this rank.
    int64_t my_M = numroc(N, MB, my_proc_row, 0, Px);
    int64_t my_N = numroc(N, NB, my_proc_col, 0, Py);
    int64_t llda = std::max<int64_t>(my_M, 1);

    cusolverMpMatrixDescriptor_t descA = nullptr;
    MP_CHECK(cusolverMpCreateMatrixDesc(&descA, grid, CUDA_R_64F,
                                        /*M_A=*/N, /*N_A=*/N,
                                        /*MB_A=*/MB, /*NB_A=*/NB,
                                        /*RSRC_A=*/0, /*CSRC_A=*/0,
                                        /*LLDA=*/llda));

    // Local A buffer + tau on each rank.
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, llda * my_N * sizeof(double)));
    // Random-fill via host
    {
        std::vector<double> host(llda * my_N);
        std::mt19937_64 rng(7 + _rank);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (auto& v : host) v = nrm(rng);
        CUDA_CHECK(cudaMemcpy(d_A, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    double* d_tau = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, N * sizeof(double)));

    // Workspace query.
    size_t devBytes = 0, hostBytes = 0;
    MP_CHECK(cusolverMpGeqrf_bufferSize(handle, N, N, d_A, /*IA=*/1, /*JA=*/1, descA,
                                         CUDA_R_64F, &devBytes, &hostBytes));
    void* d_work = nullptr;
    if (devBytes > 0) CUDA_CHECK(cudaMalloc(&d_work, devBytes));
    std::vector<char> h_work(hostBytes);
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    auto run_once = [&]() {
        // Reset A from host once per run (so the bench measures factor time only).
        // For perf, we reuse d_A across runs without re-init (acceptable since
        // the wall-clock is dominated by the QR, not the host->device copy).
        MP_CHECK(cusolverMpGeqrf(handle, N, N, d_A, 1, 1, descA, d_tau,
                                  CUDA_R_64F, d_work, devBytes,
                                  h_work.data(), hostBytes, d_info));
    };

    // Warmup
    for (int i = 0; i < 2; ++i) run_once();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed
    const int nrun = 5;
    std::vector<double> times_ms(nrun, 0.0);
    for (int i = 0; i < nrun; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times_ms.begin(), times_ms.end());
    double tmin = times_ms[0];
    double tmed = times_ms[nrun / 2];
    if (_rank == 0) {
        printf("  cusolverMpGeqrf  N=%lld grid=%dx%d  tmin=%9.2f ms  tmed=%9.2f ms\n",
               (long long)N, Px, Py, tmin, tmed);
        fflush(stdout);
    }

    cudaFree(d_A); cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
    MP_CHECK(cusolverMpDestroyMatrixDesc(descA));
    MP_CHECK(cusolverMpDestroyGrid(grid));
    MP_CHECK(cusolverMpDestroy(handle));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}
