// Strong-scaling benchmark for KBLAS TLR GEMM.
//
// The same executable benchmarks both supported low-rank output modes:
//   lld: TLR x TLR -> dense
//   lll: TLR x TLR -> TLR
//
// The dense baseline is dense GEMM on independently generated dense operands.
// KBLAS exposes separate input ranks kA and kB for both APIs; the command line
// keeps those ranks independent as well.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <magma_v2.h>

#include "kblas.h"
#include "kblas_tlr.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); std::exit(1); } \
} while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t e = (x); if (e != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error: %d\n", int(e)); std::exit(1); } \
} while (0)
#define CURAND_CHECK(x) do { curandStatus_t e = (x); if (e != CURAND_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuRAND error: %d\n", int(e)); std::exit(1); } \
} while (0)
#define KBLAS_CHECK(x) do { int e = (x); if (e != KBLAS_Success) { \
    std::fprintf(stderr, "KBLAS error: %d (%s)\n", e, kblasGetErrorString(e)); std::exit(1); } \
} while (0)

enum class Mode { LLD, LLL };
struct Config {
    int m, k, n, b, rank_A, rank_B, rank_C, warmup, reps;
    Mode mode;
};

template <typename T> struct Backend;

template <> struct Backend<float> {
    static const char *name() { return "Float32"; }
    static void random(curandGenerator_t g, float *p, size_t n)
    { CURAND_CHECK(curandGenerateNormal(g, p, n, 0.0f, 1.0f)); }
    static void dense(cublasHandle_t h, const Config &c,
                      const float *A, const float *B, float *C)
    {
        const float alpha = 1.0f, beta = 1.0f;
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                 c.m, c.n, c.k, &alpha, A, c.m,
                                 B, c.k, &beta, C, c.m));
    }
    static void lld_workspace(kblasHandle_t h, const Config &c)
    { kblasSgemm_tlr_lld_wsquery(h, c.m / c.b, c.n / c.b,
                                 c.rank_A, c.rank_B, c.b, c.b); }
    static void lll_workspace(kblasHandle_t h, const Config &c)
    { kblasSgemm_tlr_lll_wsquery(h, c.m / c.b, c.n / c.b,
                                 c.rank_A, c.rank_B, c.rank_C, c.rank_C,
                                 c.b, c.b); }
};

template <> struct Backend<double> {
    static const char *name() { return "Float64"; }
    static void random(curandGenerator_t g, double *p, size_t n)
    { CURAND_CHECK(curandGenerateNormalDouble(g, p, n, 0.0, 1.0)); }
    static void dense(cublasHandle_t h, const Config &c,
                      const double *A, const double *B, double *C)
    {
        const double alpha = 1.0, beta = 1.0;
        CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                 c.m, c.n, c.k, &alpha, A, c.m,
                                 B, c.k, &beta, C, c.m));
    }
    static void lld_workspace(kblasHandle_t h, const Config &c)
    { kblasDgemm_tlr_lld_wsquery(h, c.m / c.b, c.n / c.b,
                                 c.rank_A, c.rank_B, c.b, c.b); }
    static void lll_workspace(kblasHandle_t h, const Config &c)
    { kblasDgemm_tlr_lll_wsquery(h, c.m / c.b, c.n / c.b,
                                 c.rank_A, c.rank_B, c.rank_C, c.rank_C,
                                 c.b, c.b); }
};

template <typename T>
struct DeviceFactors {
    T *u = nullptr;
    T *v = nullptr;
    T **u_ptrs = nullptr;
    T **v_ptrs = nullptr;
    size_t elements = 0;
};

template <typename T>
__global__ void fill_kernel(T *x, size_t n, T value)
{
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) x[i] = value;
}

template <typename T>
void fill_device(cudaStream_t stream, T *x, size_t n, T value)
{
    constexpr int threads = 256;
    int blocks = int((n + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, stream>>>(x, n, value);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
DeviceFactors<T> allocate_factors(int rows, int cols, int b, int rank,
                                  curandGenerator_t generator)
{
    const int mt = rows / b;
    const int nt = cols / b;
    const size_t tile_count = size_t(mt) * nt;
    const size_t stride = size_t(b) * rank;
    DeviceFactors<T> result;
    result.elements = tile_count * stride;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result.u),
                          result.elements * sizeof(T)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result.v),
                          result.elements * sizeof(T)));
    Backend<T>::random(generator, result.u, result.elements);
    Backend<T>::random(generator, result.v, result.elements);

    std::vector<T *> host_u(tile_count), host_v(tile_count);
    for (int j = 0; j < nt; ++j) for (int i = 0; i < mt; ++i) {
        size_t slot = size_t(i) + size_t(j) * mt;
        host_u[slot] = result.u + slot * stride;
        host_v[slot] = result.v + slot * stride;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result.u_ptrs),
                          tile_count * sizeof(T *)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&result.v_ptrs),
                          tile_count * sizeof(T *)));
    CUDA_CHECK(cudaMemcpy(result.u_ptrs, host_u.data(),
                          tile_count * sizeof(T *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.v_ptrs, host_v.data(),
                          tile_count * sizeof(T *), cudaMemcpyHostToDevice));
    return result;
}

template <typename T>
void release(DeviceFactors<T> &x)
{
    cudaFree(x.u); cudaFree(x.v);
    cudaFree(x.u_ptrs); cudaFree(x.v_ptrs);
    x = DeviceFactors<T>{};
}

// KBLAS's dense-output overload declares its pointer tables as const T**,
// while the same table storage is used by the LLL overload as T**.
template <typename T>
const T **as_const_ptrs(T **p)
{ return (const T **)p; }

struct Timing {
    double median_ms;
    double minimum_ms;
};

template <typename Operation, typename Reset>
Timing measure_time_ms(Operation operation, Reset reset, cudaStream_t stream,
                       int warmup, int reps)
{
    for (int i = 0; i < warmup; ++i) {
        reset(); operation();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        reset();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        operation();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    const double median = reps % 2 == 0
        ? 0.5 * (samples[reps / 2 - 1] + samples[reps / 2])
        : samples[reps / 2];
    return Timing{median, samples.front()};
}

template <typename T>
double gemm_flops(int m, int n, int k)
{ return 2.0 * m * n * k; }

// These formulas follow KBLAS's real-valued FLOPS_GEMM_LR_LLD/LLL estimates.
template <typename T>
double lld_tile_flops(int b, int rank_A, int rank_B)
{
    return gemm_flops<T>(rank_A, rank_B, b) +
           gemm_flops<T>(b, rank_A, rank_B) +
           gemm_flops<T>(b, b, rank_A);
}

template <typename T>
double geqrf_flops(int m, int n)
{
    if (m > n)
        return n * (n * (0.5 - n / 3.0 + m) + m + 23.0 / 6.0) +
               n * (n * (0.5 - n / 3.0 + m) + 5.0 / 6.0);
    return m * (m * (-0.5 - m / 3.0 + n) + 2.0 * n + 23.0 / 6.0) +
           m * (m * (-0.5 - m / 3.0 + n) + n + 5.0 / 6.0);
}

template <typename T>
double orgqr_flops(int m, int n, int k)
{
    return k * (2.0 * m * n + 2.0 * n - 5.0 / 3.0 +
                k * (2.0 / 3.0 * k - (m + n) - 1.0)) +
           k * (2.0 * m * n + n - m + 1.0 / 3.0 +
                k * (2.0 / 3.0 * k - (m + n)));
}

template <typename T>
double trmm_right_flops(int m, int n)
{
    return 0.5 * n * m * (n + 1.0) + 0.5 * n * m * (n - 1.0);
}

template <typename T>
double lll_tile_flops(int b, int rank_A, int rank_B, int rank_C)
{
    const int q = rank_A + rank_C;
    return gemm_flops<T>(rank_A, rank_B, b) +
           gemm_flops<T>(b, rank_A, rank_B) +
           geqrf_flops<T>(b, q) + b * rank_C +
           geqrf_flops<T>(b, q) + trmm_right_flops<T>(q, q) +
           3.0 * q * q * q +
           orgqr_flops<T>(b, q, q) + orgqr_flops<T>(b, q, q) +
           gemm_flops<T>(b, q, q) + gemm_flops<T>(b, q, q);
}

template <typename T>
int run(const Config &c)
{
    if (c.m <= 0 || c.k <= 0 || c.n <= 0 || c.b <= 0 ||
        c.m % c.b || c.k % c.b || c.n % c.b ||
        c.rank_A <= 0 || c.rank_B <= 0 || c.rank_C <= 0 ||
        c.rank_A >= c.b || c.rank_B >= c.b || c.rank_C >= c.b) {
        std::fprintf(stderr, "dimensions must be positive, tile-divisible, and all ranks < tile size\n");
        return 2;
    }

    const int mt = c.m / c.b, kt = c.k / c.b, nt = c.n / c.b;
    const size_t dense_elements = size_t(c.m) * c.n;
    const size_t c_factor_elements = size_t(mt) * nt * c.b * c.rank_C;
    const bool lll = c.mode == Mode::LLL;

    kblasHandle_t handle;
    KBLAS_CHECK(kblasCreate(&handle));
    if (lll) {
        magma_init();
        KBLAS_CHECK(kblasEnableMagma(handle));
    }
    cudaStream_t stream = kblasGetStream(handle);
    cublasHandle_t dense_handle = kblasGetCublasHandle(handle);
    CUBLAS_CHECK(cublasSetStream(dense_handle, stream));

    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 20260728ULL));
    CURAND_CHECK(curandSetStream(generator, stream));

    DeviceFactors<T> A = allocate_factors<T>(c.m, c.k, c.b, c.rank_A, generator);
    DeviceFactors<T> B = allocate_factors<T>(c.k, c.n, c.b, c.rank_B, generator);
    DeviceFactors<T> C;
    T *dense_C = nullptr;
    int final_rank = 0;

    if (lll) {
        C = allocate_factors<T>(c.m, c.n, c.b, c.rank_C, generator);
        fill_device(stream, C.u, c_factor_elements, T(1));
        fill_device(stream, C.v, c_factor_elements, T(1));
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dense_C),
                              dense_elements * sizeof(T)));
        fill_device(stream, dense_C, dense_elements, T(1));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CURAND_CHECK(curandDestroyGenerator(generator));

    if (lll) {
        Backend<T>::lll_workspace(handle, c);
        KBLAS_CHECK(kblasAllocateWorkspace(handle));
        auto operation = [&] {
            int kC = c.rank_C;
            KBLAS_CHECK(kblas_gemm_tlr(
                handle, KBLAS_NoTrans, KBLAS_NoTrans,
                mt, nt, kt, c.b, c.b, c.b, T(1),
                A.u_ptrs, c.b, A.v_ptrs, c.b, mt, c.rank_A,
                B.u_ptrs, c.b, B.v_ptrs, c.b, kt, c.rank_B,
                T(1), C.u_ptrs, c.b, C.v_ptrs, c.b, mt, kC,
                c.rank_C, 0.0));
            final_rank = kC;
        };
        auto reset = [&] {
            fill_device(stream, C.u, c_factor_elements, T(1));
            fill_device(stream, C.v, c_factor_elements, T(1));
        };
        const Timing tlr = measure_time_ms(operation, reset, stream,
                                           c.warmup, c.reps);
        KBLAS_CHECK(kblasFreeWorkspace(handle));
        release(A); release(B); release(C);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dense_C),
                              dense_elements * sizeof(T)));
        T *dense_A = nullptr, *dense_B = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dense_A),
                              size_t(c.m) * c.k * sizeof(T)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dense_B),
                              size_t(c.k) * c.n * sizeof(T)));
        curandGenerator_t dense_generator;
        CURAND_CHECK(curandCreateGenerator(&dense_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(dense_generator, 314159ULL));
        CURAND_CHECK(curandSetStream(dense_generator, stream));
        Backend<T>::random(dense_generator, dense_A, size_t(c.m) * c.k);
        Backend<T>::random(dense_generator, dense_B, size_t(c.k) * c.n);
        CURAND_CHECK(curandDestroyGenerator(dense_generator));
        auto dense_operation = [&] { Backend<T>::dense(dense_handle, c, dense_A, dense_B, dense_C); };
        auto dense_reset = [&] { fill_device(stream, dense_C, dense_elements, T(1)); };
        const Timing dense = measure_time_ms(dense_operation, dense_reset, stream,
                                             c.warmup, c.reps);
        const double dense_flops = gemm_flops<T>(c.m, c.n, c.k);
        const double tlr_flops = double(mt) * nt * kt *
                                 lll_tile_flops<T>(c.b, c.rank_A, c.rank_B, c.rank_C);
        std::printf("lll,%s,%d,%d,%d,%d,%d,%d,%d,%.12g,%d,%d,%.9f,%.9f,%.9f,%.9f,%.12g,%.12g,%.12g\n",
                    Backend<T>::name(), c.m, mt, c.b, c.rank_A, c.rank_B,
                    c.rank_C, final_rank, double(c.rank_A) / c.b,
                    c.warmup, c.reps, tlr.median_ms, tlr.minimum_ms,
                    dense.median_ms, dense.minimum_ms, tlr_flops, dense_flops,
                    dense_flops / tlr_flops);
        cudaFree(dense_A); cudaFree(dense_B); cudaFree(dense_C);
    } else {
        Backend<T>::lld_workspace(handle, c);
        KBLAS_CHECK(kblasAllocateWorkspace(handle));
        auto operation = [&] {
            KBLAS_CHECK(kblas_gemm_tlr(
                handle, KBLAS_NoTrans, KBLAS_NoTrans,
                mt, nt, kt, c.b, c.b, c.b, T(1),
                as_const_ptrs(A.u_ptrs), c.b,
                as_const_ptrs(A.v_ptrs), c.b, mt, c.rank_A,
                as_const_ptrs(B.u_ptrs), c.b,
                as_const_ptrs(B.v_ptrs), c.b, kt, c.rank_B,
                T(1), dense_C, c.m));
        };
        auto reset = [&] { fill_device(stream, dense_C, dense_elements, T(1)); };
        const Timing tlr = measure_time_ms(operation, reset, stream,
                                           c.warmup, c.reps);
        KBLAS_CHECK(kblasFreeWorkspace(handle));
        release(A); release(B);

        T *dense_A = nullptr, *dense_B = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dense_A),
                              size_t(c.m) * c.k * sizeof(T)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dense_B),
                              size_t(c.k) * c.n * sizeof(T)));
        curandGenerator_t dense_generator;
        CURAND_CHECK(curandCreateGenerator(&dense_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(dense_generator, 314159ULL));
        CURAND_CHECK(curandSetStream(dense_generator, stream));
        Backend<T>::random(dense_generator, dense_A, size_t(c.m) * c.k);
        Backend<T>::random(dense_generator, dense_B, size_t(c.k) * c.n);
        CURAND_CHECK(curandDestroyGenerator(dense_generator));
        auto dense_operation = [&] { Backend<T>::dense(dense_handle, c, dense_A, dense_B, dense_C); };
        auto dense_reset = [&] { fill_device(stream, dense_C, dense_elements, T(1)); };
        const Timing dense = measure_time_ms(dense_operation, dense_reset, stream,
                                             c.warmup, c.reps);
        const double dense_flops = gemm_flops<T>(c.m, c.n, c.k);
        const double tlr_flops = double(mt) * nt * kt *
                                 lld_tile_flops<T>(c.b, c.rank_A, c.rank_B);
        std::printf("lld,%s,%d,%d,%d,%d,%d,%d,%d,%.12g,%d,%d,%.9f,%.9f,%.9f,%.9f,%.12g,%.12g,%.12g\n",
                    Backend<T>::name(), c.m, mt, c.b, c.rank_A, c.rank_B,
                    0, 0, double(c.rank_A) / c.b, c.warmup, c.reps,
                    tlr.median_ms, tlr.minimum_ms, dense.median_ms,
                    dense.minimum_ms, tlr_flops, dense_flops,
                    dense_flops / tlr_flops);
        cudaFree(dense_A); cudaFree(dense_B); cudaFree(dense_C);
    }

    KBLAS_CHECK(kblasDestroy(&handle));
    if (lll) magma_finalize();
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 11) {
        std::fprintf(stderr,
            "usage: %s lld|lll m k n tile_size rank_A rank_B rank_C warmup reps\n",
            argv[0]);
        return 2;
    }
    Config c{
        std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
        std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]),
        std::atoi(argv[8]), std::atoi(argv[9]), std::atoi(argv[10]),
        std::string(argv[1]) == "lll" ? Mode::LLL : Mode::LLD};
    if (std::string(argv[1]) != "lld" && std::string(argv[1]) != "lll") {
        std::fprintf(stderr, "mode must be lld or lll\n");
        return 2;
    }
#ifdef BENCH_FLOAT
    return run<float>(c);
#else
    return run<double>(c);
#endif
}
