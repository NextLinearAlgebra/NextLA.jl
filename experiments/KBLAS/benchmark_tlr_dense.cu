// Standalone fixed-rank KBLAS TLR x TLR -> dense benchmark.
// The executable prints one CSV row for one (m,k,n,b,r) configuration.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "kblas.h"
#include "kblas_tlr.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); std::exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t e = (x); if (e != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error: %d\n", int(e)); std::exit(1); } } while (0)
#define CURAND_CHECK(x) do { curandStatus_t e = (x); if (e != CURAND_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuRAND error: %d\n", int(e)); std::exit(1); } } while (0)
#define KBLAS_CHECK(x) do { int e = (x); if (e != KBLAS_Success) { \
    std::fprintf(stderr, "KBLAS error: %d (%s)\n", e, kblasGetErrorString(e)); std::exit(1); } } while (0)

struct Config { int m, k, n, b, r, warmup, reps; };

template <typename T> struct Backend;
template <> struct Backend<float> {
    static const char *name() { return "Float32"; }
    static void random(curandGenerator_t g, float *p, size_t n)
    { CURAND_CHECK(curandGenerateNormal(g, p, n, 0.0f, 1.0f)); }
    static void dense(cublasHandle_t h, const Config &c, const float *A,
                      const float *B, float *C)
    { const float a = 1.0f, b = 1.0f; CUBLAS_CHECK(cublasSgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_N, c.m, c.n, c.k, &a, A, c.m, B, c.k, &b, C, c.m)); }
    static void workspace(kblasHandle_t h, const Config &c)
    { kblasSgemm_tlr_lld_wsquery(h, c.m / c.b, c.n / c.b, c.r, c.r, c.b, c.b); }
};
template <> struct Backend<double> {
    static const char *name() { return "Float64"; }
    static void random(curandGenerator_t g, double *p, size_t n)
    { CURAND_CHECK(curandGenerateNormalDouble(g, p, n, 0.0, 1.0)); }
    static void dense(cublasHandle_t h, const Config &c, const double *A,
                      const double *B, double *C)
    { const double a = 1.0, b = 1.0; CUBLAS_CHECK(cublasDgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_N, c.m, c.n, c.k, &a, A, c.m, B, c.k, &b, C, c.m)); }
    static void workspace(kblasHandle_t h, const Config &c)
    { kblasDgemm_tlr_lld_wsquery(h, c.m / c.b, c.n / c.b, c.r, c.r, c.b, c.b); }
};

template <typename T>
struct DeviceFactors {
    T *u = nullptr, *v = nullptr;
    T **uptrs = nullptr, **vptrs = nullptr;
};

template <typename T>
DeviceFactors<T> upload(const std::vector<T> &u, const std::vector<T> &v,
                        int b, int rank, int mt, int kt)
{
    DeviceFactors<T> out;
    const size_t bytes = u.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&out.u, bytes)); CUDA_CHECK(cudaMalloc(&out.v, bytes));
    CUDA_CHECK(cudaMemcpy(out.u, u.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.v, v.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<T *> hu(mt * kt), hv(mt * kt);
    const size_t stride = size_t(b) * rank;
    for (int j = 0; j < kt; ++j) for (int i = 0; i < mt; ++i) {
        const int slot = i + j * mt;
        hu[slot] = out.u + size_t(slot) * stride;
        hv[slot] = out.v + size_t(slot) * stride;
    }
    CUDA_CHECK(cudaMalloc(&out.uptrs, hu.size() * sizeof(T *)));
    CUDA_CHECK(cudaMalloc(&out.vptrs, hv.size() * sizeof(T *)));
    CUDA_CHECK(cudaMemcpy(out.uptrs, hu.data(), hu.size() * sizeof(T *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.vptrs, hv.data(), hv.size() * sizeof(T *), cudaMemcpyHostToDevice));
    return out;
}

template <typename T>
void release(DeviceFactors<T> &x)
{
    cudaFree(x.u); cudaFree(x.v); cudaFree(x.uptrs); cudaFree(x.vptrs);
}

template <typename Operation, typename T>
double time_operation(Operation operation, T *C, const T *C0,
                      size_t cbytes, cudaStream_t stream, int warmup, int reps)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    for (int it = 0; it < warmup + reps; ++it) {
        CUDA_CHECK(cudaMemcpyAsync(C, C0, cbytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (it >= warmup) CUDA_CHECK(cudaEventRecord(start, stream));
        operation();
        if (it >= warmup) {
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            samples.push_back(ms);
        }
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename T>
int run(const Config &c)
{
    if (c.m % c.b || c.k % c.b || c.n % c.b || c.r >= c.b) {
        std::fprintf(stderr, "dimensions must be tile-divisible and r < b\n"); return 2;
    }
    const int mt = c.m / c.b, kt = c.k / c.b, nt = c.n / c.b;
    const size_t countA = size_t(mt) * kt * c.b * c.r;
    const size_t countB = size_t(kt) * nt * c.b * c.r;
    const size_t denseA = size_t(c.m) * c.k, denseB = size_t(c.k) * c.n;
    const size_t denseC = size_t(c.m) * c.n;
    T *dA = nullptr, *dB = nullptr, *dC0 = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, denseA * sizeof(T))); CUDA_CHECK(cudaMalloc(&dB, denseB * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dC0, denseC * sizeof(T))); CUDA_CHECK(cudaMalloc(&dC, denseC * sizeof(T)));
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 20260728ULL));
    Backend<T>::random(gen, dA, denseA); Backend<T>::random(gen, dB, denseB);
    Backend<T>::random(gen, dC0, denseC);
    CURAND_CHECK(curandDestroyGenerator(gen));
    // The factors and dense operands are generated independently; both paths
    // are timed independently, as in the KBLAS reference benchmark.
    curandGenerator_t factors;
    CURAND_CHECK(curandCreateGenerator(&factors, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(factors, 314159ULL));
    T *dAu = nullptr, *dAv = nullptr, *dBu = nullptr, *dBv = nullptr;
    CUDA_CHECK(cudaMalloc(&dAu, countA * sizeof(T))); CUDA_CHECK(cudaMalloc(&dAv, countA * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dBu, countB * sizeof(T))); CUDA_CHECK(cudaMalloc(&dBv, countB * sizeof(T)));
    Backend<T>::random(factors, dAu, countA); Backend<T>::random(factors, dAv, countA);
    Backend<T>::random(factors, dBu, countB); Backend<T>::random(factors, dBv, countB);
    CURAND_CHECK(curandDestroyGenerator(factors));
    std::vector<const T *> hAu(mt * kt), hAv(mt * kt), hBu(kt * nt), hBv(kt * nt);
    const size_t stride = size_t(c.b) * c.r;
    for (int j = 0; j < kt; ++j) for (int i = 0; i < mt; ++i) {
        int s = i + j * mt; hAu[s] = dAu + size_t(s) * stride; hAv[s] = dAv + size_t(s) * stride;
    }
    for (int j = 0; j < nt; ++j) for (int i = 0; i < kt; ++i) {
        int s = i + j * kt; hBu[s] = dBu + size_t(s) * stride; hBv[s] = dBv + size_t(s) * stride;
    }
    const T **dAuPtrs = nullptr, **dAvPtrs = nullptr, **dBuPtrs = nullptr, **dBvPtrs = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dAuPtrs, hAu.size() * sizeof(T *))); CUDA_CHECK(cudaMalloc((void **)&dAvPtrs, hAv.size() * sizeof(T *)));
    CUDA_CHECK(cudaMalloc((void **)&dBuPtrs, hBu.size() * sizeof(T *))); CUDA_CHECK(cudaMalloc((void **)&dBvPtrs, hBv.size() * sizeof(T *)));
    CUDA_CHECK(cudaMemcpy(dAuPtrs, hAu.data(), hAu.size() * sizeof(T *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAvPtrs, hAv.data(), hAv.size() * sizeof(T *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBuPtrs, hBu.data(), hBu.size() * sizeof(T *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBvPtrs, hBv.data(), hBv.size() * sizeof(T *), cudaMemcpyHostToDevice));

    kblasHandle_t handle; KBLAS_CHECK(kblasCreate(&handle));
    Backend<T>::workspace(handle, c);
    KBLAS_CHECK(kblasAllocateWorkspace(handle));
    cudaStream_t stream = kblasGetStream(handle);
    cublasHandle_t dense_handle = kblasGetCublasHandle(handle);
    CUBLAS_CHECK(cublasSetStream(dense_handle, stream));
    auto tlr = [&] { KBLAS_CHECK(kblas_gemm_tlr(handle, 'N', 'N', mt, nt, kt,
        c.b, c.b, c.b, T(1), dAuPtrs, c.b, dAvPtrs, c.b, mt, c.r,
        dBuPtrs, c.b, dBvPtrs, c.b, kt, c.r, T(1), dC, c.m)); };
    auto dense = [&] { Backend<T>::dense(dense_handle, c, dA, dB, dC); };
    const double dense_ms = time_operation(dense, dC, dC0, denseC * sizeof(T), stream, c.warmup, c.reps);
    const double tlr_ms = time_operation(tlr, dC, dC0, denseC * sizeof(T), stream, c.warmup, c.reps);
    const double dense_flops = 2.0 * c.m * c.k * c.n;
    const double tlr_flops = 2.0 * mt * nt * kt * (2.0 * c.b * c.r * c.r + c.b * c.b * c.r);
    std::printf("%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", Backend<T>::name(),
        c.m, c.k, c.n, c.b, c.r, dense_ms, tlr_ms, dense_ms / tlr_ms,
        dense_flops / (dense_ms * 1.0e6), tlr_flops / (tlr_ms * 1.0e6));
    kblasFreeWorkspace(handle); kblasDestroy(&handle);
    cudaFree(dAuPtrs); cudaFree(dAvPtrs); cudaFree(dBuPtrs); cudaFree(dBvPtrs);
    cudaFree(dAu); cudaFree(dAv); cudaFree(dBu); cudaFree(dBv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC0); cudaFree(dC);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 8) { std::fprintf(stderr, "usage: %s m k n b r warmup reps\n", argv[0]); return 2; }
    Config c{std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),
             std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7])};
#ifdef BENCH_FLOAT
    return run<float>(c);
#else
    return run<double>(c);
#endif
}
