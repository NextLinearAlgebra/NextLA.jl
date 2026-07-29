// KBLAS half of the NextLA-vs-KBLAS TLR accumulation benchmark.
//
// This executable writes the exact input factors, KBLAS output factors, rank,
// and timing for each case.  The Julia half then consumes those files, so no
// Julia C++ FFI or KBLAS recompilation is needed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <magma_v2.h>
#include "kblas.h"
#include "kblas_tlr.h"

namespace fs = std::filesystem;

constexpr uint32_t MAGIC = 0x4e4b544cU; // "NKTL"
constexpr uint32_t VERSION = 1;
constexpr int WARMUP = 3;
constexpr int NREPS = 10;

enum Profile : int32_t { Uniform = 0, ARowSkew = 1, BColumnSkew = 2 };

struct Config { int b, nt, r, rC; };
struct Header {
    uint32_t magic, version;
    int32_t b, nt, r, rC, profile, beta, kblas_rank;
    double kblas_ms;
};

inline void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}
inline void check_kblas(int status, const char *what) {
    if (status != KBLAS_Success) {
        std::fprintf(stderr, "KBLAS %s failed: %d (%s)\n", what, status, kblasGetErrorString(status));
        std::exit(1);
    }
}

int profile_rank(Profile profile, char operand, int i, int j, int r) {
    const int low = std::max(1, r / 4);
    if (profile == ARowSkew && operand == 'A') return (i % 2 == 0) ? low : r;
    if (profile == BColumnSkew && operand == 'B') return (j % 2 == 0) ? low : r;
    return r;
}

// Deterministic thin orthogonal factors, generated on the CPU in Float64 then
// stored as Float32.  Julia reads these exact stored values from the result file.
// The hash deliberately depends nonlinearly on (row, column), unlike a single
// sine wave (which would span only two dimensions as rank grows).
inline double seeded_unit(int seed, int which, int row, int col) {
    uint32_t x = uint32_t(seed) ^ (0x9e3779b9U * uint32_t(which + 1)) ^
                 (0x85ebca6bU * uint32_t(row + 1)) ^ (0xc2b2ae35U * uint32_t(col + 1));
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return 2.0 * (double(x) / 4294967295.0) - 1.0;
}

void factor_pair(float *U, float *V, int b, int width, int rank, int seed) {
    std::fill(U, U + size_t(b) * width, 0.0f);
    std::fill(V, V + size_t(b) * width, 0.0f);
    std::vector<double> qU(size_t(b) * rank), qV(size_t(b) * rank);
    for (int which = 0; which < 2; ++which) {
        auto &q = which == 0 ? qU : qV;
        for (int c = 0; c < rank; ++c) {
            for (int row = 0; row < b; ++row)
                q[row + size_t(c) * b] = seeded_unit(seed, which, row, c);
            for (int p = 0; p < c; ++p) {
                double dot = 0.0;
                for (int row = 0; row < b; ++row) dot += q[row + size_t(p) * b] * q[row + size_t(c) * b];
                for (int row = 0; row < b; ++row) q[row + size_t(c) * b] -= dot * q[row + size_t(p) * b];
            }
            double norm = 0.0;
            for (int row = 0; row < b; ++row) norm += q[row + size_t(c) * b] * q[row + size_t(c) * b];
            norm = std::sqrt(norm);
            for (int row = 0; row < b; ++row) q[row + size_t(c) * b] /= norm;
        }
    }
    for (int c = 0; c < rank; ++c) {
        const double scale = std::sqrt(rank == 1 ? 1.0 : 1.0 - 0.5 * c / double(rank - 1));
        for (int row = 0; row < b; ++row) {
            U[row + size_t(c) * b] = float(qU[row + size_t(c) * b] * scale);
            V[row + size_t(c) * b] = float(qV[row + size_t(c) * b] * scale);
        }
    }
}

void fill_operand(std::vector<float> &U, std::vector<float> &V, const Config &cfg,
                  Profile profile, char operand, int seed, int capacity, int c_rank) {
    const size_t stride = size_t(cfg.b) * capacity;
    for (int j = 0; j < cfg.nt; ++j) for (int i = 0; i < cfg.nt; ++i) {
        const int logical = i + j * cfg.nt;
        const int rank = operand == 'C' ? c_rank : profile_rank(profile, operand, i, j, cfg.r);
        factor_pair(U.data() + size_t(logical) * stride, V.data() + size_t(logical) * stride,
                    cfg.b, capacity, rank, seed + 101 * logical);
    }
}

template <typename T>
void write_vector(FILE *f, const std::vector<T> &x) {
    if (std::fwrite(x.data(), sizeof(T), x.size(), f) != x.size()) {
        std::perror("writing benchmark record"); std::exit(1);
    }
}

template <typename T>
void write_scalar(FILE *f, const T &x) {
    if (std::fwrite(&x, sizeof(T), 1, f) != 1) {
        std::perror("writing benchmark scalar"); std::exit(1);
    }
}

void write_record(const fs::path &path, const Header &h,
                  const std::vector<float> &au, const std::vector<float> &av,
                  const std::vector<float> &bu, const std::vector<float> &bv,
                  const std::vector<float> &cu0, const std::vector<float> &cv0,
                  const std::vector<float> &cu, const std::vector<float> &cv) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    // Write fields individually: the Julia reader must not depend on C++ ABI
    // padding before the final Float64 timing value.
    write_scalar(f, h.magic); write_scalar(f, h.version);
    write_scalar(f, h.b); write_scalar(f, h.nt); write_scalar(f, h.r); write_scalar(f, h.rC);
    write_scalar(f, h.profile); write_scalar(f, h.beta); write_scalar(f, h.kblas_rank);
    write_scalar(f, h.kblas_ms);
    write_vector(f, au); write_vector(f, av); write_vector(f, bu); write_vector(f, bv);
    write_vector(f, cu0); write_vector(f, cv0); write_vector(f, cu); write_vector(f, cv);
    std::fclose(f);
}

struct DeviceFactors {
    float *u = nullptr, *v = nullptr;
    float **uptrs = nullptr, **vptrs = nullptr;
};

DeviceFactors upload(const std::vector<float> &u, const std::vector<float> &v,
                     int b, int width, int nt) {
    DeviceFactors out;
    const size_t bytes = u.size() * sizeof(float);
    check_cuda(cudaMalloc(&out.u, bytes), "allocating factor U");
    check_cuda(cudaMalloc(&out.v, bytes), "allocating factor V");
    check_cuda(cudaMemcpy(out.u, u.data(), bytes, cudaMemcpyHostToDevice), "uploading factor U");
    check_cuda(cudaMemcpy(out.v, v.data(), bytes, cudaMemcpyHostToDevice), "uploading factor V");
    std::vector<float *> huptrs(nt * nt), hvptrs(nt * nt);
    const size_t stride = size_t(b) * width;
    for (int logical = 0; logical < nt * nt; ++logical) {
        huptrs[logical] = out.u + size_t(logical) * stride;
        hvptrs[logical] = out.v + size_t(logical) * stride;
        if (huptrs[logical] != out.u + size_t(logical) * stride ||
            hvptrs[logical] != out.v + size_t(logical) * stride) {
            std::fprintf(stderr, "KBLAS logical pointer-table mapping failed\n"); std::exit(1);
        }
    }
    check_cuda(cudaMalloc(&out.uptrs, huptrs.size() * sizeof(float *)), "allocating U pointer table");
    check_cuda(cudaMalloc(&out.vptrs, hvptrs.size() * sizeof(float *)), "allocating V pointer table");
    check_cuda(cudaMemcpy(out.uptrs, huptrs.data(), huptrs.size() * sizeof(float *), cudaMemcpyHostToDevice), "uploading U pointers");
    check_cuda(cudaMemcpy(out.vptrs, hvptrs.data(), hvptrs.size() * sizeof(float *), cudaMemcpyHostToDevice), "uploading V pointers");
    return out;
}

void free_factors(DeviceFactors &x) {
    cudaFree(x.u); cudaFree(x.v); cudaFree(x.uptrs); cudaFree(x.vptrs);
}

double run_kblas(const Config &cfg, float beta, int initial_rank_c,
                 DeviceFactors &A, DeviceFactors &B, DeviceFactors &C,
                 const DeviceFactors &C0, int &final_rank) {
    kblasHandle_t handle;
    check_kblas(kblasCreate(&handle), "handle creation");
    // KBLAS's LLL compression uses right/upper batched TRMM, for which this
    // configured KBLAS build delegates to MAGMA.
    magma_init();
    check_kblas(kblasEnableMagma(handle), "enabling MAGMA support");
    check_kblas(kblasCreateStreams(handle, 2), "stream creation");
    cudaStream_t stream = kblasGetStream(handle);
    // The first k-tile grows kC from zero in beta=0, so query for the output
    // capacity rather than the initial active rank.
    kblasSgemm_tlr_lll_wsquery(handle, cfg.nt, cfg.nt, cfg.r, cfg.r,
                               cfg.rC, cfg.rC, cfg.b, cfg.b);
    check_kblas(kblasAllocateWorkspace(handle), "workspace allocation");
    const size_t cbytes = size_t(cfg.b) * cfg.rC * cfg.nt * cfg.nt * sizeof(float);
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "creating start event");
    check_cuda(cudaEventCreate(&stop), "creating stop event");
    std::vector<float> samples;
    for (int it = 0; it < WARMUP + NREPS; ++it) {
        check_cuda(cudaMemcpyAsync(C.u, C0.u, cbytes, cudaMemcpyDeviceToDevice, stream), "resetting C U");
        check_cuda(cudaMemcpyAsync(C.v, C0.v, cbytes, cudaMemcpyDeviceToDevice, stream), "resetting C V");
        check_cuda(cudaStreamSynchronize(stream), "synchronizing C reset");
        int kC = initial_rank_c;
        if (it >= WARMUP) check_cuda(cudaEventRecord(start, stream), "recording start event");
        check_kblas(kblas_gemm_tlr(handle, KBLAS_NoTrans, KBLAS_NoTrans,
                                   cfg.nt, cfg.nt, cfg.nt, cfg.b, cfg.b, cfg.b,
                                   1.0f, A.uptrs, cfg.b, A.vptrs, cfg.b, cfg.nt, cfg.r,
                                   B.uptrs, cfg.b, B.vptrs, cfg.b, cfg.nt, cfg.r,
                                   beta, C.uptrs, cfg.b, C.vptrs, cfg.b, cfg.nt, kC,
                                   cfg.rC, 0.0), "TLR accumulation GEMM");
        if (it >= WARMUP) {
            check_cuda(cudaEventRecord(stop, stream), "recording stop event");
            check_cuda(cudaEventSynchronize(stop), "synchronizing timed GEMM");
            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, stop), "reading elapsed time");
            samples.push_back(ms);
            final_rank = kC;
        }
    }
    // Restore a materialized final result for serialization.
    check_cuda(cudaMemcpyAsync(C.u, C0.u, cbytes, cudaMemcpyDeviceToDevice, stream), "restoring C U");
    check_cuda(cudaMemcpyAsync(C.v, C0.v, cbytes, cudaMemcpyDeviceToDevice, stream), "restoring C V");
    int kC = initial_rank_c;
    check_kblas(kblas_gemm_tlr(handle, KBLAS_NoTrans, KBLAS_NoTrans,
                               cfg.nt, cfg.nt, cfg.nt, cfg.b, cfg.b, cfg.b,
                               1.0f, A.uptrs, cfg.b, A.vptrs, cfg.b, cfg.nt, cfg.r,
                               B.uptrs, cfg.b, B.vptrs, cfg.b, cfg.nt, cfg.r,
                               beta, C.uptrs, cfg.b, C.vptrs, cfg.b, cfg.nt, kC,
                               cfg.rC, 0.0), "final TLR accumulation GEMM");
    final_rank = kC;
    std::sort(samples.begin(), samples.end());
    const double median = 0.5 * (samples[NREPS / 2 - 1] + samples[NREPS / 2]);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    kblasFreeWorkspace(handle); kblasDestroy(&handle); magma_finalize();
    return median;
}

void run_case(const Config &cfg, Profile profile, int beta, const fs::path &outdir) {
    const size_t ab_size = size_t(cfg.b) * cfg.r * cfg.nt * cfg.nt;
    const size_t c_size = size_t(cfg.b) * cfg.rC * cfg.nt * cfg.nt;
    std::vector<float> au(ab_size), av(ab_size), bu(ab_size), bv(ab_size);
    std::vector<float> cu0(c_size, 0.0f), cv0(c_size, 0.0f), cu(c_size), cv(c_size);
    fill_operand(au, av, cfg, profile, 'A', 1000 + 10 * int(profile), cfg.r, cfg.r);
    fill_operand(bu, bv, cfg, profile, 'B', 2000 + 10 * int(profile), cfg.r, cfg.r);
    if (beta) fill_operand(cu0, cv0, cfg, profile, 'C', 3000 + 10 * int(profile), cfg.rC, cfg.r);
    DeviceFactors dA = upload(au, av, cfg.b, cfg.r, cfg.nt);
    DeviceFactors dB = upload(bu, bv, cfg.b, cfg.r, cfg.nt);
    DeviceFactors dC0 = upload(cu0, cv0, cfg.b, cfg.rC, cfg.nt);
    DeviceFactors dC = upload(cu0, cv0, cfg.b, cfg.rC, cfg.nt);
    int kblas_rank = 0;
    const double kblas_ms = run_kblas(cfg, beta ? 1.0f : 0.0f, beta ? cfg.r : 0,
                                      dA, dB, dC, dC0, kblas_rank);
    check_cuda(cudaMemcpy(cu.data(), dC.u, c_size * sizeof(float), cudaMemcpyDeviceToHost), "downloading KBLAS U");
    check_cuda(cudaMemcpy(cv.data(), dC.v, c_size * sizeof(float), cudaMemcpyDeviceToHost), "downloading KBLAS V");
    Header h{MAGIC, VERSION, cfg.b, cfg.nt, cfg.r, cfg.rC, int(profile), beta, kblas_rank, kblas_ms};
    const char *profile_name = profile == Uniform ? "uniform" : profile == ARowSkew ? "a_row_skew" : "b_column_skew";
    char filename[256];
    std::snprintf(filename, sizeof(filename), "b%d_nt%d_r%d_%s_beta%d.bin", cfg.b, cfg.nt, cfg.r, profile_name, beta);
    write_record(outdir / filename, h, au, av, bu, bv, cu0, cv0, cu, cv);
    std::printf("KBLAS b=%d nt=%d r=%d profile=%s beta=%d: %.3f ms, kC=%d\n",
                cfg.b, cfg.nt, cfg.r, profile_name, beta, kblas_ms, kblas_rank);
    free_factors(dA); free_factors(dB); free_factors(dC0); free_factors(dC);
}

int main(int argc, char **argv) {
    fs::path outdir = "scripts/.kblas/results";
    bool smoke = false;
    std::vector<int> requested_sizes;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) outdir = argv[++i];
        else if (std::strcmp(argv[i], "--smoke") == 0) smoke = true;
        else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            std::string value = argv[++i];
            size_t begin = 0;
            while (begin < value.size()) {
                size_t end = value.find(',', begin);
                requested_sizes.push_back(std::stoi(value.substr(begin, end - begin)));
                if (end == std::string::npos) break;
                begin = end + 1;
            }
        }
        else { std::fprintf(stderr, "usage: %s [--output DIR] [--smoke]\n", argv[0]); return 2; }
    }
    fs::create_directories(outdir);
    std::vector<Config> configs = smoke ? std::vector<Config>{{32, 2, 2, 8}} :
        std::vector<Config>{{32,64,8,8}, {64,16,16,16}, {64,32,16,16},
                            {64,48,24,24}, {128,16,32,32}, {256,16,64,64}};
    if (!requested_sizes.empty()) {
        configs.clear();
        const int b = 512, r = 32, rC = 32;
        for (int n : requested_sizes) {
            if (n <= 0 || n % b != 0) {
                std::fprintf(stderr, "requested size %d must be positive and divisible by %d\n", n, b);
                return 2;
            }
            configs.push_back(Config{b, n / b, r, rC});
        }
    }
    for (const auto &cfg : configs)
        for (Profile profile : {Uniform, ARowSkew, BColumnSkew})
            for (int beta : {0, 1}) run_case(cfg, profile, beta, outdir);
    return 0;
}
