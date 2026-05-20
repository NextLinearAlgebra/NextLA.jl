// Optional injected vendor medians for five-column METRICS (plan E1).
//
//   NEXTLA_VENDOR_METRICS_TABLE — optional file: each non-comment line "N P fp64_ms fp32_ms"
//       (P = MPI world size used for the vendor run, e.g. Px*Py for 2D grids).
//   NEXTLA_VENDOR_METRICS_PATH — optional single line "<fp64_ms> <fp32_ms>" (legacy).
//   NEXTLA_VENDOR_FP64_MS / NEXTLA_VENDOR_FP32_MS — override table/path when set.
//   nextla_read_vendor_ms_for_np(N, mpi_np) merges table + env (env wins).

#ifndef BENCH_VENDOR_METRICS_HPP
#define BENCH_VENDOR_METRICS_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
#define NEXTLA_BENCH_HAVE_CUBLAS_TF32 1
#else
#define NEXTLA_BENCH_HAVE_CUBLAS_TF32 0
#endif

struct NextlaVendorMs {
    bool have64 = false;
    bool have32 = false;
    double fp64 = 0.0;
    double fp32 = 0.0;
};

// Optional multi-row table (NEXTLA_VENDOR_METRICS_TABLE): lines
//   N P vendor_fp64_ms vendor_fp32_ms
// with integer N and P (=MPI size used for vendor run). First matching row wins.
// Falls back to NEXTLA_VENDOR_METRICS_PATH and NEXTLA_VENDOR_* env vars.
inline NextlaVendorMs nextla_read_vendor_ms_from_table(int N, int mpi_np) {
    NextlaVendorMs r;
    const char* tpath = std::getenv("NEXTLA_VENDOR_METRICS_TABLE");
    if (!tpath || !tpath[0])
        return r;
    FILE* fp = std::fopen(tpath, "r");
    if (!fp)
        return r;
    char line[512];
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        int n = 0, p = 0;
        double d64 = 0, d32 = 0;
        int got = std::sscanf(line, "%d %d %lf %lf", &n, &p, &d64, &d32);
        if (got < 3)
            continue;
        if (n != N || p != mpi_np)
            continue;
        if (d64 > 0.0) {
            r.fp64 = d64;
            r.have64 = true;
        }
        if (got >= 4 && d32 > 0.0) {
            r.fp32 = d32;
            r.have32 = true;
        }
        std::fclose(fp);
        return r;
    }
    std::fclose(fp);
    return r;
}

inline NextlaVendorMs nextla_read_vendor_ms_from_env() {
    NextlaVendorMs r;
    const char* path = std::getenv("NEXTLA_VENDOR_METRICS_PATH");
    if (path && path[0]) {
        FILE* fp = std::fopen(path, "r");
        if (fp) {
            double x = 0.0, y = 0.0;
            int n = std::fscanf(fp, "%lf %lf", &x, &y);
            if (n >= 1 && x > 0.0) {
                r.fp64 = x;
                r.have64 = true;
            }
            if (n >= 2 && y > 0.0) {
                r.fp32 = y;
                r.have32 = true;
            }
            std::fclose(fp);
        }
    }
    const char* e64 = std::getenv("NEXTLA_VENDOR_FP64_MS");
    if (e64 && e64[0]) {
        char* end = nullptr;
        double v = std::strtod(e64, &end);
        if (end != e64 && v > 0.0) {
            r.fp64 = v;
            r.have64 = true;
        }
    }
    const char* e32 = std::getenv("NEXTLA_VENDOR_FP32_MS");
    if (e32 && e32[0]) {
        char* end = nullptr;
        double v = std::strtod(e32, &end);
        if (end != e32 && v > 0.0) {
            r.fp32 = v;
            r.have32 = true;
        }
    }
    return r;
}

inline NextlaVendorMs nextla_read_vendor_ms_for_np(int N, int mpi_np) {
    NextlaVendorMs t = nextla_read_vendor_ms_from_table(N, mpi_np);
    NextlaVendorMs e = nextla_read_vendor_ms_from_env();
    NextlaVendorMs out = t;
    if (e.have64) {
        out.fp64 = e.fp64;
        out.have64 = true;
    }
    if (e.have32) {
        out.fp32 = e.fp32;
        out.have32 = true;
    }
    return out;
}

inline void nextla_fprint_metrics_vendor_columns(FILE* f, const NextlaVendorMs& v) {
    if (v.have64)
        std::fprintf(f, "vendor_fp64_ms=%.4f", v.fp64);
    else
        std::fprintf(f, "vendor_fp64_ms=NA");
    if (v.have32)
        std::fprintf(f, " vendor_fp32_ms=%.4f", v.fp32);
    else
        std::fprintf(f, " vendor_fp32_ms=NA");
}

inline void nextla_maybe_print_tf32_trailing_banner_rank0(int rank, bool use_tf32_trail) {
    if (rank != 0 || !use_tf32_trail)
        return;
#if NEXTLA_BENCH_HAVE_CUBLAS_TF32
    std::printf(
        "[bench] TF32 trailing GEMMs: ON (CUBLAS_COMPUTE_32F_FAST_TF32 / tensor ops where used).\n");
    std::fflush(stdout);
#else
    (void)use_tf32_trail;
#endif
}

#endif
