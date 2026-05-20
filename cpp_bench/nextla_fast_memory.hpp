// TeX (qr_schur_xpartition.tex): each rank has fast memory of capacity M measured in
// *matrix storage elements* of size σ bytes (same counting units as N² in c = PM/N²).
//
//   M_elements = floor( (fraction × totalGlobalMem) / σ ).
//
// σ = 8 for FP64 / mixed-precision paths whose primary A and panel algebra are double,
// σ = 4 for --matrix=fp32full (entirely float resident set).
//
// fraction defaults to 0.72 (headroom for libraries, workspace, fragmentation).
// Override with env NEXTLA_FASTMEM_FRAC in (0,1].

#ifndef NEXTLA_FAST_MEMORY_HPP
#define NEXTLA_FAST_MEMORY_HPP

#include <algorithm>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime.h>

#include "matrix_mode.hpp"

inline std::size_t nextla_matrix_element_bytes(MatrixMode m) {
    return (m == MatrixMode::FP32_FULL) ? 4u : 8u;
}

inline double nextla_fastmem_frac_from_env() {
    double frac = 0.72;
    const char* ev = std::getenv("NEXTLA_FASTMEM_FRAC");
    if (ev && ev[0]) {
        char* end = nullptr;
        double v = std::strtod(ev, &end);
        if (v > 0.05 && v <= 1.0) frac = v;
    }
    return frac;
}

// TeX "M": fast-memory capacity counting σ-sized matrix elements (see qr_schur_xpartition.tex).
inline std::int64_t nextla_device_fast_memory_budget_elements(int cuda_device, MatrixMode m) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, cuda_device) != cudaSuccess) return 0;
    const double frac = nextla_fastmem_frac_from_env();
    const std::size_t sigma = nextla_matrix_element_bytes(m);
    const double usable = (double)prop.totalGlobalMem * frac;
    std::int64_t M = (std::int64_t)(usable / (double)sigma);
    return std::max(M, (std::int64_t)1);
}

#endif  // NEXTLA_FAST_MEMORY_HPP
