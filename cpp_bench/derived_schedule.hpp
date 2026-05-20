// derived_schedule.hpp — Processor-agnostic 2.5D schedule from TeX + SC'21.
//
// AUDIT / PDF PINS (update when SC_factorization.pdf is checked in locally):
//   * Kwasniewski et al., "Parallel matrix multiplication based on sparse
//     factorizations beyond tensors", SC'21 (Conflux).  Pin block-cyclic
//     assignment, replication depth, and processor grid figures here once
//     section/figure numbers are confirmed from the PDF.
//   * Non-square first layer P₁ = p_r × p_c: when P₁ is not a perfect square,
//     we pick a factor pair (p_r, p_c) minimizing |log(p_c / p_r)| (balanced
//     aspect ratio).  Cross-check with SC_factorization.pdf when available.
//   * In-repo derivation: NextLA.jl/qr_schur_xpartition.tex
//       – §A3aprime  (block map π(I,J), replication)
//       – §A3derive  (c_mem = floor(P·M/N²), c_cap = floor(P^{1/3}), V = c)
//       – §A3b       (admissible panel width b: c ≤ b ≤ N/√P₁), P₁ = p_r p_c
//       – §A1-lookahead, §A1-IR (Phase Q5/Q6; MP / IR policy in TeX)
//
// This header is host-only C++ (no CUDA).  Benches include it for rank-0
// printouts, asserts, and optional default grid / block size when the TeX
// budget M is supplied or computed (see nextla_fast_memory.hpp).

#ifndef DERIVED_SCHEDULE_HPP
#define DERIVED_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

struct DerivedSchedule {
    int P = 0;
    int N = 0;
    // TeX fast-memory budget "M" (qr_schur_xpartition.tex): capacity counting primary
    // matrix storage elements of size σ (8 B FP64 paths, 4 B fp32full). Same units as N² in c=PM/N².
    std::int64_t M_fp64_words = 0;  // legacy field name; 0 = auto from device + matrix mode in benches

    int c = 1;       // replication depth P_z = V
    int P1 = 0;      // P / c  (= p_r * p_c)
    int pr = 0, pc = 0;  // first-layer grid; may be rectangular (pr <= pc, pr * pc == P1)
    int px = 0, py = 0, pz = 0;  // process grid [px, py, pz] with px=pr, py=pc convention in benches

    int b = 0;       // panel block size (after rounding)
    int passes_default = 3;
    bool lookahead_default = true;

    bool ok = false;
    std::string error;

    // True when P₁ is a perfect square and we use a square p_r = p_c = √P₁ grid.
    bool p1_is_square() const { return pr > 0 && pr == pc && pr * pc == P1; }
    bool degenerate_p1() const { return P1 == 1; }  // row-slab + full replication specialization (TeX)
};

inline int floor_c_cap(int P) {
    if (P <= 0) return 0;
    return (int)std::floor(std::cbrt((double)P));
}

// c_mem = floor(P * M / N^2) from §A3derive (M in FP64 words).
inline std::int64_t floor_c_mem(std::int64_t P, std::int64_t M_fp64_words, std::int64_t N) {
    if (N <= 0 || M_fp64_words <= 0 || P <= 0) return 0;
    const std::int64_t N2 = N * N;
    return M_fp64_words * P / N2;
}

// For P₁ = p_r * p_c, choose factors with p_r <= p_c minimizing |log(p_c / p_r)|.
// If P₁ is a perfect square, returns p_r = p_c = √P₁.
inline void balanced_factors_p1(int P1, int* pr_out, int* pc_out) {
    if (P1 <= 0) {
        *pr_out = *pc_out = 0;
        return;
    }
    int s = (int)std::sqrt((double)P1);
    if (s * s == P1) {
        *pr_out = *pc_out = s;
        return;
    }
    int best_r = 1, best_c = P1;
    double best = 1e300;
    for (int r = 1; (long long)r * r <= P1; ++r) {
        if (P1 % r != 0) continue;
        int p = P1 / r;
        int rr = r, pp = p;
        if (rr > pp) std::swap(rr, pp);
        double score = std::abs(std::log((double)pp / (double)rr));
        if (score < best) {
            best = score;
            best_r = rr;
            best_c = pp;
        }
    }
    *pr_out = best_r;
    *pc_out = best_c;
}

// Default b: min(floor(√M), floor(N/√P₁)) rounded down to largest multiple of c still ≥ c;
// P₁ = p_r * p_c, √P₁ = √(p_r p_c).
inline int default_block_b(std::int64_t M_fp64_words, int N, int pr, int pc, int c) {
    if (N <= 0 || pr <= 0 || pc <= 0 || c <= 0) return 0;
    long long P1ll = (long long)pr * (long long)pc;
    if (P1ll <= 0) return 0;
    double sqrtP1 = std::sqrt((double)P1ll);
    int b_mem = (M_fp64_words > 0) ? (int)std::floor(std::sqrt((double)M_fp64_words)) : 0;
    int b_geom = (int)std::floor((double)N / sqrtP1);
    int b0 = (b_mem > 0) ? std::min(b_mem, b_geom) : b_geom;
    if (b0 < c) b0 = c;
    int b = (b0 / c) * c;
    if (b < c) b = c;
    return b;
}

// §A3b window: c ≤ b ≤ N / √(p_r p_c).
inline bool b_in_window(int b, int c, int N, int pr, int pc) {
    if (b <= 0 || c <= 0 || N <= 0 || pr <= 0 || pc <= 0) return false;
    double sqrtP1 = std::sqrt((double)pr * (double)pc);
    if (sqrtP1 <= 0.0) return false;
    int hi = (int)std::floor((double)N / sqrtP1);
    return b >= c && b <= hi;
}

// Search c downward from c_top: first c dividing P such that balanced (p_r,p_c) for P₁=P/c
// yields a default b inside §A3b. Returns 0 if none; sets *pr_out,*pc_out when return > 0.
inline int pick_replication_with_grid(int P, int c_top, std::int64_t M_fp64_words, int N,
                                       int* pr_out, int* pc_out) {
    if (P <= 0) return 0;
    int hi = std::max(1, std::min(c_top, P));
    for (int c = hi; c >= 1; --c) {
        if (P % c != 0) continue;
        int P1 = P / c;
        int pr, pc;
        balanced_factors_p1(P1, &pr, &pc);
        if (pr * pc != P1) continue;
        int b = default_block_b(M_fp64_words, N, pr, pc, c);
        if (b <= 0) continue;
        if (!b_in_window(b, c, N, pr, pc)) continue;
        *pr_out = pr;
        *pc_out = pc;
        return c;
    }
    return 0;
}

// Legacy: first c with P₁ a perfect square only (kept for callers that need old behaviour).
inline int pick_replication_c(int P, int c_top) {
    int pr, pc;
    if (P <= 0) return 0;
    int hi = std::max(1, std::min(c_top, P));
    for (int c = hi; c >= 1; --c) {
        if (P % c != 0) continue;
        int P1 = P / c;
        int s = (int)std::sqrt((double)P1);
        if (s * s != P1) continue;
        balanced_factors_p1(P1, &pr, &pc);
        if (pr == pc && pr * pc == P1) return c;
    }
    return 0;
}

// strict_no_fallback: if no (c,p_r,p_c) fits §A3b, set ok=false instead of legacy 1×P×1.
inline DerivedSchedule compute_derived_schedule(int P, int N, std::int64_t M_fp64_words,
                                                  bool strict_no_fallback = false) {
    DerivedSchedule D;
    D.P = P;
    D.N = N;
    D.M_fp64_words = M_fp64_words;
    if (P <= 0 || N <= 0) {
        D.error = "P and N must be positive";
        return D;
    }
    int c_cap = floor_c_cap(P);
    std::int64_t c_mem = (M_fp64_words > 0) ? floor_c_mem(P, M_fp64_words, N) : 0;
    int c_top = c_cap;
    if (M_fp64_words > 0) {
        int cm = (int)std::min<std::int64_t>(c_mem, (std::int64_t)c_cap);
        c_top = std::max(1, cm);
    }
    if (c_top < 1) c_top = 1;

    int pr = 0, pc = 0;
    int c = pick_replication_with_grid(P, c_top, M_fp64_words, N, &pr, &pc);
    if (c == 0) {
        if (strict_no_fallback) {
            D.ok = false;
            D.error = "no faithful (c,p_r,p_c) grid in search window (use --smoke or explicit --px/--py/--pz, or relax --strict-derived-grid)";
            return D;
        }
        // Legacy 1 × P × 1 row partition.
        D.c = P;
        D.P1 = 1;
        D.pr = D.pc = 1;
        D.px = 1;
        D.py = P;
        D.pz = 1;
        D.b = default_block_b(M_fp64_words, N, 1, 1, D.c);
        if (D.b == 0) D.b = std::min(N, 512);
        D.ok = b_in_window(D.b, D.c, N, 1, 1);
        if (!D.ok) D.error = "fallback 1×P×1: b outside §A3b window";
        else D.error.clear();
        return D;
    }
    D.c = c;
    D.P1 = P / c;
    D.pr = pr;
    D.pc = pc;
    D.px = D.pr;
    D.py = D.pc;
    D.pz = D.c;
    D.b = default_block_b(M_fp64_words, N, D.pr, D.pc, D.c);
    if (D.b == 0) D.b = std::min(N, std::max(D.c, 256));
    if (!b_in_window(D.b, D.c, N, D.pr, D.pc)) {
        std::ostringstream oss;
        oss << "Derived b=" << D.b << " outside §A3b window";
        D.error = oss.str();
        D.ok = false;
        return D;
    }
    D.ok = true;
    return D;
}

inline bool derived_grid_matches_manual(int P, int N, std::int64_t M_fp64_words,
                                        int px, int py, int pz) {
    if (M_fp64_words <= 0) return true;
    DerivedSchedule D = compute_derived_schedule(P, N, M_fp64_words, false);
    if (!D.ok) return false;
    return D.px == px && D.py == py && D.pz == pz;
}

inline std::string format_derived_schedule(const DerivedSchedule& D) {
    std::ostringstream o;
    o << "DerivedSchedule: P=" << D.P << " N=" << D.N << " M_elements=" << D.M_fp64_words
      << " => c=V=" << D.c << " P1=" << D.P1 << " pr=" << D.pr << " pc=" << D.pc
      << " grid[px,py,pz]=[" << D.px << "," << D.py << "," << D.pz << "]"
      << " b=" << D.b << " P1_square=" << (D.p1_is_square() ? "yes" : "no")
      << " degenerate_P1=" << (D.degenerate_p1() ? "yes" : "no");
    if (!D.ok) o << " ERROR: " << D.error;
    return o.str();
}

// 1D "c = P" benches (householder / givens / qdwh): same TeX quantities in degenerate form.
inline DerivedSchedule compute_degenerate_1d_schedule(int P, int N, std::int64_t M_fp64_words) {
    DerivedSchedule D;
    D.P = P;
    D.N = N;
    D.M_fp64_words = M_fp64_words;
    D.c = P;
    D.P1 = 1;
    D.pr = D.pc = 1;
    D.px = 1;
    D.py = P;
    D.pz = 1;
    D.b = default_block_b(M_fp64_words, N, 1, 1, P);
    if (D.b == 0) D.b = std::min(N, 512);
    D.ok = b_in_window(D.b, D.c, N, 1, 1);  // hi = N
    D.passes_default = 3;
    D.lookahead_default = true;
    if (!D.ok) D.error = "b window fail for 1D degenerate schedule";
    return D;
}

#endif  // DERIVED_SCHEDULE_HPP
