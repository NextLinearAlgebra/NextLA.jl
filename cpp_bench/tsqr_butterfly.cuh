// tsqr_butterfly.cuh — Shared scaffolding for butterfly recursive-halving
// panel reduction used by Path-h (TSQR) and Path-g (tournament-Givens).
//
// Mathematical contract (Kwasniewski SC'21 + tex §Phase Q3_h, Q3_g):
//   For col_comm of size P_r = Px*Pz, the panel-row reduction is done
//   in log_2(P_r) butterfly stages.  At stage s:
//     partner(r) = r XOR (1 << s)
//   Each rank exchanges its b×b R block with partner via NCCL P2P,
//   stacks the two R's into a 2b × b buffer (deterministic
//   lower-rank-on-top convention), and runs the variant's per-stage
//   QR on the 2b × b stack.  Both ranks in a pair end up with the same
//   updated R (so no separate broadcast is needed at the end).
//
//   Per-rank bandwidth across all stages: 2 b^2 log_2(P_r) words.
//   Per-rank synchronization rounds: log_2(P_r).

#ifndef TSQR_BUTTERFLY_CUH
#define TSQR_BUTTERFLY_CUH

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdint>

// ceil(log2(n)) for n >= 1.
static inline int tsqr_butterfly_log2_ceil(int n) {
    int l = 0;
    int p = 1;
    while (p < n) { p <<= 1; ++l; }
    return l;
}

// Partner rank within col_comm at butterfly stage s.
static inline int tsqr_butterfly_partner(int col_rank, int s) {
    return col_rank ^ (1 << s);
}

// "Half" of stage-s pair this rank occupies (0 = top, 1 = bottom).
// Determines which b rows of the stage-s explicit Q^(s) (2b × b) this
// rank picks up when extracting its tree contribution G_self.
//
// In the butterfly: at stage s, pair members are r and r XOR (1<<s).
// Lower-numbered is "top half"; higher-numbered is "bottom half".
//   half = bit s of col_rank.
static inline int tsqr_butterfly_half(int col_rank, int s) {
    return (col_rank >> s) & 1;
}

// Send + Recv my b×b R with the partner.  Both ranks place their own R
// at d_R_self and receive partner's R into d_R_recv.  Uses NCCL group
// calls so the pair fuses into one transaction (no deadlock risk).
//
// Caller must serialize the comm stream with the comp stream before /
// after via cudaEventRecord + cudaStreamWaitEvent.
static inline ncclResult_t tsqr_butterfly_exchange(double* d_R_self,
                                                     double* d_R_recv,
                                                     int b,
                                                     int partner,
                                                     ncclComm_t nccl_col,
                                                     cudaStream_t s_comm) {
    ncclResult_t rc;
    rc = ncclGroupStart();
    if (rc != ncclSuccess) return rc;
    rc = ncclSend(d_R_self, (size_t)b * b, ncclDouble, partner, nccl_col, s_comm);
    if (rc != ncclSuccess) { ncclGroupEnd(); return rc; }
    rc = ncclRecv(d_R_recv, (size_t)b * b, ncclDouble, partner, nccl_col, s_comm);
    if (rc != ncclSuccess) { ncclGroupEnd(); return rc; }
    return ncclGroupEnd();
}

// Stack two b×b R blocks into a 2b × b column-major buffer.  Convention:
// lower-numbered rank's R on TOP (rows [0,b)); higher-numbered on
// BOTTOM (rows [b,2b)).  Both ranks build the same stacked layout
// regardless of which side they came from.
//
// d_R_self: this rank's R  (b × b)
// d_R_partner: partner's R (b × b)
// d_stacked: output (2b × b, ld = 2b)
__global__ static void tsqr_butterfly_stack_kernel(const double* __restrict__ R_top,
                                                     const double* __restrict__ R_bot,
                                                     double* __restrict__ stacked,
                                                     int b) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)b * b;
    if (idx >= total) return;
    int i = (int)(idx % b);  // 0..b-1
    int j = (int)(idx / b);  // 0..b-1
    // stacked[i,        j] = R_top[i, j]
    // stacked[i + b,    j] = R_bot[i, j]
    stacked[(size_t)i        + (size_t)j * 2 * b] = R_top[(size_t)i + (size_t)j * b];
    stacked[(size_t)(i + b)  + (size_t)j * 2 * b] = R_bot[(size_t)i + (size_t)j * b];
}

// Convenience wrapper: chooses lower-rank-on-top ordering then stacks.
// my_rank vs partner_rank determines which R goes on top.
static inline void tsqr_butterfly_stack(const double* d_R_self,
                                          const double* d_R_partner,
                                          double* d_stacked,
                                          int b,
                                          int my_rank, int partner_rank,
                                          cudaStream_t s) {
    const double* top = (my_rank < partner_rank) ? d_R_self : d_R_partner;
    const double* bot = (my_rank < partner_rank) ? d_R_partner : d_R_self;
    int threads = 256;
    long long total = (long long)b * b;
    long long blocks = (total + threads - 1) / threads;
    tsqr_butterfly_stack_kernel<<<(unsigned)blocks, threads, 0, s>>>(top, bot, d_stacked, b);
}

// Copy the upper-triangular b × b block from the top of a column-major
// m × b matrix (m >= b) into a clean b × b buffer with strict-lower
// zeros.  Used to extract R from a just-computed Dgeqrf in-place
// factorization (the strict lower contains reflectors V).
__global__ static void tsqr_butterfly_copy_upper_R_kernel(const double* __restrict__ panel,
                                                            int lda,
                                                            double* __restrict__ R_out,
                                                            int b) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)b * b;
    if (idx >= total) return;
    int i = (int)(idx % b);
    int j = (int)(idx / b);
    if (i <= j) {
        R_out[(size_t)i + (size_t)j * b] = panel[(size_t)i + (size_t)j * lda];
    } else {
        R_out[(size_t)i + (size_t)j * b] = 0.0;
    }
}

static inline void tsqr_butterfly_copy_upper_R(const double* d_panel, int lda,
                                                 double* d_R_out, int b,
                                                 cudaStream_t s) {
    int threads = 256;
    long long total = (long long)b * b;
    long long blocks = (total + threads - 1) / threads;
    tsqr_butterfly_copy_upper_R_kernel<<<(unsigned)blocks, threads, 0, s>>>(d_panel, lda, d_R_out, b);
}

// Initialize a column-major m × n matrix to identity (1 on diagonal,
// 0 elsewhere).  Used by tournament-Givens to seed the G_self / local-Q
// replay buffers.
__global__ static void tsqr_butterfly_eye_kernel(double* __restrict__ M, int m, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m * n;
    if (idx >= total) return;
    int i = (int)(idx % m);
    int j = (int)(idx / m);
    M[idx] = (i == j) ? 1.0 : 0.0;
}

static inline void tsqr_butterfly_eye(double* d_M, int m, int n, cudaStream_t s) {
    int threads = 256;
    long long total = (long long)m * n;
    long long blocks = (total + threads - 1) / threads;
    tsqr_butterfly_eye_kernel<<<(unsigned)blocks, threads, 0, s>>>(d_M, m, n);
}

#endif  // TSQR_BUTTERFLY_CUH
