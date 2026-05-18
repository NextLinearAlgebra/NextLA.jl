// full25d_kernels.cuh — Device kernels shared by the full-2.5D variant
// runners (Householder, Givens, QDWH).  Static-inline so multiple .cu
// translation units may safely include this header without a link error.

#ifndef FULL25D_KERNELS_CUH
#define FULL25D_KERNELS_CUH

#include <cuda_runtime.h>

// Rearrange NCCL AllGather output (rank-block layout, P_col · m_loc · sb
// doubles) into a single column-major m_total × sb panel.
//   recv[r * m_loc * sb + j * m_loc + i_loc]  →  full[(r * m_loc + i_loc) + j * m_total]
__global__ static void f25d_rearrange_recv_to_panel(const double* __restrict__ recv,
                                                     double* __restrict__ full,
                                                     int m_loc, int sb, int P_col,
                                                     int m_total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)m_total * sb;
    if (idx >= total) return;
    int i = (int)(idx % m_total);
    int j = (int)(idx / m_total);
    int r = i / m_loc;
    int i_local = i - r * m_loc;
    full[idx] = recv[(long long)r * m_loc * sb + (long long)j * m_loc + i_local];
}

#endif  // FULL25D_KERNELS_CUH
