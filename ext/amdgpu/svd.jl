# Batched thin SVD on AMD.
#
# rocSOLVER's closest counterpart to cuSOLVER's `gesvdaStridedBatched` is the
# batched one-sided Jacobi SVD `rocsolver_?gesvdj_strided_batched`. Like
# `gesvda` it factors the tall-skinny panel directly, so it forms no Gram and
# does not square the condition number — which is the property
# `ara_truncate!` relies on.
#
# !!! warning "Unverified"
#     This path has never been executed: the AMDGPU extension does not
#     precompile on the development machine (`AMDGPUBackend` undefined in
#     AMDGPU, a pre-existing version mismatch), so the wrapper below is written
#     against the rocSOLVER signature but has no test coverage. Until it runs,
#     the generic looped-LAPACK method in `numerics/ara.jl` remains correct for
#     ROCArrays — it simply copies to the host. Treat a first run of this
#     method as bring-up, not as a regression.

for (jname, elty, rty) in ((:rocsolver_sgesvdj_strided_batched, :Float32, :Float32),
                           (:rocsolver_dgesvdj_strided_batched, :Float64, :Float64))
    @eval function NextLA.TLRmodule.batched_thin_svd!(A::ROCArray{$elty,3})
        m, n, count = size(A)
        m >= n ||
            throw(ArgumentError("batched_thin_svd! requires size(A,1) >= size(A,2)"))
        count == 0 && return (similar(A), similar(A, $rty, n, 0),
                              similar(A, n, n, count))

        U = similar(A, $elty, m, n, count)
        V = similar(A, $elty, n, n, count)
        S = similar(A, $rty, n, count)
        residual = similar(A, $rty, count)
        n_sweeps = similar(A, Int32, count)
        info = similar(A, Int32, count)

        # `abstol <= 0` asks rocSOLVER for machine precision; the truncation
        # tolerance is applied afterwards to the singular values, not here.
        AMDGPU.rocSOLVER.$jname(
            AMDGPU.rocBLAS.handle(),
            AMDGPU.rocSOLVER.rocblas_svect_singular,   # left: thin U
            AMDGPU.rocSOLVER.rocblas_svect_singular,   # right: thin V
            m, n, A, m, m * n,
            $rty(0), residual, Int32(100), n_sweeps,
            S, n, U, m, m * n, V, n, n * n,
            info, count,
        )
        return (U, S, V)
    end
end
