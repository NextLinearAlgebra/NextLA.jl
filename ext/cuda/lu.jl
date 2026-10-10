# LU leaf for NextLA.lu_recursive! and lu_recursive_nopiv! on CUDA: cuSOLVER's
# getrf without pivoting. CUDA.jl's getrf! always pivots, and
# LinearAlgebra.lu!(A, NoPivot()) falls back to scalar indexing on a CuArray, so
# this calls cusolverDn?getrf with a null pivot array, which cuSOLVER documents
# as "no pivoting", following CUDA.jl's own getrf! (lib/cusolver/dense.jl). A
# zero pivot throws ZeroPivotException, as lu!(A, NoPivot()) does on the CPU.

import LinearAlgebra

for (bname, fname, elty) in ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
                             (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64),
                             (:cusolverDnCgetrf_bufferSize, :cusolverDnCgetrf, :ComplexF32),
                             (:cusolverDnZgetrf_bufferSize, :cusolverDnZgetrf, :ComplexF64))
    @eval function NextLA._lu_leaf!(A::CUDA.StridedCuMatrix{$elty})
        m, n = size(A)
        lda = max(1, stride(A, 2))
        dh = CUSOLVER.dense_handle()
        function bufferSize()
            out = Ref{Cint}(0)
            CUSOLVER.$bname(dh, m, n, A, lda, out)
            return out[] * sizeof($elty)
        end
        CUSOLVER.with_workspace(dh.workspace_gpu, bufferSize) do buffer
            CUSOLVER.$fname(dh, m, n, A, lda, buffer, CuPtr{Cint}(0), dh.info)
        end
        info = CUDA.@allowscalar dh.info[1]
        info > 0 && throw(LinearAlgebra.ZeroPivotException(Int(info)))
        return A
    end
end
