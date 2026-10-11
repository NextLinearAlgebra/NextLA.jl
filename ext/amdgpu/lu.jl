# LU leaf for NextLA.lu_recursive! and lu_recursive_nopiv! on AMDGPU.
#
# LinearAlgebra.lu! calls LAPACK.getrf!(A; check). AMDGPU 2.1.2 overrides only
# the keyword-free LAPACK.getrf!(A::ROCMatrix), which a keyword call skips, so
# lu! on a ROCMatrix falls into the generic getrf! and never reaches rocSOLVER.
# This calls rocSOLVER's getrf_npvt directly, the unpivoted factorisation the
# recursion needs (NoPivot() in lu! would fall back to scalar indexing).
# rocSOLVER takes a ROCMatrix only, and the recursion hands its leaves down as
# views, so a view is factored in a contiguous copy and written back: O(b^2)
# traffic against O(b^3) work. A zero pivot raises ZeroPivotException, as
# lu!(A, NoPivot()) does on the CPU.

const _ROCLUTypes = Union{Float32, Float64, ComplexF32, ComplexF64}

for (fname, elty) in ((:rocsolver_sgetrf_npvt, :Float32), (:rocsolver_dgetrf_npvt, :Float64),
                      (:rocsolver_cgetrf_npvt, :ComplexF32), (:rocsolver_zgetrf_npvt, :ComplexF64))
    @eval function _rocsolver_getrf_checked!(A::AMDGPU.ROCMatrix{$elty})
        m, n = size(A)
        devinfo = AMDGPU.ROCArray{Cint}(undef, 1)
        AMDGPU.rocSOLVER.$fname(AMDGPU.rocBLAS.handle(), m, n, A, max(1, stride(A, 2)), devinfo)
        info = AMDGPU.@allowscalar devinfo[1]
        AMDGPU.unsafe_free!(devinfo)
        info > 0 && throw(LinearAlgebra.ZeroPivotException(Int(info)))
        return A
    end
end

NextLA._lu_leaf!(A::AMDGPU.ROCMatrix{<:_ROCLUTypes}) = _rocsolver_getrf_checked!(A)

function NextLA._lu_leaf!(A::SubArray{T, 2, <:AMDGPU.ROCMatrix{T}}) where {T <: _ROCLUTypes}
    B = similar(parent(A), T, size(A))
    B .= A
    _rocsolver_getrf_checked!(B)
    A .= B
    return A
end
