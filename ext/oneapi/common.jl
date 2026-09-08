const oneMKL = oneAPI.oneMKL
# oneAPI.Support holds the batched solver and TRSM entry points. Nothing in
# the GEMM layer uses it -- gemm goes through oneMKL above -- so it arrives
# with the first feature that does.
const support = oneAPI.Support

@inline NextLA.SUBGROUP_SIZE(::Type{<:oneAPI.oneAPIBackend}) = Val(32)

# TRSM entry points, added with src/trsm/batched.jl. oneMKL has both a
# pointer-batched and a strided-batched TRSM.
@inline _onemkl_trsm_fname(::Type{Float32}, ::Val{:pointer}) = support.onemklStrsm_batch
@inline _onemkl_trsm_fname(::Type{Float64}, ::Val{:pointer}) = support.onemklDtrsm_batch
@inline _onemkl_trsm_fname(::Type{ComplexF32}, ::Val{:pointer}) = support.onemklCtrsm_batch
@inline _onemkl_trsm_fname(::Type{ComplexF64}, ::Val{:pointer}) = support.onemklZtrsm_batch
@inline _onemkl_trsm_fname(::Type{Float32}, ::Val{:strided}) = support.onemklStrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{Float64}, ::Val{:strided}) = support.onemklDtrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{ComplexF32}, ::Val{:strided}) = support.onemklCtrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{ComplexF64}, ::Val{:strided}) = support.onemklZtrsm_batch_strided
