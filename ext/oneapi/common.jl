const oneMKL = oneAPI.oneMKL
const support = oneAPI.Support

@inline NextLA.SUBGROUP_SIZE(::Type{<:oneAPI.oneAPIBackend}) = Val(32)

@inline _onemkl_trsm_fname(::Type{Float32}, ::Val{:pointer}) = support.onemklStrsm_batch
@inline _onemkl_trsm_fname(::Type{Float64}, ::Val{:pointer}) = support.onemklDtrsm_batch
@inline _onemkl_trsm_fname(::Type{ComplexF32}, ::Val{:pointer}) = support.onemklCtrsm_batch
@inline _onemkl_trsm_fname(::Type{ComplexF64}, ::Val{:pointer}) = support.onemklZtrsm_batch
@inline _onemkl_trsm_fname(::Type{Float32}, ::Val{:strided}) = support.onemklStrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{Float64}, ::Val{:strided}) = support.onemklDtrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{ComplexF32}, ::Val{:strided}) = support.onemklCtrsm_batch_strided
@inline _onemkl_trsm_fname(::Type{ComplexF64}, ::Val{:strided}) = support.onemklZtrsm_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{Float32}) = support.onemklSpotrf_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{Float64}) = support.onemklDpotrf_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{ComplexF32}) = support.onemklCpotrf_batch_strided
@inline _onemkl_potrf_strided_fname(::Type{ComplexF64}) = support.onemklZpotrf_batch_strided
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{Float32}) = support.onemklSpotrf_batch_strided_scratchpad_size
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{Float64}) = support.onemklDpotrf_batch_strided_scratchpad_size
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{ComplexF32}) = support.onemklCpotrf_batch_strided_scratchpad_size
@inline _onemkl_potrf_strided_scratchpad_fname(::Type{ComplexF64}) = support.onemklZpotrf_batch_strided_scratchpad_size
