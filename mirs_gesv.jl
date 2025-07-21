using CUDA
using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using ..CUBLAS: unsafe_batch
using CUDA.CUSOLVER: CuSolverIRSParameters, CuSolverIRSInformation,
    cusolverDnIRSParamsSetSolverMainPrecision,
    cusolverDnIRSParamsSetSolverLowestPrecision,
    cusolverDnIRSParamsSetRefinementSolver,
    cusolverDnIRSParamsSetTol,
    cusolverDnIRSParamsSetTolInner,
    cusolverDnIRSParamsSetMaxIters,
    cusolverDnIRSParamsSetMaxItersInner,
    cusolverDnIRSParamsEnableFallback,
    cusolverDnIRSParamsDisableFallback,
    cusolverDnIRSInfosRequestResidual,
    cusolverDnIRSXgesv,
    cusolverDnIRSXgesv_bufferSize,
    cusolverDnIRSInfosGetResidualHistory 


using GPUArrays: @allowscalar
using LinearAlgebra: BlasFloat
using CUDA.CUSOLVER: with_workspace


function cusolverDnCreate()
  handle_ref = Ref{cusolverDnHandle_t}()
  cusolverDnCreate(handle_ref)
  return handle_ref[]
end
using CUDA.CUSOLVER

function dense_handle()
    return CUDA.CUSOLVER.dense_handle()
end


# gesv
function gesv!(X::CuVecOrMat{T}, A::CuMatrix{T}, B::CuVecOrMat{T}; fallback::Bool=true,
               residual_history::Bool=false, irs_precision::String, refinement_solver::String="CLASSICAL",
               maxiters::Int=0, maxiters_inner::Int=0, tol::Float64=0.0, tol_inner::Float64=0.0) where T <: BlasFloat

    params = CuSolverIRSParameters()
    info = CuSolverIRSInformation()
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldx = max(1, stride(X, 2))
    niters = Ref{Cint}()
    dh = dense_handle()

    # if irs_precision == "AUTO"
    #     (T == Float32)    && (irs_precision = "R_32F")
    #     (T == Float64)    && (irs_precision = "R_64F")
    #     (T == ComplexF32) && (irs_precision = "C_32F")
    #     (T == ComplexF64) && (irs_precision = "C_64F")
    # else
    #     (T == Float32)    && (irs_precision ∈ ("R_32F", "R_16F", "R_16BF", "R_TF32") || error("$irs_precision is not supported."))
    #     (T == Float64)    && (irs_precision ∈ ("R_64F", "R_32F", "R_16F", "R_16BF", "R_TF32") || error("$irs_precision is not supported."))
    #     (T == ComplexF32) && (irs_precision ∈ ("C_32F", "C_16F", "C_16BF", "C_TF32") || error("$irs_precision is not supported."))
    #     (T == ComplexF64) && (irs_precision ∈ ("C_64F", "C_32F", "C_16F", "C_16BF", "C_TF32") || error("$irs_precision is not supported."))
    # end
    cusolverDnIRSParamsSetSolverMainPrecision(params, T)
    cusolverDnIRSParamsSetSolverLowestPrecision(params, irs_precision)
    cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    (tol != 0.0) && cusolverDnIRSParamsSetTol(params, tol)
    (tol_inner != 0.0) && cusolverDnIRSParamsSetTolInner(params, tol_inner)
    (maxiters != 0) && cusolverDnIRSParamsSetMaxIters(params, maxiters)
    (maxiters_inner != 0) && cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    fallback ? cusolverDnIRSParamsEnableFallback(params) : cusolverDnIRSParamsDisableFallback(params)
    residual_history && cusolverDnIRSInfosRequestResidual(info)

    function bufferSize()
        buffer_size = Ref{Csize_t}(0)
        cusolverDnIRSXgesv_bufferSize(dh, params, n, nrhs, buffer_size)
        return buffer_size[]
    end

    with_workspace(dh.workspace_gpu, bufferSize) do buffer
        cusolverDnIRSXgesv(dh, params, info, n, nrhs, A, lda, B, ldb,
                           X, ldx, buffer, sizeof(buffer), niters, dh.info)
    end

    # Copy the solver flag and delete the device memory
    flag = @allowscalar dh.info[1]
    print(flag)


    chklapackerror(flag |> BlasInt)

    
    
    return X, info, niters[], flag

end

