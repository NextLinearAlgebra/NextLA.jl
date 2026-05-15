module cudaext

using NextLA

if isdefined(Base, :get_extension)
    import CUDA
else
    import ..CUDA
end

function NextLA.probe_device(backend::CUDA.CUDABackend, ::Type{T}) where {T}
	dev = CUDA.device()
	sm_count = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
	smem_bytes = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
	P = max(1, Int(sm_count))
	M = max(1, Int(smem_bytes) ÷ max(1, sizeof(T)))
	return P, M
end

function NextLA._scqr3_gram_backend_caps(backend::CUDA.CUDABackend, ::Type{T}) where {T}
	dev = CUDA.device()
	max_th = Int(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
	smem = Int(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
	return (max_th, smem)
end

# cuSOLVER POTRF for GPU Cholesky — avoids the single-thread serial KA kernel.
function NextLA._scqr3_potrf!(::CUDA.CUDABackend, G::AbstractMatrix, b::Int)
	CUDA.CUSOLVER.potrf!('U', view(G, 1:b, 1:b))
end

end
