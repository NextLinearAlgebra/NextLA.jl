module amdext

using NextLA

if isdefined(Base, :get_extension)
	import AMDGPU
else
	import ..AMDGPU
end

function NextLA.probe_device(backend::AMDGPU.ROCBackend, ::Type{T}) where {T}
	dev = AMDGPU.device()
	props = AMDGPU.properties(dev)
	sm_count = props.multiProcessorCount
	smem_bytes = props.sharedMemPerMultiprocessor
	P = Int(sm_count)
	M = max(1, Int(smem_bytes) ÷ max(1, sizeof(T)))
	return P, M
end

function NextLA._scqr3_gram_backend_caps(backend::AMDGPU.ROCBackend, ::Type{T}) where {T}
	dev = AMDGPU.device()
	props = AMDGPU.properties(dev)
	max_th = Int(props.maxThreadsPerBlock)
	smem = Int(props.sharedMemPerBlock)
	return (max_th, smem)
end

end
