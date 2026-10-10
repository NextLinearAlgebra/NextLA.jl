NextLA.create_streams(::AMDGPU.AMDGPUBackend, n::Int) = [AMDGPU.HIPStream() for _ in 1:n]

function NextLA.with_stream(f, ::AMDGPU.AMDGPUBackend, s::AMDGPU.HIPStream)
    AMDGPU.stream!(f, s)
end

NextLA.sync_stream(::AMDGPU.AMDGPUBackend, s::AMDGPU.HIPStream) = AMDGPU.synchronize(s)

function NextLA.sync_streams_with_default(::AMDGPU.AMDGPUBackend, streams)
    AMDGPU.synchronize()
end
