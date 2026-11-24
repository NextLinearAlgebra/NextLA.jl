NextLA.create_streams(::AMDGPU.ROCBackend, n::Int) = [AMDGPU.HIPStream() for _ in 1:n]

function NextLA.with_stream(f, ::AMDGPU.ROCBackend, s::AMDGPU.HIPStream)
    AMDGPU.stream!(f, s)
end

NextLA.sync_stream(::AMDGPU.ROCBackend, s::AMDGPU.HIPStream) = AMDGPU.synchronize(s)

function NextLA.sync_streams_with_default(::AMDGPU.ROCBackend, streams)
    AMDGPU.synchronize()
end
