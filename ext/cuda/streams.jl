NextLA.create_streams(::CUDA.CUDABackend, n::Int) = [CUDA.CuStream() for _ in 1:n]

function NextLA.with_stream(f, ::CUDA.CUDABackend, s::CUDA.CuStream)
    CUDA.stream!(f, s)
end

NextLA.sync_stream(::CUDA.CUDABackend, s::CUDA.CuStream) = CUDA.synchronize(s)

function NextLA.sync_streams_with_default(::CUDA.CUDABackend,
                                          streams::AbstractVector{<:CUDA.CuStream})
    ev = CUDA.CuEvent()
    CUDA.record(ev)
    for s in streams
        CUDA.wait(ev, s)
    end
end
