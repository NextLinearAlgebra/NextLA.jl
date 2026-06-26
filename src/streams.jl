"""
    create_streams(backend, n) → Vector

Create `n` independent execution streams for `backend`.  On non-GPU backends
returns a length-`n` vector of `nothing` values (sequential fallback).
"""
create_streams(_, n::Int) = fill(nothing, n)

"""
    with_stream(f, backend, stream)

Execute `f()` directing backend operations onto `stream`.  On CPU / when
`stream === nothing`, calls `f()` directly.
"""
with_stream(f, _, ::Nothing) = f()

"""
    sync_stream(backend, stream)

Block the calling thread until all work submitted to `stream` completes.
No-op on CPU or when `stream === nothing`.
"""
sync_stream(_, ::Nothing) = nothing

"""
    sync_streams_with_default(backend, streams)

Insert a GPU-side barrier that makes every element of `streams` wait for all
work currently outstanding on the backend's *default* stream to finish.

Use this after a kernel / BLAS call on the default stream to ensure that
category streams launching co-dependent work see the result.

- On CPU:  no-op (execution is sequential).
- On CUDA: records a `CuEvent` on the current stream, then calls
           `CUDA.wait(event, s)` for each stream `s`.  This is a pure
           GPU-side dependency — the CPU does **not** block.
- On other GPU backends: falls back to a full device synchronise.
"""
sync_streams_with_default(_, _) = nothing
