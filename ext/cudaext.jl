module cudaext

using NextLA

if isdefined(Base, :get_extension)
    import CUDA
else
    import ..CUDA
end

function __init__()
    # Enable TF32 Tensor Cores in cuBLAS when NEXTLA_TF32=1. Off by default —
    # TF32 degrades FP32 GEMM to ~10 mantissa bits and breaks the κ≥1e6 cases
    # in validate_ortho_modes.jl. cuSOLVER on Hopper uses TF32 internally for
    # FP32 geqrf, so this flag is what makes the FP32 comparison fair.
    if get(ENV, "NEXTLA_TF32", "0") == "1"
        try
            CUDA.math_mode!(CUDA.FAST_MATH)
            @info "NextLA cudaext: TF32 enabled (CUDA.FAST_MATH)"
        catch e
            @warn "NEXTLA_TF32 requested but CUDA.math_mode!(FAST_MATH) failed" exception=e
        end
    end
end

"""
    probe_device(::CUDA.CUDABackend, ::Type{T}) -> (P, M)

Map the X-partition device model (P processors with M words of fast memory
each) onto a CUDA GPU. The paper treats M as implementation-defined; on a GPU
the natural "fast memory" per SM is shared memory + the register file, both
of which are explicitly partitioned per-SM. We use that sum by default, giving
roughly 60K FP64 words on Hopper (228 KB SMEM + 256 KB regs / SM = 484 KB).

Env overrides (per-SM units in bytes; let you sweep the X-partition `c` budget
without code changes):

  - `NEXTLA_M_BYTES`   bytes of fast memory per SM (default: smem + 4·regs).
  - `NEXTLA_P_OVERRIDE` number of processors (default: SM count).

These ride straight into `compute_params`, which derives `c = ⌊PM/N²⌋` and the
X-partition cube side `b = √M` from them.
"""
function NextLA.probe_device(backend::CUDA.CUDABackend, ::Type{T}) where {T}
	dev = CUDA.device()
	sm_count = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
	smem_bytes = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
	reg_words  = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR))
	default_M_bytes = smem_bytes + 4 * reg_words
	Mbytes = parse(Int, get(ENV, "NEXTLA_M_BYTES", string(default_M_bytes)))
	Pv     = parse(Int, get(ENV, "NEXTLA_P_OVERRIDE", string(sm_count)))
	P = max(1, Pv)
	M = max(1, Mbytes ÷ max(1, sizeof(T)))
	return P, M
end

function NextLA._scqr3_gram_backend_caps(backend::CUDA.CUDABackend, ::Type{T}) where {T}
	dev = CUDA.device()
	max_th = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
	smem = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
	return (max_th, smem)
end

# cuSOLVER POTRF for GPU Cholesky — avoids the single-thread serial KA kernel.
function NextLA._scqr3_potrf!(::CUDA.CUDABackend, G::AbstractMatrix, b::Int)
	CUDA.CUSOLVER.potrf!('U', view(G, 1:b, 1:b))
end

# cuBLAS SYRK/HERK for the Gram step — writes only the upper triangle of
# Gv = Av' * Av at half the flops of cuBLAS GEMM. On Hopper this saves ~15-25%
# of the panel Gram time depending on b. Falls through to the generic `mul!`
# path (GEMM) when env `NEXTLA_USE_SYRK=0`.
import LinearAlgebra
function NextLA._scqr3_syrk_herk!(::CUDA.CUDABackend,
		Gv::AbstractMatrix{T}, Av::AbstractMatrix{T}) where {T<:LinearAlgebra.BlasFloat}
	if T <: LinearAlgebra.BlasReal
		CUDA.CUBLAS.syrk!('U', 'T', one(T), Av, zero(T), Gv)
	else
		CUDA.CUBLAS.herk!('U', 'C', one(real(T)), Av, zero(real(T)), Gv)
	end
	return Gv
end

# ─────────────────────────────────────────────────────────────────────────────
# CUDA graph capture for the geqrf_2p5d! panel iteration.
#
# Folds the ~26 per-panel kernel launches (3 sCQR3 iterations: SYRK + trace +
# POTRF + TRSM, then trailing GEMM × 2 + R/W scatter writes) into a single
# captured CUDA graph. After the first capture an executable graph is
# instantiated and cached in `exec_ref`; subsequent panels try `cuGraphExecUpdate`
# to patch buffer pointers (cheap) and only re-instantiate if topology changes
# (e.g. the last partial panel with sb < b_full — currently we just bypass
# capture for that case in geqrf_2p5d!).
NextLA._graph_capture_supported(::CUDA.CUDABackend) = true

function NextLA.capture_panel!(::CUDA.CUDABackend, exec_ref::Ref{Any}, body::Function)
	# Capture: re-record body each call. Operations are recorded into a
	# CuGraph rather than executed when the stream is in capture mode.
	GC.enable(false)
	graph = try
		CUDA.capture(throw_error=false) do
			body()
		end
	finally
		GC.enable(true)
	end
	if graph === nothing
		# Capture failed — typically a JIT compile happened during recording.
		# Execute the body once outside capture (so JIT lands in the regular
		# code cache), then re-record for next time.
		body()
		GC.enable(false)
		graph = try
			CUDA.capture(throw_error=true) do
				body()
			end
		finally
			GC.enable(true)
		end
		if !isassigned(exec_ref) || !_update_exec_or_false(exec_ref[], graph)
			exec_ref[] = CUDA.instantiate(graph)
		end
		return nothing
	end
	# Patch the cached executable; fall back to instantiate on topology change.
	if !isassigned(exec_ref) || !_update_exec_or_false(exec_ref[], graph)
		exec_ref[] = CUDA.instantiate(graph)
	end
	CUDA.launch(exec_ref[]::CUDA.CuGraphExec)
	return nothing
end

_update_exec_or_false(exec::CUDA.CuGraphExec, graph::CUDA.CuGraph) =
	CUDA.update(exec, graph; throw_error=false)
_update_exec_or_false(::Any, ::CUDA.CuGraph) = false

end
