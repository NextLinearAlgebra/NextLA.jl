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
	# CUDA graph capture of the panel is experimental: it succeeds on the
	# first instantiation but subsequent calls can replay against pointers
	# whose backing CuArrays were GC'd between panels (Julia view objects
	# rebuilt per iteration, cached graph holds the old pointer). We try
	# capture with RELAXED mode (tolerates the inner workgroup_reduce sync
	# and the cuSOLVER POTRF info-fetch), and fall back to direct execution
	# on any failure. Measured single-GPU benefit is small (~3-5 % of total
	# QR time at N=8000); the lifetime issues outweigh the gain.
	flags = CUDA.STREAM_CAPTURE_MODE_RELAXED
	try
		GC.enable(false)
		graph = try
			CUDA.capture(; flags=flags, throw_error=false) do
				body()
			end
		finally
			GC.enable(true)
		end
		if graph === nothing
			body()  # capture aborted (typically JIT); run body normally
			return nothing
		end
		if !isassigned(exec_ref) || !_update_exec_or_false(exec_ref[], graph)
			exec_ref[] = CUDA.instantiate(graph)
		end
		CUDA.launch(exec_ref[]::CUDA.CuGraphExec)
	catch e
		# Fall back to direct execution if capture / launch fails for any
		# reason (dangling pointers from GC'd views, topology mismatch the
		# update path can't patch, etc.).
		@debug "capture_panel! fallback" exception=e
		body()
	end
	return nothing
end

_update_exec_or_false(exec::CUDA.CuGraphExec, graph::CUDA.CuGraph) =
	CUDA.update(exec, graph; throw_error=false)
_update_exec_or_false(::Any, ::CUDA.CuGraph) = false

# ─────────────────────────────────────────────────────────────────────────────
# Householder QR variant overrides (geqrf_2p5d_householder!).
#
# Use cuSOLVER's tuned unblocked geqrf + larft + orgqr for the panel-level
# Householder operations; the trailing-update WY-form apply runs through
# cuBLAS GEMM (in `_household_apply_QT!`, the default implementation already
# uses `LinearAlgebra.mul!`).
function NextLA._household_panel_geqrf!(::CUDA.CUDABackend,
		A_panel::AbstractMatrix{T}, tau_panel::AbstractVector{T}) where {T<:LinearAlgebra.BlasFloat}
	# cuSOLVER geqrf! returns (A, tau); we want it in our preallocated tau_panel.
	# Call the in-place form that accepts a tau buffer.
	CUDA.CUSOLVER.geqrf!(A_panel, tau_panel)
	return nothing
end

function NextLA._household_build_T!(::CUDA.CUDABackend,
		V::AbstractMatrix{T}, tau::AbstractVector{T},
		T_out::AbstractMatrix{T}, m::Int, b::Int) where {T<:LinearAlgebra.BlasFloat}
	CUDA.CUSOLVER.larft!('F', 'C', view(V, 1:m, 1:b), view(tau, 1:b), view(T_out, 1:b, 1:b))
	return nothing
end

function NextLA._household_expand_Q!(::CUDA.CUDABackend,
		A_panel::AbstractMatrix{T}, tau_panel::AbstractVector{T},
		m::Int, b::Int) where {T<:LinearAlgebra.BlasFloat}
	CUDA.CUSOLVER.orgqr!(A_panel, tau_panel)
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Look-Ahead pipelining (Phase Q5 in qr_schur_xpartition.tex §A.1).
# Two CUDA streams σ_0 (default), σ_1 (panel). The trailing update is split
# into A_next (column slab of width b that becomes panel-(k+1)) and A_rest
# (the remaining tail). σ_0 runs Phase Q1 + S_4/S_5(A_next), then on the
# *current* step continues with S_4/S_5(A_rest). σ_1 starts Panel-(k+1)
# (Phase Q1) right after S_5(A_next), in parallel with σ_0's S_4/S_5(A_rest).
import LinearAlgebra: mul!, rdiv!, UpperTriangular

function NextLA._lookahead_run!(::CUDA.CUDABackend, m, n, A, R_acc, tau, p, tile,
		c_eff, b_full, G_buf, R_buf, info_buf, W_buf, n_streams::Int, ortho::Symbol)
	# Two-stream version (n_streams >= 2 == 2 for now; deeper look-ahead is
	# a straight extension but the X-partition optimum is s=2 per §A.1
	# Phase Q5b).
	if n_streams == 1
		# Degenerate: call the sequential implementation.
		return NextLA.geqrf_2p5d!(m, n, A, R_acc, tau; params=p, ortho=ortho)
	end

	be = NextLA.KernelAbstractions.get_backend(A)
	T = eltype(A)
	k_eff = min(m, n)

	# σ_0 = default stream of the calling task; σ_1 = a fresh CUDA stream for
	# the look-ahead panel. cuBLAS uses the task-current stream automatically.
	stream_main = CUDA.stream()
	stream_la = CUDA.CuStream()

	# Scratch for the *look-ahead* panel (factored on σ_1). Separate from
	# G_buf/R_buf because σ_0 may still be reading those when σ_1 starts.
	G_la = similar(A, b_full, b_full)
	R_la = similar(A, b_full, b_full)
	info_la = fill!(similar(A, Int, 1), 0)

	# Events for cross-stream sync.
	evt_a_next_done   = CUDA.CuEvent()
	evt_panel_next_done = CUDA.CuEvent()

	fill!(R_acc, zero(T))

	# ── Step 1: factor the very first panel on σ_0 (no look-ahead yet) ───────
	k = 1
	if k_eff < 1; return nothing; end
	sb = min(b_full, k_eff)
	A_panel = @view A[1:m, k:(k + sb - 1)]
	p_panel = sb == b_full ? p : NextLA.compute_params(be, T,max(m, sb); b=sb, c=p.c)
	partials_use = NextLA.effective_c(p_panel) > 1 ?
		(sb == b_full ? similar(A, b_full, b_full, p.Px * p.Pz) :
		 similar(A, sb, sb, p_panel.Px * p_panel.Pz)) : nothing
	NextLA.scqr3!(m, sb, A_panel, R_buf, G_buf, info_buf; params=p_panel, partials=partials_use)
	NextLA._geqrf_write_R_panel!(be, R_acc, R_buf, k, sb)

	# Track which panel-buffers each stream currently owns.
	main_G, main_R, main_info = G_buf, R_buf, info_buf
	la_G, la_R, la_info = G_la, R_la, info_la

	while k <= k_eff
		sb = min(b_full, k_eff - k + 1)
		A_panel = @view A[1:m, k:(k + sb - 1)]
		n_tr = n - (k + sb - 1)

		if n_tr == 0
			# Last panel — already factored above (k=1) or in the previous
			# iter (k>1, see σ_1 below). Nothing to do.
			break
		end

		next_sb = min(b_full, k_eff - (k + sb) + 1)  # width of A_next slab
		has_next = n_tr >= next_sb && next_sb > 0
		has_rest = n_tr > next_sb
		A_tr_full = @view A[1:m, (k + sb):n]

		if has_next
			# === σ_0: S_4/S_5 on A_next ─────────────────────────────────────
			A_next = @view A[1:m, (k + sb):(k + sb + next_sb - 1)]
			W_next = @view W_buf[1:sb, 1:next_sb]
			mul!(W_next, A_panel', A_next)
			mul!(A_next, A_panel, W_next, -one(T), one(T))
			NextLA._geqrf_write_W_block!(be, R_acc, W_next, k, k + sb, sb, next_sb)
			CUDA.record(evt_a_next_done, stream_main)

			# === σ_1: wait, then factor Panel-(k+1) ─────────────────────────
			CUDA.wait(evt_a_next_done, stream_la)
			# Switch the task's current stream to σ_1 for the cuBLAS/cuSOLVER
			# calls inside scqr3!.
			CUDA.stream!(stream_la) do
				A_panel_next = @view A[1:m, (k + sb):(k + sb + next_sb - 1)]
				p_next = next_sb == b_full ? p :
					NextLA.compute_params(be, T,max(m, next_sb); b=next_sb, c=p.c)
				partials_n = NextLA.effective_c(p_next) > 1 ?
					(next_sb == b_full ? similar(A, b_full, b_full, p.Px * p.Pz) :
					 similar(A, next_sb, next_sb, p_next.Px * p_next.Pz)) : nothing
				NextLA.scqr3!(m, next_sb, A_panel_next, la_R, la_G, la_info;
				       params=p_next, partials=partials_n)
				NextLA._geqrf_write_R_panel!(be, R_acc, la_R, k + sb, next_sb)
			end
			CUDA.record(evt_panel_next_done, stream_la)
		end

		if has_rest
			# === σ_0: S_4/S_5 on A_rest (concurrent with σ_1's panel) ────────
			A_rest = @view A[1:m, (k + sb + next_sb):n]
			n_rest = size(A_rest, 2)
			W_rest = @view W_buf[1:sb, 1:n_rest]
			mul!(W_rest, A_panel', A_rest)
			mul!(A_rest, A_panel, W_rest, -one(T), one(T))
			NextLA._geqrf_write_W_block!(be, R_acc, W_rest, k, k + sb + next_sb, sb, n_rest)
		end

		# Sync: σ_0 must wait for σ_1's panel to finish before the next step
		# (the next step's A_panel is what σ_1 just wrote).
		if has_next
			CUDA.wait(evt_panel_next_done, stream_main)
			# Swap roles for the next step: σ_1's output buffers become "main".
			main_G, la_G = la_G, main_G
			main_R, la_R = la_R, main_R
			main_info, la_info = la_info, main_info
		end

		k += sb
	end

	# Final synchronize on σ_1 to make sure the last panel-next is committed.
	CUDA.synchronize(stream_la)
	return nothing
end

end
