export geqrf_2p5d_householder!

"""
    geqrf_2p5d_householder!(m, n, A, R_acc, tau; params=nothing, b=nothing)

Path-(h) Householder variant of `geqrf_2p5d!`. Performs blocked QR using
cuSOLVER's panel `geqrf` (Householder reflectors → in lower trapezoid of the
panel; reflector taus in `tau_buf`) plus a WY-representation trailing update
via three cuBLAS GEMMs. The 2.5D processor grid layout (`params.c`,
`params.Px`, `params.Py`, `params.Pz`) is honoured for sizing decisions but
the panel reduction itself is sequential Householder (not TSQR), so the panel
phase is **Quasi-DAAP** rather than DAAP-proper (cf.
[`qr_schur_xpartition.tex`](qr_schur_xpartition.tex) §A.1, Path~(h)).

The interface matches `geqrf_2p5d!`: on return, `A[1:m, 1:n]` stores the
explicit orthonormal `Q` panel-by-panel (NOT Householder vectors — they are
expanded via `orgqr` after each panel for compatibility with downstream code
that consumes an explicit `Q`), and `R_acc[1:n, 1:n]` holds the upper-
triangular `R`. `tau` receives the reflector scalars from cuSOLVER.

Flop budget vs `geqrf_2p5d!` (sCQR3):
  - Panel: `2 m b²` vs sCQR3's `~9 m b²` (3 sCQR3 iters of Gram+POTRF+TRSM).
  - Trailing: `~4 m b n_tr` vs sCQR3's `~4 m b n_tr` (identical).
  - Total: `(4/3) N³` vs sCQR3's `4 N³`.

On a single GPU the 2.5D `√c/√P` bandwidth advantage does not physically
activate (no separate replicas); this variant therefore reduces to a
"blocked-Householder + explicit Q" geqrf that is structurally close to
cuSOLVER's own `geqrf!`. Use as the Path-(h) baseline for benchmarks.
"""
function geqrf_2p5d_householder!(m::Integer, n::Integer,
                                  A::AbstractMatrix{T},
                                  R_acc::AbstractMatrix{T},
                                  tau::AbstractVector{T};
                                  params::Union{DeviceParams{T}, Nothing} = nothing,
                                  b::Union{Integer, Nothing} = nothing) where {T}
    m = Int(m); n = Int(n)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    n >= 0 || throw(ArgumentError("n must be ≥ 0"))
    (m == 0 || n == 0) && return nothing
    size(A, 1) >= m && size(A, 2) >= n ||
        throw(ArgumentError("A too small for m=$m, n=$n"))
    size(R_acc, 1) >= n && size(R_acc, 2) >= n ||
        throw(ArgumentError("R_acc too small for n=$n"))
    length(tau) >= min(m, n) ||
        throw(ArgumentError("tau must have length ≥ min(m,n)=$(min(m,n))"))

    be = KernelAbstractions.get_backend(A)
    k_eff = min(m, n)
    N_budget = max(m, n)

    p = if params === nothing
        bval = b === nothing ? min(32, k_eff) : max(1, Int(b))
        compute_params(be, T, N_budget; b = bval, c = nothing)
    else
        params
    end
    b_full = p.b
    b_full >= 1 || throw(ArgumentError("DeviceParams.b must be ≥ 1"))

    fill!(R_acc, zero(T))

    # ── Persistent scratch ────────────────────────────────────────────────────
    # T_buf: WY representation T-matrix (b × b), upper-triangular.
    T_buf = similar(A, b_full, b_full)
    # Y_buf: workspace for trailing-update GEMM chain (b × max-trailing-cols).
    Y_buf = similar(A, b_full, n > b_full ? n - b_full : 1)

    # ── Outer loop over panels ────────────────────────────────────────────────
    k = 1
    while k <= k_eff
        sb = min(b_full, k_eff - k + 1)
        m_panel = m - k + 1

        # Active panel slice and its tau range.
        A_panel = @view A[k:m, k:(k + sb - 1)]
        tau_panel = @view tau[k:(k + sb - 1)]

        # Phase Q1(h): unblocked Householder QR of the panel.
        # cuSOLVER's geqrf! writes V (Householder vectors) into the strict lower
        # triangle of A_panel and R into the upper triangle; tau_panel receives
        # the reflector scalars.
        _household_panel_geqrf!(be, A_panel, tau_panel)

        # Scatter R from the upper triangle of the panel into R_acc[k:k+sb-1, k:k+sb-1].
        _copy_triu_into_R_acc!(be, R_acc, A_panel, k, sb)

        # Phase Q2(h): WY-form trailing update of A[k:m, k+sb:n].
        n_tr = n - (k + sb - 1)
        if n_tr > 0
            A_trailing = @view A[k:m, (k + sb):n]
            T_block = @view T_buf[1:sb, 1:sb]
            # Build T (compact WY): T = larft(V_panel, tau_panel).
            _household_build_T!(be, A_panel, tau_panel, T_block, m_panel, sb)
            # Apply Q^T to A_trailing: A_trailing := (I - V T^T V^T) A_trailing.
            _household_apply_QT!(be, A_panel, T_block, A_trailing,
                                  view(Y_buf, 1:sb, 1:n_tr),
                                  m_panel, sb, n_tr)
            # Write the off-diagonal R block: R_acc[k:k+sb-1, k+sb:n] = (the
            # top sb rows of A_trailing, which now hold R after Q^T application).
            _copy_block_into_R_acc!(be, R_acc, A_trailing, k, k + sb, sb, n_tr)
            # Zero out those rows in A_trailing (they belong to R, not Q).
            _zero_top_rows!(be, A_trailing, sb, n_tr)
        end

        # Phase Q1(h) post: expand the Householder vectors of this panel into an
        # explicit orthonormal Q block so downstream code (`Q = A`) works as in
        # the sCQR3 path. Done after the trailing update so we don't re-read V.
        _household_expand_Q!(be, A_panel, tau_panel, m_panel, sb)

        k += sb
    end
    return nothing
end

# ── Backend hooks (defaults route through generic primitives) ─────────────────
# Specialized in `ext/cudaext.jl` to use cuSOLVER + cuBLAS directly.

"""
    _household_panel_geqrf!(be, A_panel, tau_panel)

In-place unblocked Householder QR of the panel. After the call, the strict
lower trapezoid of `A_panel` holds the Householder vectors (`V`), the upper
triangle holds `R`, and `tau_panel` holds the reflector scalars.
"""
function _household_panel_geqrf!(::KernelAbstractions.CPU,
                                  A_panel::AbstractMatrix{T},
                                  tau_panel::AbstractVector{T}) where {T<:LinearAlgebra.BlasFloat}
    A_cpu, _ = LinearAlgebra.LAPACK.geqrf!(A_panel)
    return nothing
end

function _household_panel_geqrf!(be, A_panel::AbstractMatrix{T},
                                  tau_panel::AbstractVector{T}) where {T<:LinearAlgebra.BlasFloat}
    # Generic fallback: use the existing pure-KA geqr2 kernel (slow on GPU but
    # correct). The CUDA backend overrides this in cudaext.jl.
    m, n = size(A_panel)
    work = similar(A_panel, max(n, 1))
    geqr2!(m, n, A_panel, tau_panel, work)
    return nothing
end

"""
    _household_build_T!(be, V, tau, T_out, m, b)

Construct the upper-triangular WY factor `T` such that
`I - V T V^T == prod_{k=1}^{b} (I - tau_k * v_k * v_k^T)`.
"""
function _household_build_T!(::KernelAbstractions.CPU,
                              V::AbstractMatrix{T}, tau::AbstractVector{T},
                              T_out::AbstractMatrix{T}, m::Int, b::Int) where {T<:LinearAlgebra.BlasFloat}
    larft!('F', 'C', m, b, V, tau, T_out)
    return nothing
end

function _household_build_T!(be, V::AbstractMatrix{T}, tau::AbstractVector{T},
                              T_out::AbstractMatrix{T}, m::Int, b::Int) where {T<:LinearAlgebra.BlasFloat}
    # Generic fallback: use the existing KA larft.
    larft!('F', 'C', m, b, V, tau, T_out)
    return nothing
end

"""
    _household_apply_QT!(be, V, T_mat, C, Y_workspace, m, b, n_tr)

Apply `Q^T = I - V T^T V^T` in-place to a trailing block `C`:
    C := (I - V T^T V^T) C
using three GEMMs:
    Y := V^T C           (b × m × n_tr)
    Y := T^T Y           (b × b × n_tr)
    C := C - V Y         (m × b × n_tr)

`V` has unit-lower-trapezoidal structure (1 on diagonal implicit). We
temporarily save/set the diagonal of V to 1.0 for the GEMM, then restore.
"""
function _household_apply_QT!(be, V::AbstractMatrix{T}, T_mat::AbstractMatrix{T},
                               C::AbstractMatrix{T}, Y::AbstractMatrix{T},
                               m::Int, b::Int, n_tr::Int) where {T<:LinearAlgebra.BlasFloat}
    # Save the diagonal of V (it currently holds R's diagonal), then set it
    # to 1 so V looks unit-lower-trapezoidal for the GEMM.
    diag_save = similar(V, T, (b,))
    _copy_diag!(be, diag_save, V, b)
    _scqr3_fill_diag!(be, V, one(T), b)
    # And zero the strict upper triangle of V (currently holds R off-diagonal).
    upper_save = similar(V, T, (b, b))
    _copy_strict_upper!(be, upper_save, V, b)
    _zero_strict_upper!(be, V, b)

    # Y := V^T C  (b × n_tr; reads m_panel rows of V and C).
    mul!(Y, V', C)
    # Y := T^T Y
    mul!(Y, UpperTriangular(T_mat)', view(Y, 1:b, 1:n_tr))  # alias OK with TRMM
    # C := C - V Y
    mul!(C, V, Y, -one(T), one(T))

    # Restore V (strict-upper and diagonal).
    _restore_strict_upper!(be, V, upper_save, b)
    _restore_diag!(be, V, diag_save, b)
    return nothing
end

"""
    _household_expand_Q!(be, A_panel, tau, m, b)

Expand the Householder vectors stored in `A_panel`'s lower trapezoid into an
explicit orthonormal `Q` block (overwrites `A_panel` with `Q`). Routes to
cuSOLVER's `orgqr!` on CUDA.
"""
function _household_expand_Q!(::KernelAbstractions.CPU,
                               A_panel::AbstractMatrix{T},
                               tau_panel::AbstractVector{T},
                               m::Int, b::Int) where {T<:LinearAlgebra.BlasFloat}
    LinearAlgebra.LAPACK.orgqr!(A_panel, tau_panel)
    return nothing
end

function _household_expand_Q!(be, A_panel::AbstractMatrix{T},
                               tau_panel::AbstractVector{T},
                               m::Int, b::Int) where {T<:LinearAlgebra.BlasFloat}
    # Backend default: no-op (caller of geqrf_2p5d_householder! reads V/tau).
    # CUDA override uses CUSOLVER.orgqr!.
    error("_household_expand_Q!: no implementation for backend $(typeof(be)); CUDA backend required")
end

# ── Small helper kernels for diag / upper-triangle manipulation ───────────────

@kernel function _copy_diag_kernel!(out, V, b::Int)
    j = @index(Global, Linear)
    @inbounds if j <= b
        out[j] = V[j, j]
    end
end

function _copy_diag!(be, out::AbstractVector{T}, V::AbstractMatrix{T}, b::Int) where {T}
    _copy_diag_kernel!(be)(out, V, b; ndrange = b)
end

@kernel function _restore_diag_kernel!(V, src, b::Int)
    j = @index(Global, Linear)
    @inbounds if j <= b
        V[j, j] = src[j]
    end
end

function _restore_diag!(be, V::AbstractMatrix{T}, src::AbstractVector{T}, b::Int) where {T}
    _restore_diag_kernel!(be)(V, src, b; ndrange = b)
end

@kernel function _copy_strict_upper_kernel!(out, V, b::Int)
    lin = @index(Global, Linear)
    if lin <= b * b
        i = (lin - 1) ÷ b + 1
        j = (lin - 1) % b + 1
        @inbounds out[i, j] = (i < j) ? V[i, j] : zero(eltype(out))
    end
end

function _copy_strict_upper!(be, out::AbstractMatrix{T}, V::AbstractMatrix{T}, b::Int) where {T}
    _copy_strict_upper_kernel!(be)(out, V, b; ndrange = b * b)
end

@kernel function _zero_strict_upper_kernel!(V, b::Int)
    lin = @index(Global, Linear)
    if lin <= b * b
        i = (lin - 1) ÷ b + 1
        j = (lin - 1) % b + 1
        @inbounds if i < j
            V[i, j] = zero(eltype(V))
        end
    end
end

function _zero_strict_upper!(be, V::AbstractMatrix{T}, b::Int) where {T}
    _zero_strict_upper_kernel!(be)(V, b; ndrange = b * b)
end

@kernel function _restore_strict_upper_kernel!(V, src, b::Int)
    lin = @index(Global, Linear)
    if lin <= b * b
        i = (lin - 1) ÷ b + 1
        j = (lin - 1) % b + 1
        @inbounds if i < j
            V[i, j] = src[i, j]
        end
    end
end

function _restore_strict_upper!(be, V::AbstractMatrix{T}, src::AbstractMatrix{T}, b::Int) where {T}
    _restore_strict_upper_kernel!(be)(V, src, b; ndrange = b * b)
end

@kernel function _copy_triu_into_R_acc_kernel!(R_acc, A_panel, k::Int, sb::Int)
    lin = @index(Global, Linear)
    if lin <= sb * sb
        i = (lin - 1) % sb + 1
        j = (lin - 1) ÷ sb + 1
        @inbounds R_acc[k + i - 1, k + j - 1] = (i <= j) ? A_panel[i, j] : zero(eltype(R_acc))
    end
end

function _copy_triu_into_R_acc!(be, R_acc::AbstractMatrix{T}, A_panel::AbstractMatrix{T},
                                 k::Int, sb::Int) where {T}
    _copy_triu_into_R_acc_kernel!(be)(R_acc, A_panel, k, sb; ndrange = sb * sb)
end

@kernel function _copy_block_into_R_acc_kernel!(R_acc, src, k_row::Int, k_col::Int,
                                                  sb::Int, n_tr::Int)
    lin = @index(Global, Linear)
    if lin <= sb * n_tr
        i = (lin - 1) % sb + 1
        j = (lin - 1) ÷ sb + 1
        @inbounds R_acc[k_row + i - 1, k_col + j - 1] = src[i, j]
    end
end

function _copy_block_into_R_acc!(be, R_acc::AbstractMatrix{T}, src::AbstractMatrix{T},
                                  k_row::Int, k_col::Int, sb::Int, n_tr::Int) where {T}
    _copy_block_into_R_acc_kernel!(be)(R_acc, src, k_row, k_col, sb, n_tr; ndrange = sb * n_tr)
end

@kernel function _zero_top_rows_kernel!(A_trailing, sb::Int, n_tr::Int)
    lin = @index(Global, Linear)
    if lin <= sb * n_tr
        i = (lin - 1) % sb + 1
        j = (lin - 1) ÷ sb + 1
        @inbounds A_trailing[i, j] = zero(eltype(A_trailing))
    end
end

function _zero_top_rows!(be, A_trailing::AbstractMatrix{T}, sb::Int, n_tr::Int) where {T}
    _zero_top_rows_kernel!(be)(A_trailing, sb, n_tr; ndrange = sb * n_tr)
end
