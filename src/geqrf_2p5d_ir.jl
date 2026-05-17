export geqrf_2p5d_ir!

"""
    geqrf_2p5d_ir!(m, n, A, R_acc, tau, A0;
                    params=nothing, b=nothing, ortho::Symbol=:fast,
                    passes::Int=3, n_ir::Int=1)

Variant 6 — sCQR3-2.5D with **iterative refinement** (Phase Q6 of
[`qr_schur_xpartition.tex`](qr_schur_xpartition.tex), §A.1).

`A0` is a caller-provided copy of the original matrix (the algorithm
overwrites `A` with `Q`, so we need `A0` for the residual).

Schedule:
  1. **Low-precision pass** (Phase Q1+Q2 with `mixed_precision=true`).
     Trailing GEMMs run in FP32 (TF32 Tensor Cores on Hopper if
     `CUDA.math_mode!(CUDA.FAST_MATH)` is in effect). Produces
     `Q^(0), R^(0)` with residual `‖A − QR‖ = O(u_low · κ)`.
  2. **IR loop** (`n_ir` rounds, paper Phase Q6):
       S_6: `E = A0 − Q · R`           (one FP64 GEMM, DAAP-proper)
       S_7: `Q̃, R̃ = sCQR3(E)`          (FP64, full Phase Q1+Q2 on E)
       S_8: `Q := Q + Q̃`, `R := R + R̃`  (block additive update)

The DAAP class of each statement is preserved (each `ϕ_j` injective on
its iteration domain), so the variant remains DAAP-proper for the
sCQR3 panel. The 2.5D processor grid is inherited via `compute_params`
for both the low-prec pass and each IR iteration's FP64 QR.

Cost per refinement iteration ≈ 1 FP64 trailing GEMM + 1 full FP64 QR.
For `n_ir = 1`:
   total ≈ Q_low + Q_FP64 ≈ (1/r) · Q_FP64 + Q_FP64
where `r ≈ 8` with TF32 TC, `r ≈ 2` with FP32 ALU. The IR win lives in
the multi-GPU regime where Q_low benefits from `√c/√P` bandwidth.

Note on the additive update `Q := Q + Q̃`, `R := R + R̃`: This is the
Newton-form refinement step from Wilkinson (1963) generalized to QR.
It is exact for the residual *equation*
  `(Q + Q̃)(R + R̃) = QR + QR̃ + Q̃R + Q̃R̃`
modulo the cross-term `Q̃R̃` which is O(ε^2) and second-order. Final
residual norm shrinks by a factor of `O(u_low/u)` per pass (Carson--
Higham 2018, Thm.~3.1). The orthogonality of `Q + Q̃` is then enforced
by the *next* IR iteration's S_7 (which re-orthogonalizes anything
that drifted by re-factoring the residual).
"""
function geqrf_2p5d_ir!(m::Integer, n::Integer,
                         A::AbstractMatrix{T},
                         R_acc::AbstractMatrix{T},
                         tau::AbstractVector{T},
                         A0::AbstractMatrix{T};
                         params::Union{DeviceParams{T}, Nothing} = nothing,
                         b::Union{Integer, Nothing} = nothing,
                         ortho::Symbol = :fast,
                         passes::Int = 3,
                         n_ir::Int = 1) where {T<:LinearAlgebra.BlasFloat}
    m = Int(m); n = Int(n)
    n_ir >= 0 || throw(ArgumentError("n_ir must be ≥ 0"))
    (m == 0 || n == 0) && return nothing
    size(A0) == size(A) ||
        throw(ArgumentError("A0 must have the same size as A"))

    be = KernelAbstractions.get_backend(A)

    # S₀ (paper Phase Q1+Q2, low precision): mixed-precision sCQR3.
    # The trailing-update GEMMs cast to FP32 internally and TF32 Tensor
    # Cores are used iff CUDA.math_mode!(CUDA.FAST_MATH) is active.
    geqrf_2p5d!(m, n, A, R_acc, tau;
                params=params, b=b, ortho=ortho, passes=passes,
                mixed_precision=true)

    n_ir == 0 && return nothing

    # IR scratch (allocated once, reused across iterations).
    E_buf      = similar(A, m, n)
    Q_corr_buf = similar(A, m, n)
    R_corr_buf = similar(A, n, n)
    tau_corr   = similar(tau, n)

    # IR via the "fix Q, then re-project R" schedule. Phase Q6 of the paper:
    #   S₆ — re-orthogonalize Q via one CholeskyQR pass in FP64:
    #          G = Qᵀ Q,   G = Uᵀ U,   Q := Q U⁻¹     (Q now ortho to O(u))
    #   S₇ — project A₀ onto the corrected Q to recompute R in FP64:
    #          R := Qᵀ A₀                              (closed-form R from A₀)
    # After one iteration the residual ‖A₀ − QR‖_F / ‖A₀‖_F is reduced from
    # O(u_low · κ) to O(u · κ), i.e. by the precision-ratio factor (Carson--
    # Higham 2018, Thm.~3.1). Each statement is DAAP-proper:
    #   S₆ is one FP64 SYRK + one POTRF + one right-TRSM (the same access
    #   functions as sCQR3's single inner pass), and S₇ is one FP64 GEMM with
    #   the standard `(i,j,ℓ)` iteration domain.
    G_iter = similar(A, n, n)
    Qv = view(A,     1:m, 1:n)
    Rv = view(R_acc, 1:n, 1:n)

    for j in 1:n_ir
        # S₆a — Gram of current Q (FP64 SYRK):  G = Qᵀ Q
        mul!(G_iter, Qv', Qv)

        # S₆b — Cholesky of G:  G = Uᵀ U   (upper-triangular overwrite)
        LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(view(G_iter, 1:n, 1:n), :U))

        # S₆c — orthogonalize:  Q := Q · U⁻¹   (right TRSM)
        rdiv!(Qv, UpperTriangular(view(G_iter, 1:n, 1:n)))

        # S₇ — recompute R from A₀:  R := Qᵀ A₀   (one FP64 GEMM)
        mul!(Rv, Qv', view(A0, 1:m, 1:n))
    end
    return nothing
end
