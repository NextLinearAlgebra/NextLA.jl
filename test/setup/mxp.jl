# Helpers shared by the mixed-precision tests, test/mxp/**.
#
# Error bounds. Every accuracy check uses the unit roundoff u = eps(T)/2 of the
# precision the result depends on (Higham, Accuracy and Stability of Numerical
# Algorithms, 2nd ed., 2002, §2.1), instead of a fixed number:
#   - storing a value in T changes it by at most u(T) relative (§2.1, round to
#     nearest, normal range); mxp_storage_ok checks that element by element;
#   - an operation over n terms in precision T is checked against
#     MXP_LAMBDA * sqrt(n) * u(T), the probabilistic bound of Higham and Mary,
#     "A new approach to probabilistic rounding error analysis", SIAM J. Sci.
#     Comput. 41(5), 2019, A2815-A2835, instead of the worst case n * u(T), which
#     is far from what is observed and would hide errors of a wrong precision.
#   - a solve or factorization is checked by its normwise backward error
#     (§7.1, Rigal and Gaches; §9.3 for LU, §10.1 for Cholesky), which does not
#     depend on the condition number.
# With Float32 data, a block rounded to Float16 by mistake gives an error near
# u(Float16) = 4.9e-4, far above the Float32 bound, so these checks see it.

using NextLA.MixedPrecision

# The backends the tests run on: every available one, or the ones given with the
# runner flag --backends (see test/gpu_backends.jl).
const MXP_BACKENDS = available_backends()
@info "Mixed-precision test backends" backends=first.(MXP_BACKENDS)

# The containers support Float8_E5M2 through an optional package that is not a
# test dependency, so the storage types tested here are these.
const MXP_NARROW = (Float16, NextLA.BFloat16)

const MXP_LAMBDA = 8

mxp_u(::Type{T}) where {T} = Float64(eps(T)) / 2

"""
    mxp_tol(T, n)

Normwise relative error allowed for a result computed over `n` terms whose
narrowest precision is `T`: `MXP_LAMBDA * sqrt(n) * u(T)`.
"""
mxp_tol(::Type{T}, n::Integer) where {T} = MXP_LAMBDA * sqrt(n) * mxp_u(T)

"""The narrowest type in a `precisions` vector (largest unit roundoff)."""
mxp_narrowest(precisions) = argmax(mxp_u, precisions)

mxp_relerr(X, ref) = norm(Float64.(Array(X)) - ref) / norm(ref)

# ---------------------------------------------------------------------------
# Storage: the precision each element of a container is stored in.
# ---------------------------------------------------------------------------

"""
    mxp_eltype_map(M) -> Matrix{DataType}

The storage type of every element of the container `M`, from its block tree.
"""
function mxp_eltype_map(M)
    E = Matrix{DataType}(undef, size(M))
    _mxp_map!(E, M)
    return E
end

function _mxp_map!(E, M::FullMixedPrec)
    M.Base !== nothing && return fill!(E, eltype(M.Base))
    mid, n = size(M.A11, 1), M.n
    _mxp_map!(view(E, 1:mid, 1:mid), M.A11)
    _mxp_map!(view(E, mid+1:n, mid+1:n), M.A22)
    fill!(view(E, mid+1:n, 1:mid), eltype(M.A21))
    fill!(view(E, 1:mid, mid+1:n), eltype(M.A12))
    return E
end

function _mxp_map!(E, M::Union{SymmMixedPrec, TriMixedPrec})
    M.Base !== nothing && return fill!(E, eltype(M.Base))
    mid, n = size(M.A11, 1), M.n
    _mxp_map!(view(E, 1:mid, 1:mid), M.A11)
    _mxp_map!(view(E, mid+1:n, mid+1:n), M.A22)
    fill!(view(E, mid+1:n, 1:mid), eltype(M.OffDiag))
    fill!(view(E, 1:mid, mid+1:n), eltype(M.OffDiag))
    return E
end

function _mxp_map!(E, M::TiledTriMixedPrec{TD, TO}) where {TD, TO}
    fill!(E, TO)
    o = M.offsets
    for k in 1:length(M.diag)
        fill!(view(E, o[k]:o[k+1]-1, o[k]:o[k+1]-1), TD)
    end
    return E
end

"""
    mxp_storage_ok(R, A, E)

Whether every element of the reconstruction `R` is within one rounding of `A` in
its storage type `E[i, j]`. A conversion from Float64 to a type narrower than
Float32 may round twice, through Float32 (BFloat16s.jl converts that way), so
those elements are allowed u(T) + u(Float32).
"""
function mxp_storage_ok(R, A, E)
    bound(T) = sizeof(T) < 4 ? mxp_u(T) + mxp_u(Float32) : mxp_u(T)
    return all(abs(Float64(R[i]) - Float64(A[i])) <= bound(E[i]) * abs(Float64(A[i])) for i in eachindex(A))
end

# ---------------------------------------------------------------------------
# Matrices. Entries of rand() + 0.5 stay out of the Float16 subnormal range, so
# the storage bound above holds; diagonal dominance keeps LU without pivoting
# and the solves well conditioned (Higham 2002, §9.5).
# ---------------------------------------------------------------------------

mxp_rand(T, m, n) = rand(T, m, n) .+ T(0.5)
# R * R' from BLAS need not be exactly symmetric; (S + S')/2 is, so the stored
# triangle and its mirror are the same matrix.
mxp_spd(T, n) = (R = mxp_rand(T, n, n); S = R * R' + T(n) * I; (S + S') / 2)
mxp_diagdom(T, n) = mxp_rand(T, n, n) + T(2n) * I
mxp_tri(uplo, X) = uplo == 'L' ? tril(X) : triu(X)
mxp_trimat(T, n, uplo) = mxp_tri(uplo, mxp_rand(T, n, n)) + T(n) * I

# ---------------------------------------------------------------------------
# Backward errors (Higham 2002): of a triangular solve op(A) X = B or X op(A) = B,
# ‖B - op(A) X‖ / (‖A‖ ‖X‖ + ‖B‖) (Rigal and Gaches, Thm 7.1); of LU, ‖A - LU‖/‖A‖;
# of Cholesky, ‖A - LLᵀ‖/‖A‖. All in Float64, on the host.
# ---------------------------------------------------------------------------

function mxp_trsm_backward_error(side, uplo, trans, diag, A, X, B)
    Ad = Float64.(Array(A))
    Ad = uplo == 'L' ? tril(Ad) : triu(Ad)
    diag == 'U' && (Ad[LinearAlgebra.diagind(Ad)] .= 1)
    opA = trans == 'N' ? Ad : transpose(Ad)
    Xd, Bd = Float64.(Array(X)), Float64.(B)
    R = side == 'L' ? Bd - opA * Xd : Bd - Xd * opA
    return norm(R) / (norm(opA) * norm(Xd) + norm(Bd))
end

mxp_lu_backward_error(F, A) =
    (Fd = Float64.(Array(F)); norm(UnitLowerTriangular(Fd) * UpperTriangular(Fd) - A) / norm(A))

function mxp_chol_backward_error(F, A, uplo)
    Fd = Float64.(Array(F))
    return norm((uplo == 'L' ? Fd * Fd' : Fd' * Fd) - A) / norm(A)
end

# BLAS on Float64 copies: the exact answer of a triangular multiply, up to u(Float64).
function mxp_trmm_reference(side, uplo, trans, diag, A, B)
    ref = Float64.(B)
    LinearAlgebra.BLAS.trmm!(side, uplo, trans, diag, 1.0, Float64.(Array(A)), ref)
    return ref
end

"""
    mxp_on(AT, A)

`A` moved to the array type of a backend; `Array` leaves it as it is.
"""
mxp_on(AT, A) = AT === Array ? A : AT(A)

# ---------------------------------------------------------------------------
# GPU. The Symm/Tri/Tiled constructors, adaptive_precisions and the
# reconstruct_matrix of Symm/Tri are tested on the device on every Julia version
# (fixes F5 and F3: tril!(copy(view)) instead of tril(view), and the triangle
# test inside the reconstruct broadcast).
# ---------------------------------------------------------------------------

"""
    mxp_host(M)

A copy of the container `M` with every block moved to the host, to check a result
computed on a device against the same container read on the CPU.
"""
_mxp_host(x) = x === nothing ? nothing : x isa MixedPrecision.AbstractMixedPrec ? mxp_host(x) : Array(x)
mxp_host(M::FullMixedPrec{T}) where {T} =
    FullMixedPrec{T}(_mxp_host(M.A11), _mxp_host(M.A22), _mxp_host(M.A21), _mxp_host(M.A12),
                     M.A21_scale, M.A12_scale, M.Base_scale, _mxp_host(M.Base), M.n)
mxp_host(M::SymmMixedPrec{T}) where {T} =
    SymmMixedPrec{T}(_mxp_host(M.A11), _mxp_host(M.A22), _mxp_host(M.OffDiag), M.OffDiag_scale,
                     M.Base_scale, _mxp_host(M.Base), M.uplo, M.n)
mxp_host(M::TriMixedPrec{T}) where {T} =
    TriMixedPrec{T}(_mxp_host(M.A11), _mxp_host(M.A22), _mxp_host(M.OffDiag), M.OffDiag_scale,
                    M.Base_scale, _mxp_host(M.Base), M.uplo, M.n)
