"""
    LASD7

Julia translation of LAPACK `*LASD7` (merge–deflate step for divide-and-conquer SVD), from
Reference-LAPACK `slasd7.f` / `dlasd7.f`.

# Algorithm (high level)

1. **Form the rank-one update vector `z`**: scale pieces of `vl`/`vf` by `α`/`β`, shift the
   leading block of `d`, `vf`, `idxq` one index right (Fortran DO 10), and zero the appropriate
   `vf`/`vl` entries.
2. **Apply `idxq` block transforms** (Fortran DO 30) so lower-block indices are global.
3. **Merge** the two ascending diagonal segments of `d` (length `nl` and `nr`, stored in
   `dsigma[2:n]`) via `slamrg` → permutation in `idx[2:n]`; scatter back into `d`, `z`, `vf`, `vl`.
4. **Deflate** the secular equation order: small `|zⱼ|` → move index to tail; near-equal
   `dⱼ - dⱼₚᵣₑᵥ ≤ tol` → apply a Givens on `(zⱼₚᵣₑᵥ, zⱼ)`, update `vf`/`vl`, record rotation
   if `icompq=1`, pack deflated indices (including the generalized `DO 85` fill).
5. **Pack** non-deflated data into `dsigma`, `z`, `vf`, `vl` workspaces; copy deflated tail
   into `d[k+1:n]`; fix `dsigma[2]` if tiny; finish `z[1]` (and `sqre` Givens on rows `m` and `1`
   with **BLAS `srot` vector order**: first index `m`, second `1`).

`T` may be `Float16`, `Float32`, or `Float64` (same code path; use `eps(T)` like `slamch('e')`).

# GPU / accelerators

Use `AbstractVector` / `AbstractMatrix` with in-place updates only; no heap allocs in the core
routine. Kernel launches can map one-to-one onto the explicit loops later.

# API

The public entry point is [`lasd7!`](@ref).
"""
# module LASD7

# export lasd7!

using LinearAlgebra: BlasInt

# Small BLAS/LAPACK building blocks: SLAMRG, SLAPY2 (hypot), plane rotation, machine epsilon.

@inline _eps(T::Type{<:AbstractFloat}) = eps(T)

"""Merge two sorted segments of `A` into ascending order; write 1-based indices into `index` (length n1+n2)."""
function slamrg_cursor!(
    index::AbstractVector{BlasInt},
    A::AbstractVector{T},
    n1::Integer,
    n2::Integer,
    strd1::Integer,
    strd2::Integer,
) where {T <: AbstractFloat}
    n1sv = n1
    n2sv = n2
    ind1 = strd1 > 0 ? 1 : n1
    ind2 = strd2 > 0 ? n1 + 1 : n1 + n2
    i = 1
    @inbounds while n1sv > 0 && n2sv > 0
        if A[ind1] <= A[ind2]
            index[i] = ind1
            i += 1
            ind1 += strd1
            n1sv -= 1
        else
            index[i] = ind2
            i += 1
            ind2 += strd2
            n2sv -= 1
        end
    end
    @inbounds if n1sv == 0
        for _ = 1:n2sv
            index[i] = ind2
            i += 1
            ind2 += strd2
        end
    else
        for _ = 1:n1sv
            index[i] = ind1
            i += 1
            ind1 += strd1
        end
    end
    return
end

"""SLAPY2 / `hypot`: √(|x|²+|y|²) with stable scaling."""
@inline lapy2(x::T, y::T) where {T <: AbstractFloat} = hypot(x, y)

@inline rot_pair!(x::AbstractVector{T}, idx1::Int, idx2::Int, c::T, s::T) where {T <: AbstractFloat} =
    @inbounds begin
        xi = x[idx1]
        yi = x[idx2]
        x[idx1] = muladd(c, xi, s * yi)
        x[idx2] = muladd(-s, xi, c * yi)
    end

"""
    lasd7!(icompq, nl, nr, sqre,
           k_ref, d, z, zw, vf, vfw, vl, vlw, alpha, beta,
           dsigma, idx, idxp, idxq, perm,
           givptr_ref, givcol, ldgcol, givnum, ldgnum,
           c_ref, s_ref, info_ref) -> Nothing

LAPACK-compatible port of **`SLASD7` / `DLASD7`**. All integer scalars follow LAPACK conventions
(`icompq ∈ {0,1}`, `sqre ∈ {0,1}`, `nl ≥ 1`, `nr ≥ 1`, …).

Outputs written to `k_ref`, `info_ref`, and optionally `givptr_ref`, `c_ref`, `s_ref`.
Arrays must use **1-based** indexing (`Base.require_one_based_indexing` assumed by callers).

# GPU readiness
- Uses only `AbstractVector` / `AbstractMatrix` indexing (no allocating temporaries besides locals).
- No type piracy; `T`-parameterized internals can be overloaded for GPU arrays later.

# References
Gu & Ren; Reference-LAPACK `SRC/slasd7.f`.
"""
function lasd7_cursor!(
    icompq::Integer,
    nl::Integer,
    nr::Integer,
    sqre::Integer,
    k_ref::Base.RefValue{BlasInt},
    d::AbstractVector{T},
    z::AbstractVector{T},
    zw::AbstractVector{T},
    vf::AbstractVector{T},
    vfw::AbstractVector{T},
    vl::AbstractVector{T},
    vlw::AbstractVector{T},
    alpha::T,
    beta::T,
    dsigma::AbstractVector{T},
    idx::AbstractVector{BlasInt},
    idxp::AbstractVector{BlasInt},
    idxq::AbstractVector{BlasInt},
    perm::AbstractVector{BlasInt},
    givptr_ref::Base.RefValue{BlasInt},
    givcol::AbstractMatrix{BlasInt},
    ldgcol::Integer,
    givnum::AbstractMatrix{T},
    ldgnum::Integer,
    c_ref::Base.RefValue{T},
    s_ref::Base.RefValue{T},
    info_ref::Base.RefValue{BlasInt},
) where {T <: AbstractFloat}
    info_ref[] = zero(BlasInt)
    n = nl + nr + 1
    m = n + sqre

    # ---- argument checks (LAPACK INFO) ----
    if !(icompq in (0, 1))
        info_ref[] = -1
        return
    elseif nl < 1
        info_ref[] = -2
        return
    elseif nr < 1
        info_ref[] = -3
        return
    elseif !(sqre in (0, 1))
        info_ref[] = -4
        return
    elseif ldgcol < n
        info_ref[] = -22
        return
    elseif ldgnum < n
        info_ref[] = -24
        return
    end

    zeroT = zero(T)
    oneT = one(T)
    twoT = oneT + oneT
    eight = T(8)

    nlp1 = nl + 1
    nlp2 = nl + 2

    if icompq == 1
        givptr_ref[] = 0
    end

    # Z1, shift D / VF / IDXQ over [1:nl]
    z1 = alpha * vl[nlp1]
    vl[nlp1] = zeroT
    tau = vf[nlp1]
    @inbounds for i = nl:-1:1
        z[i+1] = alpha * vl[i]
        vl[i] = zeroT
        vf[i+1] = vf[i]
        d[i+1] = d[i]
        idxq[i+1] = idxq[i] + 1
    end
    vf[1] = tau

    @inbounds for i = nlp2:m
        z[i] = beta * vf[i]
        vf[i] = zeroT
    end

    @inbounds for i = nlp2:n
        idxq[i] = idxq[i] + nlp1
    end

    @inbounds for i = 2:n
        dsigma[i] = d[Int(idxq[i])]
        zw[i] = z[Int(idxq[i])]
        vfw[i] = vf[Int(idxq[i])]
        vlw[i] = vl[Int(idxq[i])]
    end

    slamrg_cursor!(
        view(idx, 2:n),
        view(dsigma, 2:n),
        nl,
        nr,
        1,
        1,
    )

    @inbounds for i = 2:n
        idxi = 1 + Int(idx[i])
        d[i] = dsigma[idxi]
        z[i] = zw[idxi]
        vf[i] = vfw[idxi]
        vl[i] = vlw[idxi]
    end

    epss = _eps(T)
    tol = max(abs(alpha), abs(beta))
    tol = eight * eight * epss * max(abs(d[n]), tol)

    # ---- deflation scans (Fortran labels 60 / 70 / 80 / 90 / 100) ----
    k = 1
    k2 = n + 1
    early100 = false
    jprev = 2

    @inbounds for j = 2:n
        if abs(z[j]) <= tol
            k2 -= 1
            idxp[k2] = BlasInt(j)
            if j == n
                early100 = true
                break
            end
        else
            jprev = j
            break
        end
    end

    if !early100
        j = jprev
        @inbounds while true
            j += 1
            if j > n
                k += 1
                zw[k] = z[jprev]
                dsigma[k] = d[jprev]
                idxp[k] = BlasInt(jprev)
                break
            end
            if abs(z[j]) <= tol
                k2 -= 1
                idxp[k2] = BlasInt(j)
            else
                # D is sorted ascending after merge; LAPACK uses (D(J)-D(JPREV)) <= TOL
                if (d[j] - d[jprev]) <= tol
                    s = z[jprev]
                    c_rot = z[j]
                    tauh = lapy2(c_rot, s)
                    z[j] = tauh
                    z[jprev] = zeroT
                    c_rot = c_rot / tauh
                    s = -(s / tauh)

                    if icompq == 1
                        givptr_ref[] += 1
                        gp = Int(givptr_ref[])
                        idxjp = idxq[Int(idx[Int(jprev)]) + 1]
                        idxjj = idxq[Int(idx[Int(j)]) + 1]
                        if idxjp <= nlp1
                            idxjp -= BlasInt(1)
                        end
                        if idxjj <= nlp1
                            idxjj -= BlasInt(1)
                        end
                        givcol[gp, 2] = idxjp
                        givcol[gp, 1] = idxjj
                        givnum[gp, 2] = c_rot
                        givnum[gp, 1] = s
                    end
                    rot_pair!(vf, jprev, Int(j), c_rot, s)
                    rot_pair!(vl, jprev, Int(j), c_rot, s)
                    k2 -= 1
                    for jp = jprev:(j-1)
                        idxp[k2 + j - 1 - jp] = BlasInt(jp)
                    end
                    jprev = j
                else
                    k += 1
                    zw[k] = z[jprev]
                    dsigma[k] = d[jprev]
                    idxp[k] = BlasInt(jprev)
                    jprev = j
                end
            end
        end
    end

    @inbounds for j = 2:n
        jp = Int(idxp[j])
        dsigma[j] = d[jp]
        vfw[j] = vf[jp]
        vlw[j] = vl[jp]
    end

    if icompq == 1
        @inbounds for j = 2:n
            jp = Int(idxp[j])
            perm[j] = idxq[Int(idx[jp]) + 1]
            if perm[j] <= nlp1
                perm[j] -= 1
            end
        end
    end

    # Deflated tails -> D[K+1:N]
    nk = n - k
    if nk > 0
        copyto!(view(d, (k+1):n), view(dsigma, (k+1):n))
    end

    dsigma[1] = zeroT
    hlftol = tol / twoT
    if abs(dsigma[2]) <= hlftol
        dsigma[2] = hlftol
    end

    if m > n
        z[1] = lapy2(z1, z[m])
        if z[1] <= tol
            c_ref[] = oneT
            s_ref[] = zeroT
            z[1] = tol
        else
            c_ref[] = z1 / z[1]
            s_ref[] = -z[m] / z[1]
        end
        rot_pair!(vf, m, 1, c_ref[], s_ref[])
        rot_pair!(vl, m, 1, c_ref[], s_ref[])
    else
        if abs(z1) <= tol
            z[1] = tol
        else
            z[1] = z1
        end
    end

    if k > 1
        copyto!(view(z, 2:k), view(zw, 2:k))
    end
    copyto!(view(vf, 2:n), view(vfw, 2:n))
    copyto!(view(vl, 2:n), view(vlw, 2:n))

    k_ref[] = BlasInt(k)
    return
end

#end # module
