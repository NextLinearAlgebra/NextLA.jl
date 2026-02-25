# GPU kernel: single work item runs one block (avoids host-side scalar indexing)
@kernel unsafe_indices=true function ttqrt_block_kernel!(A1, A2, T_matrix, tau, work, m::Int, n::Int, ib::Int, ii::Int, sb::Int)
    @uniform T = eltype(A1)
    @uniform one = T <: Complex ? Complex(Base.one(real(T)), Base.zero(real(T))) : oneunit(T)
    @uniform zero0 = T <: Complex ? Complex(Base.zero(real(T)), Base.zero(real(T))) : zero(T)

    for i in 1:sb
        r = ii + i - 1
        mi = min(r, m)  # rows from A2 for this column
        alpha = @inbounds A1[r, r]
        alphar, alphai = real(alpha), imag(alpha)
        xnorm_sq = zero(real(T))
        for p in 1:mi
            vp = @inbounds A2[p, r]
            xnorm_sq += real(vp)^2 + imag(vp)^2
        end
        xnorm = sqrt(xnorm_sq)
        if xnorm == zero(real(T)) && alphai == zero(real(T))
            @inbounds tau[r] = zero0
            continue
        end
        beta = -copysign(sqrt(alphar^2 + alphai^2 + xnorm_sq), alphar)
        tau_val = T <: Complex ? Complex((beta - alphar) / beta, -alphai / beta) : (beta - alphar) / beta
        denom = alpha - beta
        dr, di = real(denom), imag(denom)
        abs2_d = dr * dr + di * di
        scale = T <: Complex ? Complex(dr / abs2_d, -di / abs2_d) : one / denom
        for p in 1:mi
            @inbounds A2[p, r] *= scale
        end
        @inbounds A1[r, r] = beta
        @inbounds tau[r] = tau_val

        if (sb - i) > 0 && (ii + i) <= n
            ncc = sb - i
            alpha_apply = T <: Complex ? Complex(real(tau_val), -imag(tau_val)) : tau_val
            for j in 1:ncc
                wj = Complex(real(@inbounds A1[r, r+j]), -imag(@inbounds A1[r, r+j]))
                for p in 1:mi
                    ap, ac = @inbounds A2[p, r+j], @inbounds A2[p, r]
                    wj += Complex(real(ap), -imag(ap)) * ac
                end
                @inbounds work[j] = Complex(real(wj), -imag(wj))
            end
            for j in 1:ncc
                w_conj = @inbounds work[j]
                @inbounds A1[r, r+j] -= alpha_apply * w_conj
            end
            for j in 1:ncc
                w_conj = @inbounds work[j]
                for p in 1:mi
                    @inbounds A2[p, r+j] -= alpha_apply * @inbounds A2[p, r] * w_conj
                end
            end
        end

        if i > 1
            mj = min(r - 1, m)  # effective rows for triangular A2 (pemv uses min(j-1, m))
            for p in 1:i-1
                acc = zero0
                for q in 1:mj
                    acc += Complex(real(@inbounds A2[q, ii+p-1]), -imag(@inbounds A2[q, ii+p-1])) * @inbounds A2[q, r]
                end
                @inbounds T_matrix[p, r] = (-tau_val) * acc
            end
            for p in 1:i-1
                @inbounds work[p] = @inbounds T_matrix[p, r]
            end
            for p in 1:i-1
                acc = zero0
                for q in p:i-1
                    acc += @inbounds T_matrix[p, ii+q-1] * @inbounds work[q]
                end
                @inbounds T_matrix[p, r] = acc
            end
        end
        @inbounds T_matrix[i, r] = tau_val
    end
end

function ttqrt!(m::Integer, n::Integer, ib::Integer, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, T_mat::AbstractMatrix{T}, tau::AbstractVector{T}, work::AbstractVector{T}) where {T}
    if m < 0
        throw(ArgumentError("illegal value of m"))
    end

    if n < 0
        throw(ArgumentError("illegal value of n"))
    end

    if ib < 0
        throw(ArgumentError("illegal value of ib"))
    end

    # quick return
    if m == 0 || n == 0 || ib == 0
        return
    end

    k = min(m, n)
    backend = KernelAbstractions.get_backend(A1)
    for ii in 1:ib:k
        sb = min(k - ii + 1, ib)
        ttqrt_block_kernel!(backend, 1)(A1, A2, T_mat, tau, work, m, n, ib, ii, sb, ndrange=1)
        KernelAbstractions.synchronize(backend)
        if n > ii + sb - 1
            mi = min(ii + sb - 1, m)
            ni = n - (ii + sb) + 1
            l = min(sb, max(0, mi - ii + 1))
            W = reshape(@view(work[1:sb*ni]), sb, ni)
            parfb!('L', 'C', 'F', 'C', ib, ni, mi, ni, sb, l,
                (@view A1[ii:ii+ib-1, ii+sb:ii+sb+ni-1]),
                (@view A2[1:mi, ii+sb:ii+sb+ni-1]),
                (@view A2[1:mi, ii:ii+sb-1]),
                (@view T_mat[1:sb, ii:ii+sb-1]),
                W)
        end
    end
end

"""
    ttqrt!(A, B, T_mat) -> nothing
    
Helper for triangular-triangular QR factorization.

# Arguments
- `A`: Upper triangular matrix (m × n), only upper triangle is accessed
- `B`: Upper triangular matrix (m2 × n2), m2 and n2 must equal size of A
- `T_mat`: ib × nb block reflector matrix

# Returns
- Modified `A` and `B` matrices in-place
"""
function ttqrt!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, T_mat::AbstractMatrix{T}) where {T}
    m, n = size(A)
    m2, n2 = size(B)
    ib, nb = size(T_mat)
    if n2 != n
        throw(ArgumentError("A and B must have the same number of columns: got $n and $n2"))
    end
    k = min(m, n)
    tau = similar(A, k)
    work = similar(A, ib * n)
    ttqrt!(m, n, ib, A, B, T_mat, tau, work)
end
