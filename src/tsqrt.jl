"""
    tsqrt!(m, n, ib, A1, A2, T, tau, work)

Compute the QR factorization of an (m+n)-by-n triangular-pentagonal matrix
using the compact WY representation.

This routine computes the QR factorization of a triangular-pentagonal matrix:
    [ A1 ]
    [ A2 ]
where A1 is n-by-n upper triangular and A2 is m-by-n general.

The factorization has the form:
    [ A1 ] = Q * [ R ]
    [ A2 ]       [ 0 ]
where Q is orthogonal and R is upper triangular.

# Arguments
- `m`: Number of rows of the pentagonal part A2
- `n`: Number of columns of the triangular-pentagonal matrix  
- `ib`: Block size for the compact WY representation
- `A1`: n×n upper triangular matrix (modified in-place)
- `A2`: m×n general matrix (modified in-place) 
- `T`: ib×n matrix to store block reflector coefficients
- `tau`: Vector of length n to store reflector scalar factors
- `work`: Workspace array of length ib×n

# Algorithm
The algorithm proceeds in blocks of size ib:
1. For each block, generate elementary reflectors to zero the pentagonal part
2. Apply reflectors to remaining columns using efficient block updates
3. Store reflector coefficients in compact WY form in matrix T

The compact WY representation allows for efficient application of the 
orthogonal factor Q using block operations.

# Input Validation
All dimension parameters must be non-negative and leading dimensions
must satisfy minimum requirements for valid matrix storage.

# Notes
This is a low-level computational routine typically called by higher-level
QR factorization interfaces. The matrices A1, A2 are modified in-place
to store the R factor and reflector vectors respectively.
"""
# Kernel: single work item runs one block (avoids host-side scalar indexing on GPU)
@kernel unsafe_indices=true function tsqrt_block_kernel!(A1, A2, T_matrix, tau, work, m::Int, n::Int, ib::Int, ii::Int, sb::Int)
    @uniform T = eltype(A1)
    @uniform one = T <: Complex ? Complex(Base.one(real(T)), Base.zero(real(T))) : oneunit(T)
    @uniform zero0 = T <: Complex ? Complex(Base.zero(real(T)), Base.zero(real(T))) : zero(T)

    for i in 1:sb
        r = ii + i - 1
        alpha = @inbounds A1[r, r]
        alphar, alphai = real(alpha), imag(alpha)
        xnorm_sq = zero(real(T))
        for p in 1:m
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
        for p in 1:m
            @inbounds A2[p, r] *= scale
        end
        @inbounds A1[r, r] = beta
        @inbounds tau[r] = tau_val

        if (sb - i) > 0 && (ii + i) <= n
            ncc = sb - i
            # work = W = C^H * v. LAPACK: A1 += -conj(tau)*conj(W), A2 += -conj(tau)*v*conj(W)^T
            alpha_apply = T <: Complex ? Complex(real(tau_val), -imag(tau_val)) : tau_val
            for j in 1:ncc
                wj = Complex(real(@inbounds A1[r, r+j]), -imag(@inbounds A1[r, r+j]))
                for p in 1:m
                    ap, ac = @inbounds A2[p, r+j], @inbounds A2[p, r]
                    wj += Complex(real(ap), -imag(ap)) * ac
                end
                @inbounds work[j] = Complex(real(wj), -imag(wj))
            end
            for j in 1:ncc
                # A1 update: work[j] = conj(W); LAPACK adds -conj(tau)*conj(W)
                w_conj = @inbounds work[j]  # already conj(W)
                @inbounds A1[r, r+j] -= alpha_apply * w_conj
            end
            for j in 1:ncc
                # A2 update: A2 -= conj(tau)*v*conj(W)^T; work[j] is conj(W)
                w_conj = @inbounds work[j]
                for p in 1:m
                    @inbounds A2[p, r+j] -= alpha_apply * @inbounds A2[p, r] * w_conj
                end
            end
        end

        if i > 1
            # T[1:i-1,r] := -tau * A2[:,ii:ii+i-2]^H * A2[:,r]
            for p in 1:i-1
                acc = zero0
                for q in 1:m
                    acc += Complex(real(@inbounds A2[q, ii+p-1]), -imag(@inbounds A2[q, ii+p-1])) * @inbounds A2[q, r]
                end
                @inbounds T_matrix[p, r] = (-tau_val) * acc
            end
            # T[1:i-1,r] := T[1:i-1,ii:ii+i-2] * T[1:i-1,r] (upper tri mat-vec, use work as temp)
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

function tsqrt!(m::Integer, n::Integer, ib::Integer, A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}, tau::AbstractVector{T}, work::AbstractVector{T}) where {T}
    if m < 0
        throw(ArgumentError("m must be non-negative, got $m"))
    end
    if n < 0
        throw(ArgumentError("n must be non-negative, got $n"))
    end
    if ib < 0
        throw(ArgumentError("ib must be non-negative, got $ib"))
    end

    m1 = size(A1, 1)
    k = min(m1, n)
    if m == 0 || n == 0 || ib == 0 || k == 0
        return
    end

    backend = KernelAbstractions.get_backend(A1)
    for ii in 1:ib:k
        sb = min(k - ii + 1, ib)
        tsqrt_block_kernel!(backend, 1)(A1, A2, T_matrix, tau, work, m, n, ib, ii, sb, ndrange=1)
        KernelAbstractions.synchronize(backend)
        if n >= ii + sb
            tsmqr!('L', 'C', sb, n - (ii + sb) + 1, m, n - (ii + sb) + 1, sb, ib,
                   (@view A1[ii:ii+sb-1, ii+sb:n]), (@view A2[1:m, ii+sb:n]),
                   (@view A2[1:m, ii:ii+sb-1]), (@view T_matrix[1:ib, ii:ii+sb-1]), work)
        end
    end
end

"""
    tsqrt!(A1, A2, ib) -> (A1, A2, T, tau)
    
Compute QR factorization of a triangular-pentagonal matrix using block algorithm.

This is a high-level interface that automatically allocates workspace and
computes the QR factorization of the combined matrix [A1; A2] where A1 is
upper triangular and A2 is general.

# Arguments
- `A1`: n×n upper triangular matrix (modified in-place to store R factor)
- `A2`: m×n general matrix (modified in-place to store reflector vectors)
- `ib`: Block size for the algorithm (typically 32-64 for good performance)

# Returns
- Modified `A1`: Contains the R factor of the QR factorization  
- Modified `A2`: Contains the elementary reflector vectors
- `T`: ib×n matrix containing block reflector coefficients
- `tau`: Length-n vector containing reflector scaling factors

# Input Validation
- A1 must be square (n×n)
- A2 must have same number of columns as A1 (m×n)
- Block size ib should be positive and ≤ n for efficiency

# Example
```julia
n, m = 6, 8
ib = 4
A1 = triu(randn(ComplexF64, n, n))  # Upper triangular
A2 = randn(ComplexF64, m, n)        # General matrix
A1_qr, A2_qr, T, tau = tsqrt!(copy(A1), copy(A2), ib)
```

# Algorithm Notes  
Uses blocked algorithm for efficiency with large matrices. The compact WY
representation (stored in T) enables efficient application of the Q factor.
"""
function tsqrt!(A1::AbstractMatrix{T}, A2::AbstractMatrix{T}, T_matrix::AbstractMatrix{T}) where{T}
    m1 = size(A1, 1)
    n = size(A1, 2)
    m = size(A2, 1) 
    ib, nb = size(T_matrix)

    if ib <= 0
        throw(ArgumentError("Block size ib must be positive, got $ib"))
    end
    
    k = min(m1, n)
    tau = similar(A1, k)
    work = similar(A1, ib * n)
    # Call the core computational routine
    tsqrt!(m, n, ib, A1, A2, T_matrix, tau, work)
end
