"""
    geqr2!(m, n, A, tau, work)

Compute unblocked QR factorization of an m-by-n matrix A using Householder reflectors.
The matrix A is overwritten with the Q and R factors.

# Arguments
- `m`: Number of rows in matrix A
- `n`: Number of columns in matrix A  
- `A`: Input matrix (m × n), modified in place to contain Q and R factors
- `tau`: Output vector of scalar factors (length min(m,n))
- `work`: Workspace vector (length n)

# Algorithm
Uses Householder reflectors H(i) to zero out elements below the diagonal.
For each column i, generates H(i) and applies it to remaining columns.

Implemented as a KernelAbstractions kernel (single work item), runs on any KA backend
(CPU, CUDA, ROCm, oneAPI, Metal) without CPU copies.
"""
# Kernel: single work item runs full sequential QR (larfg + larf inlined)
# unsafe_indices=true: no @index(Global) needed for single work-item
@kernel unsafe_indices=true function geqr2_kernel!(A, tau, work, m::Int, n::Int)
    @uniform k = min(m, n)
    @uniform T = eltype(A)
    # Avoid oneunit/zero - for Complex they can emit @_j_const_1 (im) which oneAPI SPIR-V rejects
    @uniform one = T <: Complex ? Complex(Base.one(real(T)), Base.zero(real(T))) : oneunit(T)
    @uniform zero0 = T <: Complex ? Complex(Base.zero(real(T)), Base.zero(real(T))) : zero(T)

    for i in 1:k
        len = m - i + 1  # length of column i
        if len <= 1
            @inbounds tau[i] = zero0
            continue
        end

        # --- larfg: generate reflector for column i ---
        alpha = @inbounds A[i, i]
        alphar = real(alpha)
        alphai = imag(alpha)
        xnorm_sq = zero(real(T))
        for p in 1:len-1
            vp = @inbounds A[i+p, i]
            xnorm_sq += real(vp)^2 + imag(vp)^2
        end
        xnorm = sqrt(xnorm_sq)

        if xnorm == zero(real(T)) && alphai == zero(real(T))
            @inbounds tau[i] = zero0
            continue
        end

        beta = -copysign(sqrt(alphar^2 + alphai^2 + xnorm_sq), alphar)
        tau_val = if T <: Complex
            # Avoid global `im` - oneAPI SPIR-V rejects addrspacecast of @_j_const_1.
            # Use Base.Complex(re, im) to avoid T(...) which KA may treat as calling a value.
            re = (beta - alphar) / beta
            im_part = -alphai / beta
            Complex(re, im_part)
        else
            (beta - alphar) / beta
        end
        # Avoid complex division - it can emit @_j_const_1. Use explicit 1/z = conj(z)/abs2(z)
        denom = alpha - beta
        dr, di = real(denom), imag(denom)
        abs2_d = dr * dr + di * di
        scale = T <: Complex ? Complex(dr / abs2_d, -di / abs2_d) : one / denom

        for p in 1:len-1
            @inbounds A[i+p, i] *= scale
        end
        @inbounds A[i, i] = beta
        @inbounds tau[i] = tau_val

        # --- larf: apply H(i)^H to A(i:m, i+1:n) from the left ---
        if i < n
            alpha_saved = beta
            @inbounds A[i, i] = one

            nr, nc = len, n - i
            # w = C^H * v  (v = column i)
            # Use Complex(real(x),-imag(x)) instead of conj(x) to avoid @_j_const_1 for oneAPI
            for j in 1:nc
                wj = zero0
                for p in 1:nr
                    ap = @inbounds A[i+p-1, i+j]
                    ac = @inbounds A[i+p-1, i]
                    wj += Complex(real(ap), -imag(ap)) * ac
                end
                @inbounds work[j] = wj
            end
            # C := C - conj(tau) * v * w^H  (apply H^H from left)
            tau_conj = T <: Complex ? Complex(real(tau_val), -imag(tau_val)) : tau_val
            for j in 1:nc
                wj = @inbounds work[j]
                wc_conj = T <: Complex ? Complex(real(wj), -imag(wj)) : wj
                for p in 1:nr
                    @inbounds A[i+p-1, i+j] -= tau_conj * @inbounds A[i+p-1, i] * wc_conj
                end
            end

            @inbounds A[i, i] = alpha_saved
        end
    end
end

function geqr2!(m::Integer, n::Integer, A::AbstractMatrix{T}, tau::AbstractVector{T}, work::AbstractVector{T}) where {T}
    if m < 0
        throw(ArgumentError("illegal value of m: $m"))
    end
    if n < 0
        throw(ArgumentError("illegal value of n: $n"))
    end
    if m == 0 || n == 0
        return
    end

    backend = KernelAbstractions.get_backend(A)
    geqr2_kernel!(backend, 1)(A, tau, work, m, n, ndrange=1)
    KernelAbstractions.synchronize(backend)
end

"""
    geqr2!(A) -> (A, tau)
    
Helper function for unblocked QR factorization using Householder reflectors.

# Arguments  
- `A`: Input matrix (m × n), modified in place
- `tau`: Output vector of scalar factors (length min(m,n))

# Returns
- Modified `A` containing Q and R factors
- `tau`: Vector of scalar factors (length min(m,n))
"""
function geqr2!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A)
    work = similar(A, n)
    geqr2!(m, n, A, tau, work)
end
