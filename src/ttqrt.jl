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

    #   original function had this todo:
    #   todo: Need to check why some cases require this to avoid
    #   uninitialized values
    #   core_zlaset(CoreBlasGeneral, ib, n, 0.0, 0.0, T, ldt);

    one = oneunit(eltype(A1))
    Tzero = zero(eltype(A1))

    k = min(m, n)
    for ii in 1:ib:k
        sb = min(k - ii + 1, ib)

        for i in 1:sb
            j = ii + i - 1 # index
            mi = min(j, m) # length
            ni = sb - i  # length

            A1[j, j], tau[j] = larfg!(mi + 1, A1[j, j], (@view A2[1:mi, j]), 1, tau[j])

            if ni > 0
                work[1:ni] .= (@view A1[j, j+1:j+ni])
                conj!((@view work[1:ni]))

                LinearAlgebra.generic_matvecmul!((@view work[1:ni]), 'C', (@view A2[1:mi, j+1:j+ni]),
                    (@view A2[1:mi, j]), LinearAlgebra.MulAddMul(one, one))
                conj!((@view work[1:ni]))

                alpha = -conj(tau[j])
                axpy!(alpha, (@view work[1:ni]), (@view A1[j, j+1:j+ni]))
                conj!((@view work[1:ni]))
                gerc!(alpha, (@view A2[1:mi, j]), (@view work[1:ni]), (@view A2[1:mi, j+1:j+ni]))
            end

            # calculate T
            if i > 1
                l = min(i - 1, max(0, m - ii + 1)) # length
                alpha = -tau[j]

                pemv!('C', 'C', min(j - 1, m), i - 1, l, alpha, (@view A2[1:m, ii:ii+i-2]),
                    (@view A2[1:m, j]), Tzero, (@view T_mat[1:i-1, j]), work)
                LinearAlgebra.generic_trimatmul!((@view T_mat[1:i-1, j]), 'U', 'N', identity, (@view T_mat[1:i-1, ii:ii+i-2]), (@view T_mat[1:i-1, j]))
            end

            T_mat[i, j] = tau[j]
        end

        if (n > ii + sb - 1)
            mi = min(ii + sb - 1, m)
            ni = n - (ii + sb) + 1
            l = min(sb, max(0, mi - ii + 1))
            # Workspace reshape for this call: sb x ni (left side)
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
    tau = Vector{T}(undef, k)

    work = zeros(T, ib * n)
    
    ttqrt!(m, n, ib, A, B, T_mat, tau, work)
end
