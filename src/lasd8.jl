using LinearAlgebra



#=
Purpose:
!>
!> SLASD8 finds the square roots of the roots of the secular equation,
!> as defined by the values in DSIGMA and Z. It makes the appropriate
!> calls to SLASD4, and stores, for each  element in D, the distance
!> to its two nearest poles (elements in DSIGMA). It also updates
!> the arrays VF and VL, the first and last components of all the
!> right singular vectors of the original bidiagonal matrix.
!>
!> SLASD8 is called from SLASD6.
!> 
Parameters
[in]	ICOMPQ	
!>          ICOMPQ is INTEGER
!>          Specifies whether singular vectors are to be computed in
!>          factored form in the calling routine:
!>          = 0: Compute singular values only.
!>          = 1: Compute singular vectors in factored form as well.
!> 
[in]	K	
!>          K is INTEGER
!>          The number of terms in the rational function to be solved
!>          by SLASD4.  K >= 1.
!> 
[out]	D	
!>          D is REAL array, dimension ( K )
!>          On output, D contains the updated singular values.
!> 
[in,out]	Z	
!>          Z is REAL array, dimension ( K )
!>          On entry, the first K elements of this array contain the
!>          components of the deflation-adjusted updating row vector.
!>          On exit, Z is updated.
!> 
[in,out]	VF	
!>          VF is REAL array, dimension ( K )
!>          On entry, VF contains  information passed through DBEDE8.
!>          On exit, VF contains the first K components of the first
!>          components of all right singular vectors of the bidiagonal
!>          matrix.
!> 
[in,out]	VL	
!>          VL is REAL array, dimension ( K )
!>          On entry, VL contains  information passed through DBEDE8.
!>          On exit, VL contains the first K components of the last
!>          components of all right singular vectors of the bidiagonal
!>          matrix.
!> 
[out]	DIFL	
!>          DIFL is REAL array, dimension ( K )
!>          On exit, DIFL(I) = D(I) - DSIGMA(I).
!> 
[out]	DIFR	
!>          DIFR is REAL array,
!>                   dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and
!>                   dimension ( K ) if ICOMPQ = 0.
!>          On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not
!>          defined and will not be referenced.
!>
!>          If ICOMPQ = 1, DIFR(1:K,2) is an array containing the
!>          normalizing factors for the right singular vector matrix.
!> 
[in]	LDDIFR	
!>          LDDIFR is INTEGER
!>          The leading dimension of DIFR, must be at least K.
!> 
[in]	DSIGMA	
!>          DSIGMA is REAL array, dimension ( K )
!>          On entry, the first K elements of this array contain the old
!>          roots of the deflated updating problem.  These are the poles
!>          of the secular equation.
!> 
[out]	WORK	
!>          WORK is REAL array, dimension (3*K)
!> 
[out]	INFO	
!>          INFO is INTEGER
!>          = 0:  successful exit.
!>          < 0:  if INFO = -i, the i-th argument had an illegal value.
!>          > 0:  if INFO = 1, a singular value did not converge

=#

function lasd8!(icompq::S, k::S, d::AbstractVector{T}, z::AbstractVector{T},
                vf::AbstractVector{T}, vl::AbstractVector{T}, difl::AbstractVector{T},
                difr::AbstractMatrix{T}, lddifr::S,
                dsigma::AbstractVector{T}, work::AbstractVector{T}, info::AbstractVector{S}) where {T <: AbstractFloat, S<:Integer}
    #=
        Important to note that info is just a vector of 1x1. We used a vector so we
            can preallocate the memory
    =#
    
    info .= 0
    # println("Starting function")
    if icompq < 0 || icompq > 1
        info[1] = -1
    elseif k < 1
        info[1] = -2
    elseif lddifr < k 
        info[1] = -9
    end

    if info[1] != 0
        return
    end

    if k == 1
        d[1] = abs(z[1])
        difl[1] = d[1]
        if icompq == 1
            difl[2] = one(T)
            difr[ 1, 2] = one(T)
        end
        return
    end

    #Book keeping

    iwk1 = one(k)
    iwk2 = iwk1 + k
    iwk3 = iwk2 + k
    iwk2i = iwk2 -1
    iwk3i = iwk3-1

    #normalize z
    rho = norm(z)
    if rho == 0 || isnan(rho)
        info .= -4
        return
    end
    z ./= rho
    rho *= rho
    #initialize work[iwk3]
    work[iwk3:iwk3+k-1] .= one(T)
    # println("z: $z")
    
    #Compute the updated singular values, the arrays difl, difr, and the updated z
    # println("k: $k")
    for j in 1:k
        #Need to add this function
        # println("Starting slasd4")
        # println("slasd8 disgma length: $(length(dsigma))")
        # println("work: $work")
        # println("dsigma: $dsigma")
        # println("z: $z")
        # println("d: $d")
        # println("info: $info")
        # println("lasd8 sigma before: $(d[j])")
        lasd4!(k, j, dsigma, z, view(work, iwk1:iwk2-1), rho,
        view(d, j:j), view(work, iwk2:iwk3-1), info)
        # println("lasd8 sigma after: $(d[j])")
        # println("work: $work")
        # println("dsigma: $dsigma")
        # println("z: $z")
        # println("d: $d")
        # println("info: $info")
        # println("")
        # println("Finishing slasd4")

        if info[1] != 0
            return
        end

        work[iwk3i + j] = work[iwk3i + j] * work[j]*work[iwk2i+j]
        difl[j] = -work[j]
        if icompq == 1
            difr[j,1] = -work[j+1]
        else
            difr[j] = -work[j+1]
        end

        for i in 1:j-1
            work[iwk3i+i] = (work[iwk3i+i]*work[i]*
                            work[iwk2i+i]/(dsigma[i] - dsigma[j])
                            /(dsigma[i]+dsigma[j]))

        end
        for i in j+1:k
            work[iwk3i+i] = (work[iwk3i+i]*work[i]*
                            work[iwk2i+i]/(dsigma[i] - dsigma[j])
                            /(dsigma[i]+dsigma[j]))

        end
    end


    for i in 1:k
        z[i] = (z[i] > 0 ? 1 : -1)*sqrt(abs(work[iwk3i+i]))
    end

    for j in 1:k
        diflj = difl[j]
        dj = d[j]
        dsigj = -dsigma[j]

        if j < k
            difrj = icompq == 1 ? -difr[j, 1] : -difr[j]
            dsigjp = -dsigma[j+1]
        end

        work[j] = -z[j] / diflj / (dsigma[j] + dj)

        for i in 1:j-1
            work[i] = z[i] / ((dsigma[i] + dsigj) - diflj) / (dsigma[i] + dj)
        end
        
        for i in j+1:k
            work[i] = z[i] / ((dsigma[i] + dsigjp) + difrj) / (dsigma[i] + dj)
        end

        temp = norm(view(work, 1:k))

        work[iwk2i + j] = dot(view(work, 1:k), vf) / temp
        work[iwk3i + j] = dot(view(work, 1:k), vl) / temp

        if icompq == 1
            difr[j, 2] = temp
        end
    end

    copyto!(vf, view(work, iwk2:iwk3-1))
    copyto!(vl, view(work, iwk3:iwk3+k-1))
    # println("d at the end of slasd8: $d")
    # println("work at the end of slasd8: $work")

end
