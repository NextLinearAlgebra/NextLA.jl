using LinearAlgebra


#=

Parameters
[in]	N1	
!>          N1 is INTEGER
!> 
[in]	N2	
!>          N2 is INTEGER
!>         These arguments contain the respective lengths of the two
!>         sorted lists to be merged.
!> 
[in]	A	
!>          A is REAL array, dimension (N1+N2)
!>         The first N1 elements of A contain a list of numbers which
!>         are sorted in either ascending or descending order.  Likewise
!>         for the final N2 elements.
!> 
[in]	STRD1	
!>          STRD1 is INTEGER
!> 
[in]	STRD2	
!>          STRD2 is INTEGER
!>         These are the strides to be taken through the array A.
!>         Allowable strides are 1 and -1.  They indicate whether a
!>         subset of A is sorted in ascending (STRDx = 1) or descending
!>         (STRDx = -1) order.
!> 
[out]	INDEX	
!>          INDEX is INTEGER array, dimension (N1+N2)
!>         On exit this array will contain a permutation such that
!>         if B( I ) = A( INDEX( I ) ) for I=1,N1+N2, then B will be
!>         sorted in ascending order.
!> 

=#
function slamrg!(n1::T, n2::T, a::AbstractVector{S},
                 strd1::T, strd2::T, index::AbstractVector{T}) where {T <: Integer, S<:AbstractFloat}

    n1sv = n1
    n2sv = n2
    ind1 = 0
    ind2 = 0
    if strd1 > 0
        ind1 = 1
    else
        ind1 = n1
    end

    if strd2 > 0 
        ind2 = 1 + n1
    else 
        ind2 = n1 + n2
    end
    i = 1
    while n1sv > 0 && n2sv > 0
        if a[ind1] <= a[ind2]
            index[i] = ind1
            i = i+1
            ind1 = ind1 + strd1
            n1sv -= 1
        else
            index[i] = ind2
            i = i + 1
            ind2 = ind2 + strd2
            n2sv -= 1
        end
    end

    if n1sv == 0
        for n1sv in 1:n2sv
            index[i] = ind2
            i = i+1
            ind2 = ind2 + strd2
        end
    else
        for n2sv in 1:n1sv
            index[i] = ind1
            i = i+1
            ind1 = ind1 + strd1
        end

    end
    return
end
