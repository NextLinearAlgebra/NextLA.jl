using LinearAlgebra


#=
Parameters
[in]	N	
!>          N is INTEGER
!>         number of elements in input vector(s)
!> 
[in,out]	SX	
!>          SX is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
!> 
[in]	INCX	
!>          INCX is INTEGER
!>         storage spacing between elements of SX
!> 
[in,out]	SY	
!>          SY is REAL array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
!> 
[in]	INCY	
!>          INCY is INTEGER
!>         storage spacing between elements of SY
!> 
[in]	C	
!>          C is REAL
!> 
[in]	S	
!>          S is REAL
!> 
=#

function srot!(n::S, sx::AbstractVector{T}, incx::S,
                sy::AbstractVector{T}, incy::S, c::T, s::T) where {T <:AbstractFloat, S <:Integer}

    if n <= 0
        return
    end

    stemp = 0
    if incx == 1 && incy == 1
        for i in 1:n
            stemp = c * sx[i] + s*sy[i]
            sy[i] = c * sy[i] - s*sx[i]
            sx[i] = stemp
        end
    else
        ix = 1
        iy = 1

        if incx < 0
            ix = (-n + 1)*incx + 1
        end

        if incy < 0
            iy = (-n + 1)*incy + 1
        end

        for i in 1:n
            stemp = c*sx[ix] + s*sy[iy]
            sy[iy] = c*sy[iy] - s*sx[ix]
            sx[ix] = stemp
            ix += incx
            iy += incy
        end
    end
end
