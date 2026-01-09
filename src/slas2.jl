

"""
computes singular values of a 2-by-2 triangular matrix of the form:

 [[F, G],
  [0, H]]

  
[in]	F	
          F is REAL
          The (1,1) element of the 2-by-2 matrix.
 
[in]	G	
          G is REAL
          The (1,2) element of the 2-by-2 matrix.
 
[in]	H	
          H is REAL
          The (2,2) element of the 2-by-2 matrix.
 
[out]	SSMIN	
          SSMIN is REAL
          The smaller singular value.
 
[out]	SSMAX	
          SSMAX is REAL
          The larger singular value.
 

"""
function slas2!(f::T, g::T, h::T, ssmin::Ref{T}, ssmax::Ref{T}) where {T<:AbstractFloat}
    fa = abs(f)
    ga = abs(g)
    ha = abs(h)
    fhmn = min(fa, ha)
    fhmx = max(fa, ha)

    if fhmn == zero(fhmn)
        ssmin[] = zero(T)

        if fhmx == zero(fhmx)
            ssmax[] = ga
        else

            ssmax[] = max(fhmx, ga)*sqrt(one(T) + (min(fhmx , ga)/max(fhmx, ga))^2)

            #Simplified version. Causes less precision at extreme values, so don't use
            # ssmax[] = sqrt(fhmx^2 + ga^2)
        end
    else
        if ga < fhmx
            as = one(T) + fhmn/fhmx
            at = (fhmx - fhmn) /fhmx
            au = (ga/fhmx)^2
            c = (2*one(T))/(sqrt(as*as+au)+sqrt(at*at+au))
            ssmin[] = fhmn*c
            ssmax[]= fhmx/c
        else
            au = fhmx/ga
            if au == zero(au)
                """
                Note from LAPACK function

                Avoid possible harmful underflow if exponent range
                asymmetric (true SSMIN may not underflow even if
                AU underflows)
                """

                ssmin[] = (fhmn*fhmx)/ga
                ssmax[] = ga
            else
                as = one(T) + fhmn/fhmx
                at = (fhmx - fhmn)/fhmx
                c = one(T)/(sqrt(one(T) + (as*au)^2) + sqrt(one(T) + (at*au)^2))
                ssmin[] = (fhmn*c)*au
                ssmin[] = ssmin[] + ssmin[]
                ssmax[] = ga/((c+c))
            end

        end
    
    end

end
