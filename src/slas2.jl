

"""
computes singular values of a 2-by-2 triangular matrix of the form:

 [[F, G],
  [0, H]]

  
[in, out]	A the 2x2 upper triangular matrix that will be used to compute the singular values
 


write singular values to the diagonal
"""
function slas2!(A::UpperTriangular{T, <:AbstractMatrix{T}}) where {T<:AbstractFloat}
    fa = abs(A[1,1])
    ga = abs(A[1,2])
    ha = abs(A[2,2])
    fhmn = min(fa, ha)
    fhmx = max(fa, ha)

    if fhmn == zero(fhmn)
        A[2,2] = zero(T)

        if fhmx == zero(fhmx)
            A[1,1] = ga
        else
            A[1,1] = max(fhmx, ga)*sqrt(one(T) + (min(fhmx , ga)/max(fhmx, ga))^2)

            #Simplified version. Causes less precision at extreme values, so don't use
            # ssmax[] = sqrt(fhmx^2 + ga^2)
        end
    else
        if ga < fhmx
            as = one(T) + fhmn/fhmx
            at = (fhmx - fhmn) /fhmx
            au = (ga/fhmx)^2
            c = (2*one(T))/(sqrt(as*as+au)+sqrt(at*at+au))
            A[2,2] = fhmn*c
            A[1,1]= fhmx/c
        else
            au = fhmx/ga
            if au == zero(au)
                #=
                Note from LAPACK function

                Avoid possible harmful underflow if exponent range
                asymmetric (true SSMIN may not underflow even if
                AU underflows)
                =#

                A[2,2] = (fhmn*fhmx)/ga
                A[1,1] = ga
            else
                as = one(T) + fhmn/fhmx
                at = (fhmx - fhmn)/fhmx
                c = one(T)/(sqrt(one(T) + (as*au)^2) + sqrt(one(T) + (at*au)^2))
                A[2,2] = 2*(fhmn*c)*au
                # A[2,2] = A[2,2] + A[2,2]
                A[1,1] = ga/((c+c))
            end

        end
    
    end

end
