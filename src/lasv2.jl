using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

#Smaller than zero cases will be -1 and >= 0 will be 1
function lasv2!(A::AbstractMatrix{T}, out_vec::AbstractVector{T}) where {T <: AbstractFloat}
    fa = abs(A[1,1])
    ha = abs(A[2,2])
    pmax = 1
    swap = ha > fa
    ssmin = 2
    ssmax = 1
    tsign = A[1,1] < 0 ? -1 : 1
    fh_sign = sign(A[1,1]) == sign(A[2,2]) ? 1 : -1

    if swap
        tsign = A[2,2] < 0 ? -1 : 1
        pmax = 3
        A[1,1], A[2,2] = A[2,2], A[1,1]
        fa, ha, = ha, fa
    end

    ga = abs( A[1,2])

    if ga == 0
        out_vec[ssmin] = ha
        out_vec[ssmax] = fa

        A[2,1] = zero(T) #slt
        A[2,2] = one(T) #crt
        A[1,1] = one(T) #clt
        A[1,2] = zero(T) #srt
    else
        gasmal = true

        if ga > fa
            pmax = 2
            tsign = A[1,2] < 0 ? -1 : 1

                mach_eps = 0

            if T == Float32
                mach_eps = ccall(
                                    (@blasfunc(slamch_), libblastrampoline),
                                    Float32,
                                    (Ref{UInt8},),
                                    UInt8('E')
                                )
            elseif T ==  Float64
                mach_eps = ccall(
                                    (@blasfunc(dlamch_), libblastrampoline),
                                    Float64,
                                    (Ref{UInt8},),
                                    UInt8('E')
                                )
            else
                mach_eps = eps(T)
            end
            if (fa/ga) < mach_eps
                gasmal = false
                out_vec[ssmax] = ga

                if ha > one(T)
                    out_vec[ssmin] = fa / (ga / ha)
                else
                    out_vec[ssmin] = (fa / ga) * ha
                end

                A[2,1] = A[2,2]/A[1,2] #slt
                A[2,2] = A[1,1]/A[1,2] #crt
                A[1,1] = one(T) #clt
                A[1,2] = one(T) #srt
            end
        end

        if gasmal

            #Normal case
            d = fa - ha
            if d == fa

                #Copes with infinite F or H
                l = one(T)
            else
                l = d/fa
            end
            #Note that 0 <= l <= 1

            m = A[1,2]/A[1,1]
            
            #Note that abs(m) <= 1/macheps
            t = T(2) - l
            
            #Note that t >= 1
            
            mm = m*m
            tt = t*t
            s = sqrt(tt+mm)

            #Note that 1 <= s <= 1 + 1/macheps

            #using A[2,2] as free memory
            A[2,1] = 0
            if l == zero(T)
                A[2,1] = abs(m)
            else
                A[2,1] = sqrt(l*l+mm)
            end

            a = T(0.5)*(s+A[2,1])

            out_vec[ssmin] = ha/a
            out_vec[ssmax] = fa*a

            if mm == zero(T)
                #Note that m is very tiny

                if l == zero(T)
                    t = copysign(T(2), A[1,1])*copysign(one(T), A[1,2])
                else
                    t = A[1,2] / copysign(d, A[1,1]) + m/t
                end
            else
                t = (m / (s + t)+m / (A[2,1] + l)) * (one(T) + a)
            end
            
            l = sqrt(t*t + T(4))

            A[1,2] = t/l #srt
            A[2,1] = (A[2,2] / A[1,1])*A[1,2] / a #slt
            A[2,2] = T(2)/l #crt
            A[1,1] = (A[2,2] + A[1,2]*m)/a #clt
        end
    end

    if swap
        #swap each row.
        A[1,1], A[1,2] = A[1,2], A[1,1]
        A[2,1], A[2,2] = A[2,2], A[2,1]
    end

    #=
        Now A[1,1] contains csl, A[1,2] contains snr,
            A[2,1] contains snl, and A[2,2] contains csr
    =#
    #Need to compute the singular vector components to figure out the sign
    if pmax == 1
        tsign =(A[2,2]*A[1,1] < 0 ? -1 : 1)*tsign
    elseif pmax == 2
        tsign = (A[1,2]*A[1,1] < 0 ? -1 : 1)*tsign
    elseif pmax == 3
        tsign = (A[1,2]*A[2,1] < 0 ? -1 : 1)*tsign
    end

    out_vec[ssmax] = copysign(out_vec[ssmax], tsign)
    out_vec[ssmin] = copysign(out_vec[ssmin], tsign*fh_sign)

end
