using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

function lasd4!(n::S, i::S, d::AbstractVector{T}, z::AbstractVector{T},
                delta::AbstractVector{T}, rho::T, sigma::AbstractArray{T}, work::AbstractVector{T},
                info::AbstractArray{S}) where {T <: AbstractFloat, S<:Integer}
    # println("Starting lasd4")
    zz = zeros(T, 3)
    dd = zeros(T, 3) 
    # println("Done allocating starting arrays")
    #Since this routine is called in an inner loop,
    #   we do no argument checking
    maxit = 400
    info .= 0
    eta = zero(T)
    # println("slasd4 d length: $(length(d))")
    if n == 1
        sigma .= sqrt(d[1]^2 + rho * z[1]^2)
        delta[1] = one(T)
        work[1] = one(T)
        return
    end

    if n == 2
        lasd5!(i, d, z, delta, rho, sigma, work)
        return
    end

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

    rhoinv = one(T)/rho

    tau2 = zero(T)
    tau = zero(T)
    #Quick return for N=1 and 2.

    if i == n

        # Initialize some basic variables
        ii = n-1
        niter = 1

        temp = rho/T(2)

        temp1 = temp / (d[n] + sqrt(d[n]^2 + temp))

        for j in 1:n
            work[j] = d[j] + d[n] + temp1
            delta[j] = (d[j] - d[n]) - temp1
        end

        psi = zero(T)
        for j in 1:n-2
            psi +=  z[j]^2/ (delta[j]*work[j])
        end

        c = rhoinv + psi
        
        w = c + z[ii]^2 / (delta[ii]*work[ii]) + z[n]^2 / (delta[n]*work[n])
        orgati = false

        if w <= 0
            temp1 = sqrt(d[n]^2 + rho)
            temp = z[n-1]^2/ ((d[n-1]+temp1) * (d[n] - d[n-1] + rho/(d[n]+temp1))) + z[n]^2/rho

            #The following tau2 is to approximate sigma_n^2 - d[n]*d[n]

            if c <= temp
                tau = rho
            else
                delsq = (d[n]-d[n-1])*(d[n]+d[n-1])
                a = -c*delsq + z[n-1]^2 + z[n]^2
                b = z[n]^2*delsq

                if a < 0
                    tau2 = T(2)*b / (sqrt(a^2+T(4)*b*c)-a)
                else
                    tau2 = (a + sqrt(a^2+T(4)*b*c))/(T(2)*c)
                end
                tau = tau2 / (d[n]+sqrt(d[n]^2+tau2))

            end

            #It can be proved that 
            # d[n]^2+rho/2 <= sigma_n^2 < d[n]^2+tau <= d[n]^2+rho

        else
            delsq = (d[n] - d[n-1])*(d[n] + d[n-1])
            a = -c*delsq + z[n-1]^2 + z[n]^2
            b = z[n]^2*delsq

            # The following tau2 is to approximate sigma_n^2 - d[n]*d[n]

            if a < 0
                tau2 = T(2)*b / (sqrt(a^2 + T(4)*b*c) - a)
            else
                tau2 = (a + sqrt(a^2+T(4)*b*c))/(T(2)*c)
            end

            tau = tau2 / (d[n] + sqrt(d[n]^2+tau2))

            #it can be proved that 
            # d[n]^2 < d[n]^2 + tau2 < sigma[n]^2 < d[n]^2 + rho/2
        end

        # println("lasd4 sigma before: $sigma")
        sigma .= d[n] + tau
        # println("lasd4 sigma after: $sigma")

        for j in 1:n
            delta[j]  = (d[j] - d[n]) - tau
            work[j] = d[j] + d[n] + tau
        end

        dpsi = zero(T)
        psi = zero(T)
        erretm = zero(T)

        for j in 1:ii
            temp = z[j] / (delta[j]*work[j])
            psi +=  z[j]*temp
            dpsi +=  temp^2
            erretm += psi
        end

        erretm = abs(erretm)


        #evaluate phi and the derivative dphi


        temp = z[n] / (delta[n]*work[n])
        phi = z[n]*temp
        dphi = temp^2
        erretm += T(8)*(-phi -psi) - phi + rhoinv

        w = rhoinv + phi + psi

        if abs(w) <= mach_eps*erretm
            return
        end

        niter = niter + 1
        dtnsq1 = work[n-1]*delta[n-1]
        dtnsq = work[n]*delta[n]

        c = w - dtnsq1*dpsi - dtnsq*dphi
        a = (dtnsq + dtnsq1)*w - dtnsq*dtnsq1*(dpsi+dphi)
        b = dtnsq*dtnsq1*w
        if c < 0
            c = abs(c)
        end

        if c == 0
            eta = rho - sigma[]^2
        elseif a >= 0
            eta = (a + sqrt(abs(a^2 - T(4)*b*c)))/(T(2)*c)
        else
            eta = T(2) * b / (a - sqrt(abs(a^2 - T(4)*b*c)))
        end

        #=
            Note, eta should be positive if w is negative, and eta should be negative otherwise. 
            However, if for some reason caused by roundoff, eta*w > 0, we simply use one Newton
            step isntead. This way will guarantee eta*w < 0
        =#

        if w*eta > 0
            eta = -w / (dpsi +dphi)
        end

        temp = eta - dtnsq

        if temp > rho
            eta = rho + dtnsq
        end

        if eta + sigma[]^2 < 0
            # println("Returning here")
            # println("$T Returning here2")

            info .= 1
            sigma .= NaN
            delta .= NaN
            work .= NaN
            return
        end
        eta /= ( sigma[] + sqrt(eta + sigma[]^2))
        tau += eta
        # println("lasd4 sigma before: $sigma")
        sigma .+= eta
        # println("lasd4 sigma after: $sigma")

        for j in 1:n
            delta[j] -= eta
            work[j] += eta
        end

        #evaluate psi and the derivative dpsi
        dpsi = zero(T)
        psi = zero(T)

        erretm = zero(T)


        for j in 1:ii
            temp = z[j] / (work[j]*delta[j])
            psi += z[j]*temp
            dpsi += temp^2
            erretm = erretm + psi
        end

        # evaluate phi and the derivative dphi
        erretm = abs(erretm)

        tau2 = work[n]*delta[n]
        temp = z[n] / tau2
        phi = z[n]*temp
        dphi = temp^2
        erretm += T(8)*(-phi -psi) - phi + rhoinv

        w = rhoinv + phi + psi

        iter = niter + 1

        for niter in iter:maxit

            #test for convergence
            if abs(w) <= mach_eps * erretm
                return
            end

            dtnsq1 = work[n-1]*delta[n-1]
            dtnsq = work[n]*delta[n]
            c = w - dtnsq1*dpsi - dtnsq*dphi
            a = ( dtnsq + dtnsq1)*w - dtnsq1*dtnsq*(dpsi+dphi)
            b = dtnsq1*dtnsq*w

            if a >= 0
                eta = (a + sqrt(abs(a^2 - T(4)*b*c)))/(T(2)*c)
            else
                eta = T(2)*b / (a - sqrt(abs(a^2 - T(4)*b*c)))
            end
            #=
                Note, eta should be positive if w is negative, and eta should be negative otherwise. 
                However, if for some reason caused by roundoff, eta*w > 0, we simply use one Newton
                step isntead. This way will guarantee eta*w < 0
            =#

            if w*eta > 0
                eta = -w/(dpsi+dphi)
            end
            temp = eta - dtnsq

            if temp <= 0
                eta /= T(2)
            end

            if eta + sigma[]^2 < 0
                # println("$T Returning here1")
                info .= 1
                sigma .= NaN
                delta .= NaN
                work .= NaN
                return
            end

            eta /= (sigma[] + sqrt(eta + sigma[]^2))
            tau +=  eta
            # println("lasd4 sigma before: $sigma")
            sigma .+= eta
            # println("lasd4 sigma after: $sigma")

            for j in 1:n
                delta[j] -=  eta
                work[j] += eta
            end

            dpsi = zero(T)
            psi = zero(T)
            erretm = zero(T)

            for j in 1:ii
                temp = z[j] / (work[j]*delta[j])
                psi +=  z[j]*temp
                dpsi += temp^2
                erretm += psi
            end

            erretm = abs(erretm)

            tau2 = work[n]* delta[n]
            temp = z[n] / tau2
            phi = z[n] * temp
            dphi = temp^2
            erretm += T(8)*(-phi - psi) - phi + rhoinv

            w = rhoinv + phi + psi

        end
        # println("Returning with fail")
        #return with info = 1, niter = maxit and not converged
        info .= 1
        return
    else
        #the case for i < n

        niter = 1
        ip1 = i + 1
        
        #calculate initial  guess
        
        delsq = (d[ip1] - d[i])*(d[ip1]+d[i])
        delsq2 = delsq / T(2)
        sq2 = sqrt((d[i]^2 + d[ip1]^2) / T(2))
        temp = delsq2 / (d[i] + sq2)
        
        for j in 1:n
            work[j] = d[j] + d[i] + temp
            delta[j] = (d[j] - d[i]) - temp
        end
        
        psi = zero(T)
        for j in 1:i-1
            psi += z[j]^2/(work[j]*delta[j])
        end
        
        phi = zero(T)
        
        for j in n:-1:i+2
            phi += z[j]^2 / ( work[j]*delta[j])
        end
        
        c = rhoinv + psi + phi
        w = c + z[i]^2/(work[i]*delta[i]) + z[ip1]^2 / (work[ip1]*delta[ip1])
        geomavg = false
        
        if w > 0
            # d[i]^2 < the ith sigma^2 < (d[i]^2 + d[i+1]^2)/2

            orgati = true

            ii = i
            sglb = zero(T)
            sgub = delsq2 / (d[i] + sq2)
            a = c*delsq + z[i]^2 + z[ip1]^2
            b = z[i]^2*delsq

            if a > 0
                tau2 = T(2)*b / (a + sqrt(abs(a^2 - T(4)*b*c)))
            else
                tau2 = (a - sqrt(abs(a^2 - T(4)*b*c))) / (T(2)*c)
            end

            #=
                tau2 now is an estimation of sigma^2 - d[i]^2. The
                following, however, is the corresponding estimation of sigma - d[i]
            =#

            tau = tau2 / (d[i] + sqrt(d[i]^2 + tau2))
            temp = sqrt(mach_eps)

            if d[i] <= temp*d[ip1] && (abs(z[i]) <= temp) && (d[i] > 0)
                tau = min(T(10)*d[i], sgub)
                geomavg = true
            end
        else
            # (d[i]^2 + d[i+1]^2)/2 <= the ith sigma^2 < d[i+1]^2/2

            #We choose d[i+1] as origin

            orgati = false
            ii = ip1
            sglb = -delsq2 / (d[ii] + sq2)
            sgub = zero(T)
            a = c* delsq - z[i]^2 - z[ip1]^2
            b = z[ip1]^2*delsq

            if a < 0
                tau2 = T(2)*b/ (a - sqrt(abs(a^2 + T(4)*b*c)))
            else
                tau2 = -(a + sqrt(abs(a^2 + T(4)*b*c)))/(T(2)*c)
            end

            #=
            tau now is an estimation of sigma^2 - d[ip1]^2. The 
            following, however, is the correpsoingind estimation of 
            sigma - d[ip1]
            =#

            tau = tau2 / (d[ip1] + sqrt(abs(d[ip1]^2 + tau2)))

        end

        # println("lasd4 sigma before: $sigma")
        sigma .= d[ii] + tau
        # println("lasd4 sigma after: $sigma")

        for j in 1:n
            work[j] = d[j] + d[ii] + tau
            delta[j] = (d[j] - d[ii]) - tau
        end

        iim1 = ii - 1
        iip1 = ii + 1

        #evaluate psi and the derivative dpsi

        dpsi = zero(T)
        psi = zero(T)
        erretm = zero(T)

        for j in 1:iim1
            temp = z[j] / (work[j]*delta[j])
            psi +=  z[j]*temp
            dpsi += temp^2
            erretm += psi
        end
        erretm = abs(erretm)

        #evaluate phi and the derivative dphi

        dphi = zero(T)
        phi = zero(T)

        for j in n:-1:iip1
            temp = z[j] / (work[j]*delta[j])
            phi += z[j]*temp
            dphi += temp^2
            erretm +=  phi
        end

        w = rhoinv + phi + psi

        # W is the value of the secular function with its ii-th element removed.

        swtch3 = false

        if orgati 
            if w < 0
                swtch3 = true
            end
        else
            if w > 0
                swtch3 = true
            end
        end
        if ii == 1 || ii == n
            swtch3 = false
        end

        temp = z[ii] / (work[ii]*delta[ii])
        dw = dpsi + dphi + temp^2
        temp = z[ii]*temp
        w += temp
        erretm += T(8) * (phi - psi) + T(2)*rhoinv + T(3)*abs(temp)

        #test for convergence

        if abs(w) <= mach_eps*erretm
            return
        end

        if w <= 0
            sglb = max(sglb, tau)
        else
            sgub = min(sgub, tau)
        end

        #calculate the new step
        niter += 1
        # eta = 0
        if !swtch3
            dtipsq = work[ip1]*delta[ip1]
            dtisq = work[i]*delta[i]

            if orgati
                c = w - dtipsq*dw + delsq * (z[i] / dtisq)^2
            else
                c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)^2
            end

            a = (dtipsq + dtisq)*w - dtipsq*dtisq*dw
            b = dtipsq*dtisq*w

            if c == 0
                if a == 0
                    if orgati
                        a = z[i]^2 + dtipsq^2*(dpsi+dphi)
                    else
                        a = z[ip1]^2 + dtisq^2*(dpsi + dphi)
                    end
                end
                eta = b /a
            elseif a <= 0
                eta = (a - sqrt(abs(a^2 - T(4)*b*c)))/(T(2)*c)
            else
                eta = T(2)*b / (a + sqrt(abs(a^2 - T(4)*b*c)))
            end
        else
            #interpolation using three most relevant points

            dtiim = work[iim1]*delta[iim1]
            dtiip = work[iip1]*delta[iip1]

            temp = rhoinv + psi + phi

            if orgati
                temp1 = z[iim1] / dtiim
                temp1 *= temp1
                c = (temp - dtiip*(dpsi+dphi)) - (d[iim1] - d[iip1])*(d[iim1]+d[iip1])*temp1
                zz[1] = z[iim1]^2

                if dpsi < temp1
                    zz[3] = dtiip^2*dphi
                else
                    zz[3] = dtiip^2*((dpsi - temp1) + dphi)
                end
            else
                temp1 = z[iip1]/dtiip
                temp1 *= temp1
                c = (temp - dtiim*(dpsi + dphi)) - (d[iip1] - d[iim1])*(d[iip1] + d[iim1])*temp1
                if dphi < temp1
                    zz[1] = dtiim^2*dpsi
                else
                    zz[1] = dtiim^2*(dpsi+(dphi-temp1))
                end
                zz[3] = z[iip1]^2
            end

            zz[2] = z[ii]^2
            dd[1] = dtiim
            dd[2] = delta[ii]*work[ii]
            dd[3] = dtiip

            #need to implement this one
            tau_laed6 = T[eta]
            laed6!(niter, orgati, c, dd, zz, w, tau_laed6, info)
            eta = tau_laed6[1]

            if info[] != 0
                # if info is not equal to zero, the laed6 failed, switch back to 2 pole interpolation

                swtch3 = false
                info .= 0
                dtipsq = work[ip1]*delta[ip1]
                dtisq = work[i]*delta[i]

                if orgati
                    c = w - dtipsq*dw + delsq*(z[i]/dtisq)^2
                else
                    c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)^2
                end
                a = (dtipsq + dtisq)*w - dtipsq*dtisq*dw
                b = dtipsq*dtisq*w

                if c == 0
                    if a == 0
                        if orgati
                            a = z[i]^2+ dtipsq^2*(dpsi+dphi)
                        else
                            a = z[ip1]^2+ dtisq^2*(dpsi+dphi)
                        end
                    end
                    eta = b/a
                elseif a <= 0
                    eta = (a - sqrt(abs(a^2  - T(4)*b*c)))/(T(2)*c)
                else
                    eta = T(2)*b / (a + sqrt(abs(a^2 - T(4)*b*c)))
                end

            end
        end

        #Left off at line 732 on https://www.netlib.org/lapack/explore-html/de/dc2/group__lasd4_ga2a4ec5313a1d81a260ff49665b81e887.html
        #=
        Note, eta should be positive if w is negative, and eta
            should be negative otherwise. However, if for
            some reason caused by roundoff, eta*w > 0,
            we simply use one Newton step isntead. This way
            will guarantee eta*w < 0
        =#

        if w*eta >= 0
            eta = -w /dw
        end

        if sigma[]^2 + eta < 0
            info .= 1
            sigma .= NaN
            delta .= NaN
            work .= NaN
            return
        end
        eta /= (sigma[] + sqrt(sigma[]^2 + eta))
        temp = tau + eta

        if temp > sgub || temp < sglb
            if w < 0
                eta = (sgub - tau)/T(2)
            else
                eta = (sglb - tau)/T(2)
            end

            if geomavg
                if w < 0
                    if tau > 0
                        eta = sqrt(sgub*tau) - tau
                    end
                else
                    if sglb > 0
                        eta = sqrt(sglb*tau) - tau
                    end
                end
            end

        end

        prew = w
        tau +=  eta
        sigma .+= eta

        for j in 1:n
            work[j] += eta
            delta[j] -= eta
        end

        #evaluate psi and the derivative dpsi

        dpsi = zero(T)
        psi = zero(T)
        erretm = zero(T)

        for j in 1:iim1
            temp = z[j] / (work[j]*delta[j])
            psi += z[j]*temp
            dpsi += temp^2
            erretm += psi
        end

        erretm = abs(erretm)

        #Evaluate phi and the derivative dphi

        dphi = zero(T)
        phi = zero(T)

        for j in n:-1:iip1
            temp = z[j] / (work[j]*delta[j])
            phi += z[j]*temp
            dphi += temp^2
            erretm += phi
        end

        tau2 = work[ii]*delta[ii]
        temp = z[ii]/tau2
        dw = dpsi + dphi + temp^2
        temp *= z[ii]
        w = rhoinv + phi + psi + temp
        erretm += T(8)*(phi - psi) + T(2)*rhoinv + T(3)*abs(temp)
        swtch = false

        if orgati 
            if -w > abs(prew)/T(10)
                swtch = true
            end
        else
            if w > abs(prew)/T(10)
                swtch = true
            end
        end

        #Main loop to update the values of the array Delta and work

        iter = niter + 1

        for niter in iter:maxit
            if abs(w) <= mach_eps*erretm
                return
            end

            if w <= 0
                sglb = max(sglb, tau)
            else
                sgub = min(sgub, tau)
            end

            #calculate the new step

            if !swtch3
                dtipsq = work[ip1]*delta[ip1]
                dtisq = work[i]*delta[i]

                if !swtch
                    if orgati
                        c = w - dtipsq*dw + delsq*(z[i]/dtisq)^2
                    else
                        c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)^2
                    end

                else
                    temp = z[ii]/(work[ii]*delta[ii])
                     
                    if orgati
                        dpsi += temp^2

                    else
                        dphi += temp^2
                    end

                    c = w - dtisq*dpsi - dtipsq*dphi
                end

                a = (dtipsq + dtisq)*w - dtipsq*dtisq*dw
                b = dtipsq*dtisq*w

                if c == 0
                    if a == 0
                        if !swtch
                            if orgati
                                a = z[i]^2 + dtipsq^2*(dpsi+dphi)
                            else
                                a = z[ip1]^2 + dtisq^2*(dpsi+dphi)
                            end
                        else
                            a = dtisq^2*dpsi + dtipsq^2*dphi
                        end
                    end
                    eta = b / a
                elseif a <= 0
                    eta = (a - sqrt(abs(a^2- T(4)*b*c))) / (T(2)*c)
                else
                    eta = T(2)*b/ (a + sqrt(abs(a^2 - T(4)*b*c)))
                end
            else
                #Interpolation using three most relevant points

                dtiim = work[iim1]*delta[iim1]
                dtiip = work[iip1]*delta[iip1]
                temp = rhoinv + psi + phi

                if swtch
                    c = temp - dtiim*dpsi - dtiip*dphi
                    zz[1] = dtiim^2*dpsi
                    zz[3] = dtiip^2*dphi
                else
                    if orgati
                        temp1 = z[iim1]/dtiim
                        temp1  *= temp1
                        temp2 = (d[iim1]-d[iip1])*(d[iim1]+d[iip1])*temp1
                        c = temp - dtiip*(dpsi+dphi) - temp2
                        zz[1] = z[iim1]^2

                        if dpsi < temp1
                            zz[3] = dtiip^2*dphi
                        else
                            zz[3] = dtiip^2*((dpsi-temp1)+dphi)
                        end

                    else
                        temp1 = z[iip1]/dtiip
                        temp1 *= temp1
                        temp2 = (d[iip1]-d[iim1])*(d[iim1]+d[iip1])*temp1
                        c = temp - dtiim*(dpsi+dphi) - temp2
                        if dphi < temp1
                            zz[1] = dtiim^2*dpsi
                        else
                            zz[1] = dtiim^2*(dpsi + (dphi - temp1))
                        end
                        zz[3] = z[iip1]^2
                    end
                end

                dd[1] = dtiim
                dd[2] = delta[ii]*work[ii]
                dd[3] = dtiip

                tau_laed6 = T[eta]
                laed6!(niter, orgati, c, dd, zz, w, tau_laed6, info)
                eta = tau_laed6[1]

                if info[] != 0
                    #=
                    If info is not 0, i.e, laed6 fialed, switch back to two pole
                        interpolation
                    =#

                    swtch3 = false
                    info .= 0
                    dtipsq = work[ip1]*delta[ip1]
                    dtisq = work[i]*delta[i]

                    if !swtch
                        if orgati
                            c = w - dtipsq*dw + delsq*(z[i]/dtisq)^2
                        else
                            c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)^2
                        end
                    else
                        temp = z[ii] / (work[ii]*delta[ii])

                        if orgati
                            dpsi += temp^2
                        else
                            dphi += temp^2
                        end
                        c = w - dtisq*dpsi - dtipsq*dphi
                    end
                    a = (dtipsq + dtisq)*w - dtipsq*dtisq*dw
                    b = dtipsq*dtisq*w

                    if c == 0
                        if a == 0
                            if !swtch   
                                if orgati
                                    a = z[i]^2+dtipsq^2*(dpsi+dphi)
                                else    
                                    a = z[ip1]^2+dtisq^2*(dpsi+dphi)
                                end
                            else
                                a = dtisq^2*dpsi + dtipsq^2*dphi
                            end

                        end
                        eta = b / a
                    elseif a <= 0
                        eta = (a - sqrt(abs(a^2-T(4)*b*c)))/(T(2)*c)
                    else
                        eta = T(2)*b / (a + sqrt(abs(a^2-T(4)*b*c)))
                    end
                end
            end

            #=
            Note, eta should be positive if w is negative, and eta
                should be negative otherwise. However, if for
                some reason caused by roundoff, eta*w > 0,
                we simply use one Newton step isntead. This way
                will guarantee eta*w < 0
            =#

            if w*eta >= 0
                eta = -w/dw
            end
            if sigma[]^2+eta < 0
                info .= 1
                sigma .= NaN
                delta .= NaN
                work .= NaN
                return
            end
            eta /= (sigma[]+sqrt(sigma[]^2+eta))
            temp = tau+eta

            if temp > sgub || temp < sglb
                if w < 0
                    eta = (sgub - tau) / T(2)
                else
                    eta = (sglb - tau)/T(2)
                end

                if geomavg
                    if w < 0
                        if tau > 0
                            eta = sqrt(sgub*tau)-tau
                        end
                    else
                        if sglb > 0
                            eta = sqrt(sglb*tau)-tau
                        end
                    end
                end
            end

            prew = w
            tau += eta
            sigma .+= eta

            for j in 1:n
                work[j] += eta
                delta[j] -= eta
            end

            dpsi = zero(T)
            psi = zero(T)
            erretm = zero(T)

            #Evaluate psi and the derivative dpsi
            for j in 1:iim1
                temp = z[j] / (delta[j]*work[j])
                psi += z[j]*temp
                dpsi +=  temp^2
                erretm += psi
            end

            erretm = abs(erretm)

            #Evaluate phi and the derivative dphi

            dphi = zero(T)
            phi = zero(T)

            for j in n:-1:iip1
                temp = z[j] / (work[j]*delta[j])
                phi += z[j]*temp
                dphi += temp^2
                erretm += phi
            end

            tau2 = work[ii]*delta[ii]
            temp = z[ii]/tau2
            dw = dpsi + dphi + temp^2
            temp *= z[ii]
            w = rhoinv + phi + psi + temp
            erretm = T(8)*(phi - psi) + erretm + T(2)*rhoinv + T(3)*abs(temp)
            
            if w*prew > 0 && abs(w) > abs(prew)/T(10)
                swtch = !swtch
            end
        end
        info .= 1

    end
end
