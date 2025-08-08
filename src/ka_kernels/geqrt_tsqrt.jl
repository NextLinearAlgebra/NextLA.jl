using KernelAbstractions.Extras: @unroll

const FACTORQR = Int(TILESIZE/QRSPLIT)

@inline function calc_tau_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    newvalue = u1 + (u1<0 ? -1 : 1)  *sqrt(u1*u1+unorm)
    taucurrent = 2(newvalue*newvalue) / (unorm + newvalue*newvalue)
    tmp_sum2 = (uv +newvalue*v1)*2/ (unorm/newvalue + newvalue)
    return newvalue, taucurrent, tmp_sum2
end
@inline function correct_tau_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    newvalue=10*eps(T)
    taucurrent=2
    tmp_sum2=(uv/newvalue +v1)*2
    return newvalue, taucurrent, tmp_sum2
end

@kernel cpu=false  inbounds=true unsafe_indices=false function QR_unsafe_kernel_2d!(input, tau) 
    i = @index(Local,Linear)

    tilecol = @private eltype(input) (TILESIZE)
    cache = @localmem eltype(input) (TILESIZE)
    tau_iter = @private eltype(input) (1)
    sharedvalue = @localmem eltype(input) (1)
    
    @unroll for j=1:TILESIZE
        tilecol[j] = input[j,i]
    end

    for iter in 1:TILESIZE-1
        tmp_sum= zero(eltype(input))
        if (i==iter)
            @unroll for j in iter+1:TILESIZE
                cache[j] = tilecol[j]
                tmp_sum+=tilecol[j]*tilecol[j]
            end
            cache[iter]=tilecol[iter]
            sharedvalue[1]=tmp_sum
        end
        @synchronize
        if (i>=iter)       
            if (i>iter)
                @unroll for j in iter+1:TILESIZE
                    tmp_sum+=cache[j]*tilecol[j]
                end
            end
            newvalue, taucurrent, tmp_sum2 = calc_tau_factor(cache[iter], sharedvalue[1], tmp_sum , tilecol[iter])
            if (abs(newvalue)<10floatmin(eltype(input)))
                newvalue, taucurrent, tmp_sum2 = correct_tau_factor(cache[iter], sharedvalue[1], tmp_sum , tilecol[iter])
            end

            
            if (i==iter)
                tau_iter[1]=taucurrent
                @unroll for j in iter+1:TILESIZE
                    tilecol[j]/=newvalue
                end
            else
                @unroll for j in iter+1:TILESIZE
                    tilecol[j]-=tmp_sum2*(cache[j]/newvalue)
                end
            end
            
            
            tilecol[iter]-=tmp_sum2

        end
        input[iter,i] = tilecol[iter]
        @synchronize
    end
    input[TILESIZE,i] = tilecol[TILESIZE]
    tau[i]=tau_iter[1]
    @synchronize
end


@kernel cpu=false  inbounds=true unsafe_indices=false function QR_unsafe_kernel2_2d!(input, input2, tau)
    k,i = @index(Local, NTuple)

    tilecol = @private eltype(input) (FACTORQR)
    cache = @localmem eltype(input) (TILESIZE)
    cache2 = @localmem eltype(input) (QRSPLIT,TILESIZE)
    tau_iter = zero(eltype(input))
    sharedvalue = @localmem eltype(input) (2)
    
    @unroll for j in 1:FACTORQR
        tilecol[j] = input2[j+(k-1)*FACTORQR, i]
    end

    for iter in 1:TILESIZE
        
        tileiter= input[iter, i]
        if (i==iter)
            @unroll for j in 1:FACTORQR
                cache[j+(k-1)*FACTORQR] = tilecol[j]
            end
        end
        @synchronize
        
        if (i>=iter)
            tmp_sum = zero(eltype(input))
            @unroll for j in 1:FACTORQR
                tmp_sum+=tilecol[j]*cache[j+(k-1)*FACTORQR]
            end
            cache2[k,i]=tmp_sum
            if (i==iter && k==1)
                sharedvalue[2]=tileiter
            end
            
        end
        
        @synchronize

        if (i>=iter)
            tmpsumiter = zero(eltype(input))
            tmp_sum = zero(eltype(input))
            @unroll for j = 1:QRSPLIT
                tmpsumiter+= cache2[j,iter]
                tmp_sum += cache2[j,i]
            end
            newvalue, taucurrent, tmp_sum2 = calc_tau_factor(sharedvalue[2], tmpsumiter, tmp_sum , tileiter)
            if (abs(newvalue)<10floatmin(eltype(input)))
                newvalue, taucurrent, tmp_sum2 = correct_tau_factor(sharedvalue[2], tmpsumiter, tmp_sum , tileiter)
            end

            tau_iter = i==iter ? taucurrent : tau_iter

            if (i>iter)
                @unroll for j in 1:FACTORQR
                    tilecol[j]-=tmp_sum2*(cache[j+(k-1)*FACTORQR]/newvalue)
                end
            else
                @unroll for j in 1:FACTORQR
                    tilecol[j]/=newvalue
                end
            end
            
            
            
            if (k==1)
                input[iter, i]=tileiter-tmp_sum2
            end
            
        end
        @synchronize
    end

    @unroll for j in 1:FACTORQR
        input2[j+(k-1)*FACTORQR, i]=tilecol[j] 
    end
    if (k==1)
        tau[i]=tau_iter
    end
    

end

if (TILESIZE<=64) 

    @kernel cpu=false  inbounds=true unsafe_indices=false function QR_unsafe_kernel2_fused!(input, input2, tau, nbtiles)
        k,i = @index(Local, NTuple)

        tilecol = @private eltype(input) (FACTORQR)
        cache = @localmem eltype(input) (TILESIZE)
        cache2 = @localmem eltype(input) (QRSPLIT,TILESIZE)
        tau_iter = @localmem eltype(input) (TILESIZE)
        tilecol_first = @localmem eltype(input) (TILESIZE,TILESIZE)
        
        @unroll for j in 1:FACTORQR
            tilecol_first[i,j+(k-1)*FACTORQR] = input[j+(k-1)*FACTORQR, i]
        end
        currstartrow=0

        for currtile in 1:nbtiles

            @unroll for j in 1:FACTORQR
                tilecol[j] = input2[currstartrow+j+(k-1)*FACTORQR, i]
            end

            @unroll for iter in 1:TILESIZE
                
                if (i==iter)
                    @unroll for j in 1:FACTORQR
                        cache[j+(k-1)*FACTORQR] = tilecol[j]
                    end
                end
                @synchronize
                
                if (i>=iter)
                    tmp_sum = zero(eltype(input))
                    @unroll for j in 1:FACTORQR
                        tmp_sum+=tilecol[j]*cache[j+(k-1)*FACTORQR]
                    end
                    cache2[k,i]=tmp_sum
                    
                end
                
                @synchronize

                if (i>=iter)
                    tmpsumiter = zero(eltype(input))
                    tmp_sum = zero(eltype(input))
                    @unroll for j = 1:QRSPLIT
                        tmpsumiter+= cache2[j,iter]
                        tmp_sum += cache2[j,i]
                    end

                    newvalue, taucurrent, tmp_sum2 = calc_tau_factor(tilecol_first[iter,iter], tmpsumiter, tmp_sum , tilecol_first[i,iter])
                    if (abs(newvalue)<10floatmin(eltype(input)))
                        newvalue, taucurrent, tmp_sum2 = correct_tau_factor(tilecol_first[iter,iter], tmpsumiter, tmp_sum , tilecol_first[i,iter])
                    end

                    if (k==1)
                        tau_iter[iter]=taucurrent
                    end
                    

                    if (i>iter)
                        @unroll for j in 1:FACTORQR
                            tilecol[j]-=tmp_sum2*(cache[j+(k-1)*FACTORQR]/newvalue)
                        end
                    else
                    
                        @unroll for j in 1:FACTORQR
                            tilecol[j]/=newvalue
                        end
                    end
                end
                @synchronize
                if (i>=iter)
                    if (k==1)
                        tilecol_first[i,iter]-=tmp_sum2
                    end
                    
                end
                @synchronize
            end
            

            @unroll for j in 1:FACTORQR
                input2[currstartrow+j+(k-1)*FACTORQR, i]=tilecol[j] 
            end
            if (k+(i-1)*QRSPLIT<=TILESIZE)
                tau[k+(i-1)*QRSPLIT,currtile]=tau_iter[k+(i-1)*QRSPLIT]
            end
            currstartrow+=TILESIZE
        end

        @unroll for j in 1:FACTORQR
            input[j+(k-1)*FACTORQR, i]=tilecol_first[i,j+(k-1)*FACTORQR]
        end

    end
else



    @kernel cpu=false  inbounds=true unsafe_indices=false function QR_unsafe_kernel2_fused!(input, input2, tau, nbtiles)
        k,i = @index(Local, NTuple)

        tilecol = @private eltype(input) (FACTORQR)
        cache = @localmem eltype(input) (TILESIZE)
        cache2 = @localmem eltype(input) (QRSPLIT,TILESIZE)
        tilecol_first_cache = @localmem eltype(input) (TILESIZE)
        tau_iter = @localmem eltype(input) (TILESIZE)
        tilecol_first = @private eltype(input) (FACTORQR)
        
        @unroll for j in 1:FACTORQR
            tilecol_first[j] = input[j+(k-1)*FACTORQR, i]
        end
        currstartrow=0

        for currtile in 1:nbtiles

            @unroll for j in 1:FACTORQR
                tilecol[j] = input2[currstartrow+j+(k-1)*FACTORQR, i]
            end

            @unroll for iter in 1:TILESIZE
                
                if (i==iter)
                    @unroll for j in 1:FACTORQR
                        cache[j+(k-1)*FACTORQR] = tilecol[j]
                    end
                end
                if (k==cld(iter,FACTORQR) && i>=iter)
                    tilecol_first_cache[i] = tilecol_first[((iter-1)%FACTORQR)+1]
                end
                @synchronize
                
                if (i>=iter)
                    tmp_sum = zero(eltype(input))
                    @unroll for j in 1:FACTORQR
                        tmp_sum+=tilecol[j]*cache[j+(k-1)*FACTORQR]
                    end
                    cache2[k,i]=tmp_sum
                    
                end
                
                @synchronize

                if (i>=iter)
                    tmpsumiter = zero(eltype(input))
                    tmp_sum = zero(eltype(input))
                    @unroll for j = 1:QRSPLIT
                        tmpsumiter+= cache2[j,iter]
                        tmp_sum += cache2[j,i]
                    end

                    newvalue, taucurrent, tmp_sum2 = calc_tau_factor(tilecol_first_cache[iter], tmpsumiter, tmp_sum , tilecol_first_cache[i])
                    if (abs(newvalue)<10floatmin(eltype(input)))
                        newvalue, taucurrent, tmp_sum2 = correct_tau_factor(tilecol_first_cache[iter], tmpsumiter, tmp_sum , tilecol_first_cache[i])
                    end

                    if (k==1)
                        tau_iter[iter]=taucurrent
                    end
                    

                    if (i>iter)
                        @unroll for j in 1:FACTORQR
                            tilecol[j]-=tmp_sum2*(cache[j+(k-1)*FACTORQR]/newvalue)
                        end
                    else
                    
                        @unroll for j in 1:FACTORQR
                            tilecol[j]/=newvalue
                        end
                    end
                end
                @synchronize
                if (i>=iter)
                    if (k==1)
                        tilecol_first_cache[i]-=tmp_sum2
                    end
                    
                end
                @synchronize
                if (k==cld(iter,FACTORQR) && i>=iter)
                    tilecol_first[((iter-1)%FACTORQR)+1] = tilecol_first_cache[i] 
                end

                @synchronize
            end
            

            @unroll for j in 1:FACTORQR
                input2[currstartrow+j+(k-1)*FACTORQR, i]=tilecol[j] 
            end
            if (k+(i-1)*QRSPLIT<=TILESIZE)
                tau[k+(i-1)*QRSPLIT,currtile]=tau_iter[k+(i-1)*QRSPLIT]
            end
            currstartrow+=TILESIZE
        end

        @unroll for j in 1:FACTORQR
            input[j+(k-1)*FACTORQR, i]=tilecol_first[j]
        end

    end
end
