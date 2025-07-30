using KernelAbstractions.Extras: @unroll

const FACTORQR = Int(TILESIZE/QRSPLIT)
const FACTORMUL = (TILESIZEMUL>TILESIZE) ? 1 : Int(TILESIZE/TILESIZEMUL)

@inline function calc_tau_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    newvalue = u1 + (u1<0 ? -1 : 1)  *sqrt(u1*u1+unorm)
    taucurrent = 2(newvalue*newvalue) / (unorm + newvalue*newvalue)
    tmp_sum2 = (uv +newvalue*v1)*2/ (unorm/newvalue + newvalue)
    return newvalue, taucurrent, tmp_sum2
end
@inline function correct_tau_factor(u1::T, unorm::T, uv::T, v1::T) where {T<:Number}
    newvalue=1
    taucurrent=2
    tmp_sum2=v1
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

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    M = @localmem eltype(A) (TILESIZE)

        @unroll for l in 1:TILESIZE
            tilecol[l] = A[l, (g-1)*TILESIZE+i]
        end

        for k in 1:TILESIZE-1
                M[i] = Min[i, k]
            @synchronize
            tmp_sum = zero(eltype(A))
            @unroll for l in k+1:TILESIZE
                tmp_sum += M[l] * tilecol[l]
            end
            tmp_sum+=tilecol[k]
            tmp_sum*=tau[k]

            @unroll for l in k+1:TILESIZE
                tilecol[l] -= tmp_sum * M[l]
            end
            tilecol[k]-=tmp_sum
            @synchronize
        end

        @unroll for l in 1:TILESIZE
            A[l, (g-1)*TILESIZE+i]=tilecol[l]
        end

end


@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (2TILESIZE)
    Mcurr= @localmem eltype(A) (TILESIZE)
    tausmem = @localmem eltype(A) (TILESIZE)


        @unroll for l in 1:TILESIZE
            tilecol[l] = B[l, i+(g-1)*TILESIZEMUL] 
        end
        @unroll for l in 1:TILESIZE
            tilecol[l+TILESIZE] = A[l, i+(g-1)*TILESIZEMUL] 
        end
        if (TILESIZEMUL>TILESIZE && i<=TILESIZE)
            @unroll for j in 0:FACTORMUL-1
                tausmem[j*TILESIZEMUL+i]=tau[j*TILESIZEMUL+i]
            end 
        end
        for k in 1:TILESIZE
            tmp_sum= zero(eltype(A))
            if (TILESIZEMUL>TILESIZE && i<=TILESIZE)
                @unroll for j in 0:FACTORMUL-1
                    Mcurr[j*TILESIZEMUL+i]=Min[j*TILESIZEMUL+i,k]
                end 
            end
            @synchronize      
            @unroll for l in 1:TILESIZE
                tmp_sum += Mcurr[l] * tilecol[l]
            end
            tmp_sum+= tilecol[k+TILESIZE] 
            tmp_sum *= tausmem[k]
            tilecol[k+TILESIZE] -= tmp_sum

            @unroll for l in 1:TILESIZE
                tilecol[l] -= tmp_sum * Mcurr[l]
            end
            @synchronize  
        end

        @unroll for l in 1:TILESIZE
            A[l, i+(g-1)*TILESIZEMUL] = tilecol[l+TILESIZE]
        end
        @unroll for l in 1:TILESIZE
            B[l, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
    
end

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_fused!(A, B, @Const(Min), @Const(tau), nbtiles)
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    tilecolA = @private eltype(A) (TILESIZE)
    Mcurr= @localmem eltype(A) (TILESIZE)
    tausmem = @localmem eltype(A) (TILESIZE)

    currstartrow=0

    #@print nbtiles " " i " " g "\n"

    @unroll for l in 1:TILESIZE
        tilecolA[l] = A[l, i+(g-1)*TILESIZEMUL] 
    end
    for currtile in 1:nbtiles
        @unroll for l in 1:TILESIZE
            tilecol[l] = B[currstartrow+l, i+(g-1)*TILESIZEMUL] 
        end
        if (TILESIZEMUL>TILESIZE && i<=TILESIZE)
            @unroll for j in 0:FACTORMUL-1
                tausmem[j*TILESIZEMUL+i,currtile+1]=tau[j*TILESIZEMUL+i,currtile]

            end 
        end
        for k in 1:TILESIZE
            tmp_sum= zero(eltype(A))
            if (TILESIZEMUL>TILESIZE && i<=TILESIZE)
                @unroll for j in 0:FACTORMUL-1
                    Mcurr[j*TILESIZEMUL+i]=Min[currstartrow+j*TILESIZEMUL+i,k]
                end 
            end
            @synchronize      
            @unroll for l in 1:TILESIZE
                tmp_sum += Mcurr[l] * tilecol[l]
            end
            tmp_sum+= tilecolA[k] 
            tmp_sum *= tausmem[k]
            tilecolA[k] -= tmp_sum

            @unroll for l in 1:TILESIZE
                tilecol[l] -= tmp_sum * Mcurr[l]
            end
            @synchronize  
        end
        @unroll for l in 1:TILESIZE
            B[currstartrow+l, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
        currstartrow+=TILESIZE
    end
    @unroll for l in 1:TILESIZE
        A[l, i+(g-1)*TILESIZEMUL] = tilecolA[l]
    end

        
    
end


#=
const MULSPLIT = 1
const SIZESPLIT = Int(TILESIZE/MULSPLIT)
@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i,j = @index(Local, NTuple)
    tilecol = @private eltype(A) (SIZESPLIT)
    Mcurr= @localmem eltype(A) (TILESIZE)
    cache = @localmem eltype(A) (TILESIZEMUL,MULSPLIT)

        for l in 1:SIZESPLIT
            tilecol[l] = B[l+(j-1)*SIZESPLIT, i+(g-1)*TILESIZEMUL] 
        end

        for k in 1:TILESIZE
            if (j==1)
                for l in 0:FACTORMUL-1
                    Mcurr[l*TILESIZEMUL+i]=Min[l*TILESIZEMUL+i,k]
                end 
            end
            @synchronize     
            tmp_sum= zero(eltype(A)) 
            for l in 1:SIZESPLIT
                tmp_sum += Mcurr[l+(j-1)*SIZESPLIT] * tilecol[l]
            end
            cache[i,j]=tmp_sum
            @synchronize
            tmp_sum= zero(eltype(A))
            for l in 1:MULSPLIT
                tmp_sum+=cache[i,l]
            end
            tmp_sum+= A[k, i+(g-1)*TILESIZEMUL]
            tmp_sum *= tau[k]
            for l in 1:SIZESPLIT
                tilecol[l] -= tmp_sum * Mcurr[l+(j-1)*SIZESPLIT]
            end
            if (j==1)
                A[k, i+(g-1)*TILESIZEMUL] -= tmp_sum
            end

            @synchronize  
        end
        for l in 1:SIZESPLIT
            B[l+(j-1)*SIZESPLIT, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
    
end

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i,j = @index(Local, NTuple)
    tilecol = @private eltype(A) (2TILESIZE)
    Mcurr= @localmem eltype(A) (TILESIZE,2)
    tausmem = @localmem eltype(A) (TILESIZE)

        if (j==1)
           for l in 1:TILESIZE
              tilecol[l] = B[l, i+(g-1)*TILESIZEMUL]
           end
           for l in 1:TILESIZE
              tilecol[l+TILESIZE] = A[l, i+(g-1)*TILESIZEMUL]
           end
        end
        if (j==2)
           for l in 1:FACTORMUL
              tausmem[l+FACTORMUL*(i-1)]=tau[l+FACTORMUL*(i-1)]
           end
        end
        curridx=1

        for k in 1:TILESIZE
            tmp_sum= zero(eltype(A))
            if (j==2)
               for l in 1:FACTORMUL
                  Mcurr[l+FACTORMUL*(i-1),curridx]=Min[l+FACTORMUL*(i-1),k]
               end
            end
            @synchronize
            if (j==1)
               for l in 1:TILESIZE
                  tmp_sum += Mcurr[l,curridx] * tilecol[l]
               end

               tmp_sum+= tilecol[k+TILESIZE]
               tmp_sum *= tausmem[k]
               tilecol[k+TILESIZE] -= tmp_sum

               for l in 1:TILESIZE
                  tilecol[l] -= tmp_sum * Mcurr[l,curridx]
               end
            end
            curridx=3-curridx
        end
        @synchronize
        if (j==1)
           for l in 1:TILESIZE
              A[l, i+(g-1)*TILESIZEMUL] = tilecol[l+TILESIZE]
           end
           for l in 1:TILESIZE
              B[l, i+(g-1)*TILESIZEMUL] = tilecol[l]
           end
        end

end

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (2TILESIZE)
    Mcurr= @localmem eltype(A) (TILESIZE,2)
    tausmem = @localmem eltype(A) (TILESIZE)
    cache = @private eltype(A) (4)

        for l in 1:TILESIZE
            tilecol[l] = B[l, i+(g-1)*TILESIZEMUL] 
        end
        for l in 1:TILESIZE
            tilecol[l+TILESIZE] = A[l, i+(g-1)*TILESIZEMUL] 
        end
        for j in 0:FACTORMUL-1
            tausmem[j*TILESIZEMUL+i]=tau[j*TILESIZEMUL+i]
        end 
        
        for k in 1:TILESIZE
            for l in 1:4
                cache[l]=zero(eltype(A))
            end
            for j in 0:FACTORMUL-1
                Mcurr[j*TILESIZEMUL+i]=Min[j*TILESIZEMUL+i,k]
            end 
            @synchronize      
            for l in 1:4:TILESIZE
                for j in 0:3
                    cache[l+j]+=Mcurr[l+j] * tilecol[l+j]
                end
            end
            tmp_sum= (cache[1]+cache[2]) + (cache[3]+cache[4])
            tmp_sum+= tilecol[k+TILESIZE] 
            tmp_sum *= tausmem[k]
            tilecol[k+TILESIZE] -= tmp_sum

            for l in 1:TILESIZE
                tilecol[l] -= tmp_sum * Mcurr[l]
            end
            @synchronize  
        end

        for l in 1:TILESIZE
            A[l, i+(g-1)*TILESIZEMUL] = tilecol[l+TILESIZE]
        end
        for l in 1:TILESIZE
            B[l, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
    
end

@kernel cpu=false  inbounds=true unsafe_indices=false  function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILESIZE)
    tilecol2 = @private eltype(A) (4)
    Mcurr= @localmem eltype(A) (TILESIZE)
    tau = @private eltype(A) (4)


        for l in 1:TILESIZE
            tilecol[l] = B[l, i+(g-1)*TILESIZEMUL] 
        end

        for klarge in 1:4:TILESIZE
            for l in 0:3
                tilecol2[l+1]=A[klarge+l, i+(g-1)*TILESIZEMUL] 
                tau[l+1]=tau[klarge+l]
            end
            for k in klarge:(klarge+3)
                tmp_sum= zero(eltype(A))
                for j in 0:FACTORMUL-1
                    Mcurr[j*TILESIZEMUL+i]=Min[j*TILESIZEMUL+i,k]
                end 

                @synchronize      
                for l in 1:TILESIZE
                    tmp_sum += Mcurr[l] * tilecol[l]
                end
                tmp_sum+= tilecol2[k-klarge+1] 
                tmp_sum *= tau[k-klarge+1]
                tilecol2[k-klarge+1] -= tmp_sum

                for l in 1:TILESIZE
                    tilecol[l] -= tmp_sum * Mcurr[l]
                end
                @synchronize  
            end
            for l in 0:3
                A[klarge+l, i+(g-1)*TILESIZEMUL] = tilecol2[l+1]
            end
        end

        for l in 1:TILESIZE
            B[l, i+(g-1)*TILESIZEMUL] = tilecol[l]
        end
    
end


=#