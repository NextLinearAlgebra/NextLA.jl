using KernelAbstractions.Extras: @unroll

const FACTORMUL = ((TILESIZEMUL>TILESIZE) ? 1 : Int(TILESIZE/TILESIZEMUL))



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
        if (TILESIZEMUL<=TILESIZE || i<=TILESIZE)
            @unroll for j in 0:FACTORMUL-1
                tausmem[j*TILESIZEMUL+i]=tau[j*TILESIZEMUL+i]
            end 
        end
        for k in 1:TILESIZE
            tmp_sum= zero(eltype(A))
            if (TILESIZEMUL<=TILESIZE || i<=TILESIZE)
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
        if (TILESIZEMUL<=TILESIZE || i<=TILESIZE)
            @unroll for j in 0:FACTORMUL-1
                tausmem[j*TILESIZEMUL+i,currtile+1]=tau[j*TILESIZEMUL+i,currtile]

            end 
        end
        for k in 1:TILESIZE
            tmp_sum= zero(eltype(A))
            if (TILESIZEMUL<=TILESIZE || i<=TILESIZE)
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

