using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test


@testset "slasd7! test random input of even sizes" begin
    for T in [Float32, Float64]
        starting = -T(1e3)
        ending = T(1e3)
        for j in 8:2:100
            for i in 1:1
                block_size = j ÷ 2
                
                nl = block_size - 1
                nr = block_size
                sqre = 0
                k = [0]
                
                n = nl + nr + 1
                m = n + sqre
                
                A = Bidiagonal((starting .+ (ending - starting) .* rand(T, j, j)), :U)
                B1 = A[1:block_size-1, 1:block_size-1]
                B2 = A[block_size+1:end, block_size+1:end]
                U1, D1, V1 = svd(B1)
                U2, D2, V2 = svd(B2)

                D = [D1; 0 ; D2]
                # D = T[833.16327, 106.60155, 25.374142, 0.0, 1098.0494, 1018.23157, 748.32874, 2.1672876]

                icompq = 1

                z = zeros(T, n)
                zw = zeros(T, m)
                vf = zeros(T, m)
                vf[1:nl] .= V1[1,:]
                vf[nl+1] = 1.5
                vf[nl+2:m] .= V2[1,:]
                # vf = T[-0.019315826, 0.61472344, 0.78850627, 1.5, 0.7306953, -0.0070360135, 0.6615086, -0.16864555]
                vfw = zeros(T,m)
                vl = zeros(T, m)
                vl[1:nl] .= V1[end,:]
                vl[nl+1] = 0.5
                vl[nl+2:m] .= V2[end,:]
                # vl = T[0.3635344, 0.7389916, -0.5672162, 0.5, 0.006113885, 0.9999556, 0.0051511647, 0.004976479]
                vlw = zeros(T, m)
                
                alpha = rand(T)
                beta = rand(T)
                # alpha = T(0.014339328)
                # beta = T(0.5906076)

                # alpha = T(0.3)
                # beta = T(0.5)
                # alpha_native = Ref{T}(T(0.3))
                beta_native = Ref{T}(T(0.5))
                alpha_native = Ref{T}(alpha)
                beta_native = Ref{T}(beta)
                dsigma = zeros(T, n)
                
                idx = zeros(Int64, n)
                idxp = zeros(Int64, n)
                idxq = zeros(Int64, n)
                idxq[1:nl] = reverse(Vector(1:nl))
                idxq[nl+2:end] = reverse(Vector(1:nr))
                # idxq = [3, 2, 1, 0, 4, 3, 2, 1]
                perm = zeros(Int64, n)
                givptr = [0]
                ldgcol = n
                ldgnum = n
                givnum = zeros(T, ldgnum, 2)
                givcol = zeros(Int64, ldgcol, 2)
                c = [T(0)]
                s = [T(0)]
                info = [0]

            
                k_native = Ref{BlasInt}(T(0))
                D_native  = deepcopy(D)
                z_native = deepcopy(z)
                zw_native = deepcopy(zw)
                vf_native = deepcopy(vf)
                vfw_native = deepcopy(vfw)
                vl_native = deepcopy(vl)
                vlw_native = deepcopy(vlw)
                dsigma_native = deepcopy(dsigma)
                idx_native = deepcopy(idx)
                idxp_native = deepcopy(idxp)
                idxq_native = deepcopy(idxq)
                perm_native = deepcopy(perm)
                givptr_native = Ref{BlasInt}(0)
                givcol_native = deepcopy(givcol)
                givnum_native = deepcopy(givnum)
                c_native = Ref{T}(T(0))
                s_native = Ref{T}(T(0))
                info_native = Ref{BlasInt}(T(0))

                # println("D_native=$D_native")
                # println("vf_native=$vf_native")
                # println("vl_native=$vl_native")
                # println("idxq_native=$idxq_native")
                # println("alpha=$alpha")
                # println("beta=$beta")

                # println("")
                # slm = ccall(
                #         (@blasfunc(slamch_), libblastrampoline),
                #         Float32,
                #         (Ref{UInt8},),
                #         UInt8('E')
                #     )
                # dlm = ccall(
                #         (@blasfunc(dlamch_), libblastrampoline),
                #         Float64,
                #         (Ref{UInt8},),
                #         UInt8('E')
                #     )
                # println("slm: $slm")
                # println("dlm: $dlm")
                # @test (slm) == (5*eps(Float32))
                # @test (2*dlm) == eps(Float64)



                NextLA.lasd7!(icompq, nl, nr, sqre, k, D,
                z, zw, vf, vfw, vl, vlw, alpha, beta, dsigma,
                idx, idxp, idxq, perm, givptr, givcol, ldgcol, givnum, ldgnum,
                c, s, info)


                # println("FInished running my function")
                if T == Float64
                    ccall(
                        (@blasfunc(dlasd7_), libblastrampoline),
                            Cvoid, 
                            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                            Ref{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                            Ptr{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt},
                            Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                            Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ref{Float64}, Ref{BlasInt}),
                            icompq, nl, nr, sqre, k_native, D_native, z_native, zw_native,
                            vf_native, vfw_native, vl_native, vlw_native, alpha_native, beta_native, dsigma_native,
                            idx_native, idxp_native, idxq_native, perm_native, givptr_native, givcol_native,
                            ldgcol, givnum_native, ldgnum, c_native, s_native, info_native
                        )
                else
                    ccall(
                        (@blasfunc(slasd7_), libblastrampoline),
                            Cvoid, 
                            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                            Ref{BlasInt}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                            Ptr{Float32}, Ptr{Float32}, Ref{Float32}, Ref{Float32}, Ptr{Float32}, Ptr{BlasInt}, Ptr{BlasInt},
                            Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                            Ptr{Float32}, Ref{BlasInt}, Ref{Float32}, Ref{Float32}, Ref{BlasInt}),
                            icompq, nl, nr, sqre, k_native, D_native, z_native, zw_native,
                            vf_native, vfw_native, vl_native, vlw_native, alpha_native, beta_native, dsigma_native,
                            idx_native, idxp_native, idxq_native, perm_native, givptr_native, givcol_native,
                            ldgcol, givnum_native, ldgnum, c_native, s_native, info_native
                        )

                end
                # println("FInished running functions")
                @test (k_native[] == k[1])
                @test isapprox(D_native, D)
                @test isapprox(z_native, z)
                @test isapprox(zw_native, zw)
                @test isapprox(vf_native, vf)
                @test isapprox(vfw_native, vfw)
                @test isapprox(vl_native, vl)
                @test isapprox(vlw_native, vlw)
                @test isapprox(dsigma_native, dsigma)
                @test (idx_native == idx)
                @test (idxq_native == idxq)
                @test (idxp_native == idxp)
                @test (perm_native == perm)
                @test (givptr_native[] == givptr[1])
                @test (givcol_native == givcol)
                @test isapprox(givnum_native, givnum)
                @test isapprox(c_native[], c[1])
                @test isapprox(s_native[], s[1])
                @test isapprox(info_native[], info[1])



            end
        end
    end
end
