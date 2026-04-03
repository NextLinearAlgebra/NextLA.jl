using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test


@testset "slasd6! test random input of even sizes" begin
    for T in [Float32, Float64]
        starting = -T(1e3)
        ending = T(1e3)
        for j in 8:10:500
            for i in 1:10

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
                
                alpha = [rand(T)]
                beta = [rand(T)]
                # alpha = T(0.014339328)
                # beta = T(0.5906076)

                # alpha = T(0.3)
                # beta = T(0.5)
                # alpha_native = Ref{T}(T(0.3))
                beta_native = Ref{T}(T(0.5))
                alpha_native = Ref{T}(alpha[])
                beta_native = Ref{T}(beta[])
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
                poles = zeros(T, ldgnum, 2)
                difl = zeros(T, n)
                difr = zeros(T, ldgnum, 2)
                work = zeros(T, 4*m)
                iwork = zeros(Int64, 3*m)
                c = [T(0)]
                s = [T(0)]
                info = [0]
            
                k_native = Ref{BlasInt}(T(0))
                D_native  = deepcopy(D)
                z_native = deepcopy(z)
                vf_native = deepcopy(vf)
                vl_native = deepcopy(vl)
                idxq_native = deepcopy(idxq)
                perm_native = deepcopy(perm)
                givptr_native = Ref{BlasInt}(0)
                givcol_native = deepcopy(givcol)
                givnum_native = deepcopy(givnum)
                c_native = Ref{T}(T(0))
                s_native = Ref{T}(T(0))
                poles_native = deepcopy(poles)
                difl_native = deepcopy(difl)
                difr_native = deepcopy(difr)
                work_native = deepcopy(work)
                iwork_native = deepcopy(iwork)
                info_native = Ref{BlasInt}(T(0))

                NextLA.lasd6!(icompq, nl, nr, sqre, D, vf, 
                            vl, alpha, beta,
                            idxq, perm, givptr, givcol,
                            ldgcol, givnum, ldgnum, poles,
                            difl, difr, z, k, c, s, 
                            work, iwork, info)


                # println("FInished running my function")
                if T == Float64
                    ccall(
                        (@blasfunc(dlasd6_), libblastrampoline),
                            Cvoid, 
                            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
                            Ptr{Float64}, Ref{Float64}, Ref{Float64},
                            Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, 
                            Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, 
                            Ptr{Float64}, Ptr{Float64},  Ptr{Float64}, Ref{BlasInt},
                            Ref{Float64}, Ref{Float64},
                            Ptr{Float64}, Ptr{BlasInt}, Ref{BlasInt}),
                            icompq, nl, nr, sqre, D_native, vf_native, 
                            vl_native, alpha_native, beta_native,
                            idxq_native, perm_native, givptr_native, givcol_native,
                            ldgcol, givnum_native, ldgnum, poles_native,
                            difl_native, difr_native, z_native, k_native,
                            c_native, s_native, 
                            work_native, iwork_native, info_native
                        )
                else
                    ccall(
                        (@blasfunc(slasd6_), libblastrampoline),
                            Cvoid, 
                            (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ptr{Float32},
                            Ptr{Float32}, Ref{Float32}, Ref{Float32},
                            Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, 
                            Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{Float32}, 
                            Ptr{Float32}, Ptr{Float32},  Ptr{Float32}, Ref{BlasInt},
                            Ref{Float32}, Ref{Float32},
                            Ptr{Float32}, Ptr{BlasInt}, Ref{BlasInt}),
                            icompq, nl, nr, sqre, D_native, vf_native, 
                            vl_native, alpha_native, beta_native,
                            idxq_native, perm_native, givptr_native, givcol_native,
                            ldgcol, givnum_native, ldgnum, poles_native,
                            difl_native, difr_native, z_native, k_native,
                            c_native, s_native, 
                            work_native, iwork_native, info_native
                        )

                end
                # println("FInished running functions")
                @test isapprox(D_native, D)
                @test isapprox(vf_native, vf)
                @test isapprox(vl_native, vl)
                @test isapprox(alpha_native[], alpha[])
                @test isapprox(beta_native[], beta[])
                @test (idxq_native == idxq)
                @test (perm_native == perm)
                @test (givptr_native[] == givptr[1])
                @test (givcol_native == givcol)
                @test isapprox(givnum_native, givnum)
                @test isapprox(poles_native, poles)
                @test isapprox(difl_native, difl)
                @test isapprox(difr_native, difr)
                @test (k_native[] == k[1])
                @test isapprox(z_native[1:k[]], z[1:k[]])
                @test isapprox(c_native[], c[1])
                @test isapprox(s_native[], s[1])
                @test isapprox(work_native, work)
                @test isapprox(iwork_native, iwork)
                @test isapprox(info_native[], info[1])



            end
        end
    end
end
