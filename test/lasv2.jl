using LinearAlgebra
using LinearAlgebra: BlasInt, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc
using NextLA
using Test



@testset "slasv2! test random input" begin
    # for T in [Float16, Float32, Float64]
    for T in [Float32, Float64]
        for i in 1:50

            starting = T(-1e3)
            ending = T(1e3)
            A = starting .+ (ending - starting).*rand(T, 2,2)
            # A = T[1 2; 0 5]
            A[2,1] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            else
                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]

            end

            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
            # if ssmax[] < 0
            #     println("ssmax less than zero: $(ssmax[])")
            # end
            # if ssmin[] < 0
            #     println("ssmin less than zero: $(ssmin[])")
            # end
        end
        
    end
end

@testset "slasv2! test zeroing out differnt parts" begin
    # for T in [Float16, Float32, Float64]
    for T in [Float32, Float64]
        starting = -min(T(1e5), floatmax(T))
        ending = min(T(1e5), floatmax(T))
        for i in 1:50

            A = starting .+ (ending - starting).*rand(T, 2,2)
            # A = T[1 2; 0 5]
            A[2,1] = 0
            A[1,1] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            else

                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]
            end

            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
        end
        for i in 1:50
            A = starting .+ (ending - starting).*rand(T, 2,2)
            # A = T[1 2; 0 5]
            A[2,1] = 0
            A[1,2] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )

            else
                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]

            end

            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
        end
        for i in 1:50
            A = starting .+ (ending - starting).*rand(T, 2,2)
            # A = T[1 2; 0 5]
            A[2,1] = 0
            A[2,2] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )

            else
                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]

            end
            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
        end
        for i in 1:50
            A = starting .+ (ending - starting).*rand(T, 2,2)
            A[2,1] = 0
            A[1,1] = 0
            A[2,2] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )

            else
                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]

            end

            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
        end
        for i in 1:50
            A = starting .+ (ending - starting).*rand(T, 2,2)
            # A = T[1 2; 0 5]
            A[2,1] = 0
            A[1,2] = 0
            A[2,2] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )

            else
                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]

            end

            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
        end
        for i in 1:50
            A = starting .+ (ending - starting).*rand(T, 2,2)
            # A = T[1 2; 0 5]
            A[2,1] = 0
            A[1,2] = 0
            A[1,1] = 0
            A_copy = deepcopy(A)
            out_vec = zeros(T, 2)
            f = Ref{T}(A[1,1])
            g = Ref{T}(A[1,2])
            h = Ref{T}(A[2,2])
            ssmin = Ref{T}(0)
            ssmax = Ref{T}(0)
            snr = Ref{T}(0)
            csr = Ref{T}(0)
            snl = Ref{T}(0)
            csl = Ref{T}(0)
            NextLA.lasv2!(A, out_vec)

            if T == Float32
                ccall(
                    (@blasfunc(slasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},
                        Ref{Float32}, Ref{Float32}, Ref{Float32},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )
            elseif T == Float64
                ccall(
                    (@blasfunc(dlasv2_), libblastrampoline),
                        Cvoid, 
                        (Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},
                        Ref{Float64}, Ref{Float64}, Ref{Float64},),
                        f, g, h, ssmin, ssmax, snr, csr, snl, csl
                    )

            else
                U, S, V = svd(A_copy)
                ssmax[] = S[1]
                ssmin[] = S[2]
                csl[] = U[1,1]
                snl[] = (U[2,1])
                csr[] = V[1,1]
                snr[] = V[2,1]

            end

            @test isapprox(ssmax[], out_vec[1])
            @test isapprox(ssmin[], out_vec[2])
            @test isapprox(snr[], A[1,2])
            @test isapprox(csr[], A[2,2])
            @test isapprox(snl[], A[2,1])
            @test isapprox(csl[], A[1,1])
        end
        A = zeros(T, 2, 2)
        A_copy = deepcopy(A)
        out_vec = zeros(T, 2)
        f = Ref{T}(A[1,1])
        g = Ref{T}(A[1,2])
        h = Ref{T}(A[2,2])
        ssmin = Ref{T}(0)
        ssmax = Ref{T}(0)
        snr = Ref{T}(0)
        csr = Ref{T}(0)
        snl = Ref{T}(0)
        csl = Ref{T}(0)
        NextLA.lasv2!(A, out_vec)

        if T == Float32
            ccall(
                (@blasfunc(slasv2_), libblastrampoline),
                    Cvoid, 
                    (Ref{Float32}, Ref{Float32}, Ref{Float32},
                    Ref{Float32}, Ref{Float32}, Ref{Float32},
                    Ref{Float32}, Ref{Float32}, Ref{Float32},),
                    f, g, h, ssmin, ssmax, snr, csr, snl, csl
                )
        elseif T == Float64
            ccall(
                (@blasfunc(dlasv2_), libblastrampoline),
                    Cvoid, 
                    (Ref{Float64}, Ref{Float64}, Ref{Float64},
                    Ref{Float64}, Ref{Float64}, Ref{Float64},
                    Ref{Float64}, Ref{Float64}, Ref{Float64},),
                    f, g, h, ssmin, ssmax, snr, csr, snl, csl
                )

        else
            U, S, V = svd(A_copy)
            ssmax[] = S[1]
            ssmin[] = S[2]
            csl[] = U[1,1]
            snl[] = (U[2,1])
            csr[] = V[1,1]
            snr[] = V[2,1]

        end

        @test isapprox(ssmax[], out_vec[1])
        @test isapprox(ssmin[], out_vec[2])
        @test isapprox(snr[], A[1,2])
        @test isapprox(csr[], A[2,2])
        @test isapprox(snl[], A[2,1])
        @test isapprox(csl[], A[1,1])
    end
end
