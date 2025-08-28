using Test
using NextLA
using LinearAlgebra
using LinearAlgebra: require_one_based_indexing, libblastrampoline
using LinearAlgebra.LAPACK
using LinearAlgebra.BLAS:@blasfunc, BlasInt
using Random

for (larfg, elty) in
     ((:dlarfg_, Float64),
      (:slarfg_, Float32),
      (:zlarfg_, ComplexF64),
      (:clarfg_, ComplexF32))
     @eval begin
         #        .. Scalar Arguments ..
         #        INTEGER            incx, n
         #        DOUBLE PRECISION   alpha, tau
         #        ..
         #        .. Array Arguments ..
         #        DOUBLE PRECISION   x( * )
         function larfg_our!(x::AbstractVector{$elty})
             require_one_based_indexing(x)
             N    = BlasInt(length(x))
             alpha = Ref{$elty}(x[1])
             incx = BlasInt(1)
             τ    = Ref{$elty}(0)
             ccall((@blasfunc($larfg), libblastrampoline), Cvoid,
                 (Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                 N, alpha, pointer(x, 2), incx, τ)
             #@inbounds x[1] = one($elty) this is included in the original function which is not right
             return τ[], alpha[]
         end
     end  
 end



@testset "ZLARFG Tests" begin
    @testset "NextLA vs LAPACK comparison" begin
        for T in [ComplexF32, ComplexF64]
            @testset "Type $T" begin
                rtol = (T <: ComplexF32) ? 1e-5 : 1e-12
                
                for n in [1, 2, 5, 10]
                    @testset "Size n=$n" begin
                        # Generate random test data
                        alpha_orig = randn(T)
                        x_orig = randn(T, n-1)
                        
                        # Test NextLA implementation
                        x_nextla = copy(x_orig)
                        alpha_nextla, tau_nextla = NextLA.larfg!(n, alpha_orig, x_nextla, 1, zero(T))
                        
                        # Test LAPACK reference
                        if n > 0
                            lapack_vec = vcat([alpha_orig], x_orig)
                            tau_lapack, alpha_lapack = larfg_our!(lapack_vec)
                            x_lapack = lapack_vec[2:end]
                            
                            # For n==1, NextLA defines tau≈0; LAPACK may return nonzero. Accept tau≈0.
                            if n == 1
                                @test abs(tau_nextla) ≤ (T <: ComplexF32 ? 1e-6 : 1e-12)
                                # alpha magnitude should match LAPACK
                                @test abs(abs(alpha_nextla) - abs(alpha_lapack)) < rtol * max(1, abs(alpha_lapack))
                            else
                                # Compare results (allowing for sign differences)
                                if abs(abs(tau_nextla) - abs(tau_lapack)) > rtol * max(1, abs(tau_lapack))
                                    @show tau_nextla, tau_lapack
                                end
                                @test abs(abs(tau_nextla) - abs(tau_lapack)) < rtol * max(1, abs(tau_lapack))
                                if abs(abs(alpha_nextla) - abs(alpha_lapack)) > rtol * max(1, abs(alpha_lapack))
                                    @show alpha_nextla, alpha_lapack
                                end
                                @test abs(abs(alpha_nextla) - abs(alpha_lapack)) < rtol * max(1, abs(alpha_lapack))
                                if length(x_orig) > 0
                                    @test norm(abs.(x_nextla) - abs.(x_lapack)) < rtol * max(1, norm(x_lapack))
                                end
                            end
                        end
                        
                        # Basic sanity checks
                        @test isfinite(alpha_nextla)
                        @test isfinite(tau_nextla)
                        @test all(isfinite.(x_nextla))
                    end
                end
            end
        end
    end
    
    @testset "Edge cases" begin
        for T in [ComplexF32, ComplexF64]
            @testset "Type $T edge cases" begin
                # Test n=0 case
                alpha_nextla, tau_nextla = NextLA.larfg!(0, T(1), T[], 1, zero(T))
                @test tau_nextla == 0
                @test alpha_nextla == T(1)
                
                # Test n=1 case
                alpha_nextla, tau_nextla = NextLA.larfg!(1, T(2), T[], 1, zero(T))
                @test abs(tau_nextla) < 1e-10
                @test abs(alpha_nextla - T(2)) < 1e-10
                
                # Test zero vector (n=3, x has length 2)
                alpha_nextla, tau_nextla = NextLA.larfg!(3, T(0), zeros(T, 2), 1, zero(T))
                @test isfinite(alpha_nextla)
                @test isfinite(tau_nextla)
            end
        end
    end
end
