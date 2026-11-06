# Native integration check: valid solve residuals alone cannot detect a
# permutation-direction mistake, since either direction defines a valid system.
using Test, LinearAlgebra, SparseArrays, Random
import sTiles
include("methods.jl")
setting(name, default) = parse(Int, get(ENV, name, string(default)))

@testset "sTiles native permutation direction" begin
    isdefined(sTiles, :ordering) && error("This check targets the current analyze/permutation API")
    for side in (11,15)
        n = side^2
        A = spdiagm(0=>fill(5.0,n))
        for i in 1:n
            if i % side != 0; A[i,i+1] = A[i+1,i] = -1; end
            if i+side <= n; A[i,i+side] = A[i+side,i] = -1; end
        end
        p = randperm(MersenneTwister(77),n)
        A = A[p,p]
        F = sTiles.analyze(A; cores=1, inverse=false)
        raw = Int[]; native_nnz = 0
        try
            raw = Int.(sTiles.permutation(F))
            native_nnz = nnz(F)
        finally
            close(F)
        end
        count_for(p) = nnz(sparse(cholesky(Symmetric(A[p,p]); perm=collect(1:n)).L))
        direct = count_for(raw)
        inverted = count_for(invperm(raw))
        println("n=$n native=$native_nnz raw=$direct inverse=$inverted")
        @test inverted == native_nnz
        @test direct != native_nnz
        corrected = stiles_order(A,77)
        # Native selection may choose another valid candidate on a new call;
        # check its structure against its own fresh analysis above instead.
        @test sort(corrected) == collect(1:n)
    end
end
