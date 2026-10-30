# Check the native factor/solve path before starting expensive catalog cases.
using LinearAlgebra, SparseArrays, Test, sTiles

@testset "sTiles native factorization and solve" begin
    cores = parse(Int, get(ENV, "STILES_CORES", get(ENV, "SLURM_CPUS_PER_TASK", "1")))
    A = spdiagm(-1 => fill(-1.0, 5), 0 => fill(4.0, 6), 1 => fill(-1.0, 5))
    truth = reshape(collect(1.0:12.0), 6, 2)
    rhs = A * truth
    F = sTiles.analyze(A; cores=cores, inverse=false)
    try
        sTiles.factorize!(F)
        solution = hcat((F \ rhs[:, j] for j in axes(rhs, 2))...)
        @test norm(solution - truth) / norm(truth) < 1e-12
        @test norm(A * solution - rhs) / norm(rhs) < 1e-12
    finally
        close(F)
    end
end
