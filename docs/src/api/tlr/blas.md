```@meta
CurrentModule = NextLA
```

# Tile low-rank BLAS

Matrix multiplication with TLR operands, in `src/tlr/blas`: `NextLA.gemm!` into a dense result,
`NextLA.gemm` into a compressed one, and the workspace sizing around them.

```@autodocs
Modules = [NextLA.TileLowRank]
Pages = [
    joinpath("blas", "gemm", "arena.jl"),
    joinpath("blas", "gemm", "compressed_accumulation", "driver.jl"),
    joinpath("blas", "gemm", "compressed_accumulation", "rolling_schedule.jl"),
    joinpath("blas", "gemm", "compressed_accumulation", "run_coupling.jl"),
    joinpath("blas", "gemm", "compressed_accumulation", "workspace.jl"),
    joinpath("blas", "gemm", "compute_policy.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "dense_diagonal.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "driver.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "runs.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "schedule.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "three_stage.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "two_stage.jl"),
    joinpath("blas", "gemm", "dense_accumulation", "workspace.jl"),
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```
