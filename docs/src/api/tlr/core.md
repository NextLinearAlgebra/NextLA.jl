```@meta
CurrentModule = NextLA
```

# Tile low-rank core

`NextLA.TileLowRank`, in `src/tlr/core`: tile low-rank (TLR) matrices, whose off-diagonal tiles are
stored as low-rank factors, and their compression from dense matrices. Load it with
`using NextLA.TileLowRank`; it is not re-exported by `NextLA`.

```@docs
NextLA.TileLowRank
```

## Containers

```@autodocs
Modules = [NextLA.TileLowRank]
Pages = [
    joinpath("core", "compressed_ftlr_matrix.jl"),
    joinpath("core", "interface.jl"),
    joinpath("core", "tlr_matrix.jl"),
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Compression

```@autodocs
Modules = [NextLA.TileLowRank]
Pages = [
    joinpath("core", "compression", "ara.jl"),
    joinpath("core", "compression", "blas_utils.jl"),
    joinpath("core", "compression", "compress.jl"),
    joinpath("core", "compression", "norms.jl"),
    joinpath("core", "compression", "uncompress.jl"),
    joinpath("core", "compression", "workspace.jl"),
]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```
