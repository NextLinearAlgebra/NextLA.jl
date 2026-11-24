```@meta
CurrentModule = NextLA
```

# Rotations

Plane (Givens) rotations in `src/dense/lapack/rotations`.

```@autodocs
Modules = [NextLA]
Pages = [joinpath("rotations", "lartg.jl")]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```
