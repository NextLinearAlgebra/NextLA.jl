```@meta
CurrentModule = NextLA
```

# QR and Householder kernels

Julia versions of the LAPACK and PLASMA kernels used by tiled QR factorizations. Most routines
come in two forms: a low-level one with explicit sizes and workspace that mirrors the LAPACK
signature, and a convenience one that infers them. These routines are not exported; call them as
`NextLA.geqrt!`, etc.

| Kernel                     | Purpose                                                            |
|:---------------------------|:-------------------------------------------------------------------|
| `larfg!`                   | generate an elementary Householder reflector                       |
| `larf!`                    | apply one reflector                                                |
| `larft!`                   | form the triangular factor `T` of a block reflector                |
| `larfb!`                   | apply a block reflector                                            |
| `geqr2!`                   | unblocked QR                                                       |
| `geqrt!`                   | blocked QR with compact-WY `T` factors                             |
| `unmqr!`                   | apply `Q` or `Qᵀ` from `geqrt!`                                    |
| `tsqrt!` / `tsmqr!`        | QR of a triangle on top of a square tile, and applying its `Q`     |
| `ttqrt!` / `ttmqr!`        | QR of a triangle on top of a triangle, and applying its `Q`        |
| `parfb!`, `pamm!`, `pemv!` | pentagonal block-reflector application and its building blocks     |

Plane (Givens) rotations, `lartg`, are on the [Rotations](@ref) page.

## Householder reflectors

```@autodocs
Modules = [NextLA]
Pages = ["larfg.jl", "larf.jl", "larft.jl", "larfb.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Tile QR

```@autodocs
Modules = [NextLA]
Pages = ["geqr2.jl", "geqrt.jl", "unmqr.jl", "tsqrt.jl", "tsmqr.jl", "ttqrt.jl", "ttmqr.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Pentagonal kernels

```@autodocs
Modules = [NextLA]
Pages = ["parfb.jl", "pamm.jl", "pemv.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Each example stands on its own and ends with a check that prints `true`.

Least squares with a blocked QR: [`geqrt!`](@ref) factors `A`, [`unmqr!`](@ref) applies `Qᴴ` to
the right-hand side, and a triangular solve with `R` finishes:

```jldoctest
using NextLA, LinearAlgebra

m, n, ib = 2048, 1024, 64
A = rand(m, n)
b = rand(m)

F = copy(A)
T = zeros(ib, n)
NextLA.geqrt!(F, T)                  # R in triu(F), reflectors below it

c = reshape(copy(b), m, 1)
NextLA.unmqr!('L', 'C', F, T, c)     # c := Qᴴ * b
x = UpperTriangular(F[1:n, :]) \ c[1:n]
x ≈ A \ b

# output

true
```

One blocked QR step built from the Householder kernels: [`geqr2!`](@ref) factors a 128-column
panel, [`larft!`](@ref) forms its block reflector and [`larfb!`](@ref) applies it to the trailing
columns:

```jldoctest
using NextLA, LinearAlgebra

n, nb = 1024, 128
A0 = rand(n, n)
A = copy(A0)
panel, rest = view(A, :, 1:nb), view(A, :, nb+1:n)

tau = zeros(nb)
NextLA.geqr2!(panel, tau)                           # unblocked QR of the panel
T = zeros(nb, nb)
NextLA.larft!('F', 'C', panel, tau, T)              # H = I - V * T * Vᴴ
NextLA.larfb!('L', 'C', 'F', 'C', panel, T, rest)   # rest := Hᴴ * rest

triu(A[1:nb, :]) ≈ qr(A0).R[1:nb, :]                # the first 128 rows of R

# output

true
```

Tile QR of a 1024 by 256 panel stored as four 256 by 256 tiles (a flat tree): [`geqrt!`](@ref)
factors the top tile and [`tsqrt!`](@ref) folds each tile below into its `R`; then
[`unmqr!`](@ref) and [`tsmqr!`](@ref) apply `Qᴴ` to other columns in the same order. `R` and the
top rows of `Qᴴ C` match `LinearAlgebra.qr` up to the signs of the rows:

```jldoctest
using NextLA, LinearAlgebra

nb, ib = 256, 32                          # tile size, inner block size
P = rand(4nb, nb)                         # the panel
C = rand(4nb, 1024)                       # columns to update with Qᴴ
A, B = copy(P), copy(C)
tile(X, i) = view(X, (i-1)*nb+1:i*nb, :)
Ts = [zeros(ib, nb) for _ in 1:4]

NextLA.geqrt!(tile(A, 1), Ts[1])
for i in 2:4
    NextLA.tsqrt!(tile(A, 1), tile(A, i), Ts[i])
end

NextLA.unmqr!('L', 'C', tile(A, 1), Ts[1], tile(B, 1))
for i in 2:4
    NextLA.tsmqr!('L', 'C', tile(B, 1), tile(B, i), tile(A, i), Ts[i])
end

F = qr(P)
abs.(triu(tile(A, 1))) ≈ abs.(F.R) && abs.(tile(B, 1)) ≈ abs.((F.Q' * C)[1:nb, :])

# output

true
```

The binary-tree variant on two tiles: each tile is factored with [`geqrt!`](@ref), then
[`ttqrt!`](@ref) merges the two triangles and [`ttmqr!`](@ref) applies the merge:

```jldoctest
using NextLA, LinearAlgebra

nb, ib = 256, 32
P = rand(2nb, nb)
C = rand(2nb, 1024)
A, B = copy(P), copy(C)
top, bot = 1:nb, nb+1:2nb
T1, T2, T12 = zeros(ib, nb), zeros(ib, nb), zeros(ib, nb)

NextLA.geqrt!(view(A, top, :), T1)
NextLA.geqrt!(view(A, bot, :), T2)
R2 = triu(A[bot, :])            # ttqrt! needs zeros below the second triangle
NextLA.ttqrt!(view(A, top, :), R2, T12)

NextLA.unmqr!('L', 'C', view(A, top, :), T1, view(B, top, :))
NextLA.unmqr!('L', 'C', view(A, bot, :), T2, view(B, bot, :))
NextLA.ttmqr!('L', 'C', view(B, top, :), view(B, bot, :), R2, T12)

F = qr(P)
abs.(triu(A[top, :])) ≈ abs.(F.R) && abs.(B[top, :]) ≈ abs.((F.Q' * C)[1:nb, :])

# output

true
```

The kernels are KernelAbstractions code and take device arrays directly. This block is not run
when the documentation is built:

```julia
using NextLA, CUDA, LinearAlgebra

A = rand(1024, 1024)
C = rand(1024, 1024)
dF, dT, dC = CuArray(A), CUDA.zeros(Float64, 64, 1024), CuArray(C)

NextLA.geqrt!(dF, dT)
NextLA.unmqr!('L', 'C', dF, dT, dC)       # dC := Qᴴ * C on the GPU
Array(dC) ≈ qr(A).Q' * C
```
