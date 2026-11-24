```@meta
CurrentModule = NextLA
```

# AXPY and GERC

Level 1 and 2 BLAS routines in `src/dense/blas`.

```@autodocs
Modules = [NextLA]
Pages = ["axpy.jl", "gerc.jl"]
Filter = x -> !(x isa Function && startswith(String(nameof(x)), "_"))
```

## Usage examples

Each example stands on its own and ends with a check that prints `true`. `axpy!` and `gerc!`
are not exported, so they are called as `NextLA.axpy!` and `NextLA.gerc!`.

### `axpy!` on long vectors

`NextLA.axpy!(alpha, x, y)` overwrites `y` with `alpha * x + y`; `alpha` must have the element
type of `x` and `y` (`2.0`, not `2`, for `Float64` vectors).

```jldoctest
using NextLA

x = rand(4096)
y = rand(4096)
y0 = copy(y)

NextLA.axpy!(2.0, x, y)        # y = 2.0 * x + y
y ≈ 2.0 * x + y0

# output

true
```

### Complex rank-1 update with `gerc!`

`gerc!` adds `alpha * x * y'` to a rectangular matrix, with `y` conjugated.

```jldoctest
using NextLA

m, n = 2048, 1024
x = rand(ComplexF64, m)
y = rand(ComplexF64, n)
A = rand(ComplexF64, m, n)
A0 = copy(A)

alpha = 0.5 - 1.0im
NextLA.gerc!(alpha, x, y, A)   # A = A + alpha * x * y'
A ≈ A0 + alpha * x * y'

# output

true
```

### On the GPU

`axpy!` and `gerc!` are written with broadcasts, so they run on GPU arrays unchanged (this block
is not run when the documentation is built).

```julia
using CUDA, NextLA

x = CUDA.rand(Float64, 2^20)
y = CUDA.rand(Float64, 2^20)
y0 = copy(y)
NextLA.axpy!(2.0, x, y)

u = CUDA.rand(ComplexF32, 4096)
v = CUDA.rand(ComplexF32, 4096)
A = CUDA.zeros(ComplexF32, 4096, 4096)
NextLA.gerc!(1.0f0 + 0.0f0im, u, v, A)

Array(y) ≈ 2.0 * Array(x) + Array(y0) && Array(A) ≈ Array(u) * Array(v)'   # true
```
