export SYRK_KERNEL!

# A tiled symmetric rank-k update written directly in KernelAbstractions, with
# no BLAS underneath. It is the portable counterpart to syrk! in
# syrk_dispatch.jl: that one needs a vendor library, this one runs on any
# backend KernelAbstractions supports.
#
# Source: vicki-development @ 694d019, by Vicki <vickicar@mit.edu>. Not
# verbatim -- as written the kernel is wrong on the CPU backend and silently
# ignores its own uplo argument. Four changes, all recorded in KNOWN_ISSUES.md:
#
#  1. N and M were computed inside the kernel body from size(C) and size(A). A
#     @synchronize splits the body into separate ndrange loops on the CPU
#     backend and only @localmem/@private survive the split, so both were out
#     of scope in the second half. They are kernel arguments now, which is why
#     matmul! in src/gemm/matmul.jl takes N, R and M rather than computing them.
#
#  2. I and J were likewise assigned before the first @synchronize and read
#     after it. `I` then resolves to LinearAlgebra.I and the kernel dies in
#     isless(::UniformScaling{Bool}, ::Int64). They are recomputed in every
#     synchronize-delimited block, exactly as src/gemm/matmul.jl:31-32,56-57 does.
#     This is the fifth routine in this repo to carry that bug.
#
#  3. Both @synchronize calls sat inside an `if gj <= gi` block that skipped
#     workgroups above the diagonal. matmul! has no such wrapper. The skip is
#     gone and the triangle is enforced at the write instead: every workgroup
#     now computes, and only the ones in the requested triangle store. That
#     costs roughly half the arithmetic on a backend that would have skipped
#     those groups, and buys a kernel whose synchronizes are unconditional.
#
#  4. uplo was accepted and ignored -- the store was hardcoded `I >= J`, lower
#     only. Both triangles work now, and the other one is left untouched, which
#     is the property that distinguishes SYRK from a GEMM.
#
# The branch's `const TILE_DIM = 32` is dropped: it duplicates src/gemm/matmul.jl:3
# and would be a redefinition in this module. The one in scope here is that
# same constant.

@kernel function syrk_kernel!(
    uplo::Char, trans::Char, alpha::Number, A::AbstractArray, beta::Number, C::AbstractArray,
    N::Int, M::Int, ::Val{BANK} = Val(1)
) where {BANK}

    gi, gj = @index(Group, NTuple)
    i,  j  = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]

    tile1 = @localmem eltype(C) (TILE_DIM + BANK, TILE_DIM)
    tile2 = @localmem eltype(C) (TILE_DIM + BANK, TILE_DIM)

    outval = @private eltype(C) 1
    @inbounds outval[1] = zero(eltype(C))

    @uniform NUM_TILES = ceil(Int, M / TILE_DIM)

    for t in 0:(NUM_TILES - 1)
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        K = t * TILE_DIM + j

        if I <= N && K <= M
            @inbounds tile1[i, j] = (trans == 'N' || trans == 'n') ? A[I, K] : A[K, I]
        else
            @inbounds tile1[i, j] = zero(eltype(C))
        end

        K = t * TILE_DIM + i
        if K <= M && J <= N
            @inbounds tile2[i, j] = (trans == 'N' || trans == 'n') ? A[J, K] : A[K, J]
        else
            @inbounds tile2[i, j] = zero(eltype(C))
        end

        @synchronize

        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j
        if I <= N && J <= N
            tmp = zero(eltype(C))
            @simd for k in 1:TILE_DIM
                @inbounds tmp += tile1[i, k] * tile2[k, j]
            end
            outval[1] += tmp
        end

        @synchronize
    end

    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j
    in_triangle = (uplo == 'L' || uplo == 'l') ? I >= J : I <= J
    if I <= N && J <= N && in_triangle
        @inbounds C[I, J] = alpha * outval[1] + beta * C[I, J]
    end
end

"""
    SYRK_KERNEL!(uplo, trans, alpha, A, beta, C)

Symmetric rank-k update `C := alpha * op(A) * op(A)' + beta * C`, computed with
KernelAbstractions rather than a vendor BLAS.

Only the `uplo` triangle of `C` is written; the other is left exactly as it
was. `trans` selects `op(A) = A` (`'N'`) or `op(A) = Aᵀ` (`'T'`).

Unlike [`syrk!`](@ref) this needs no backend library, so it is the path to use
on a backend with no native SYRK, or for an element type the vendor does not
cover. It does not synchronize; do that through the backend if the result is
read on the host.
"""
function SYRK_KERNEL!(uplo::Char, trans::Char, alpha::Number, A::AbstractArray, beta::Number, C::AbstractArray)
    (uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l') ||
        throw(ArgumentError("Unsupported uplo flag `$uplo`"))
    (trans == 'N' || trans == 'n' || trans == 'T' || trans == 't') ||
        throw(ArgumentError("Unsupported transpose flag `$trans`"))
    size(C, 1) == size(C, 2) || throw(DimensionMismatch("SYRK_KERNEL!: C must be square"))

    N = size(C, 1)
    M = (trans == 'N' || trans == 'n') ? size(A, 2) : size(A, 1)
    size(A, (trans == 'N' || trans == 'n') ? 1 : 2) == N ||
        throw(DimensionMismatch("SYRK_KERNEL!: A and C have incompatible dimensions"))

    backend = get_backend(A)
    padded = ceil(Int, N / TILE_DIM) * TILE_DIM
    syrk_kernel!(backend, (TILE_DIM, TILE_DIM))(
        uplo, trans, alpha, A, beta, C, N, M,
        ndrange = (padded, padded)
    )
    return C
end
