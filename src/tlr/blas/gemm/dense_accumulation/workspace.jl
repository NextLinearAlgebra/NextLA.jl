"""
    DenseGemmWorkspace(A, bytes)
    DenseGemmWorkspace(A, B; bytes, transA='N', transB='N')

Reusable numerical storage for GEMMs whose destination is dense.

Both constructors allocate `fld(bytes, sizeof(T))` elements of `A`'s element
type `T` on `A`'s backend. The two-operand form first checks that `bytes` is at
least `gemm_minimum_workspace_bytes(A, B; transA, transB)` and throws an
`ArgumentError` otherwise. Pass the result as `workspace` to `gemm!` or
`analyze_compressed_gemm`; `sizeof(ws)` is its usable size in bytes.

# Examples

```jldoctest
julia> using NextLA.TileLowRank, LinearAlgebra

julia> n = 1024; K = [1/(1 + abs(i - j)) for i in 1:n, j in 1:n];

julia> A = TLRMatrix(K, (256, 256); maxrank=64, tol=1e-6, rel=true);

julia> ws = DenseGemmWorkspace(A, A; bytes=gemm_maximum_workspace_bytes(A, A));

julia> C = zeros(n, n);

julia> NextLA.gemm!(C, A, A; workspace=ws);

julia> NextLA.gemm!(C, A, A; workspace=ws, beta=1.0);

julia> norm(C - 2 * K * K) / norm(2 * K * K) < 1e-5
true
```
"""
struct DenseGemmWorkspace{T,A<:AbstractVector{T}}
    storage::A
end

Base.eltype(::DenseGemmWorkspace{T}) where {T} = T
Base.sizeof(ws::DenseGemmWorkspace) = sizeof(ws.storage)

function DenseGemmWorkspace(A::AbstractTLRMatrix{T}, bytes::Int) where {T}
    bytes >= 0 || throw(ArgumentError("workspace bytes must be nonnegative"))
    storage = allocate(get_backend(A), T, fld(bytes, sizeof(T)))
    return DenseGemmWorkspace(storage)
end

function DenseGemmWorkspace(A::AbstractTLRMatrix{T},
                            B::AbstractTLRMatrix{T};
                            bytes::Int,
                            transA::Char='N', transB::Char='N') where {T}
    required = gemm_minimum_workspace_bytes(A, B; transA, transB)
    bytes >= required || throw(ArgumentError(
        "workspace has $bytes bytes; at least $required bytes are required"))
    workspace = DenseGemmWorkspace(A, bytes)
    sizeof(workspace) >= required || throw(ArgumentError(
        "workspace byte count must contain at least $required bytes of complete $T elements"))
    return workspace
end

function prepare_dense_accumulation_workspace(
    A::AbstractTLRMatrix{T}, workspace) where {T}
    ws = if workspace isa Int
        DenseGemmWorkspace(A, workspace)
    elseif workspace isa DenseGemmWorkspace
        eltype(workspace) === T || throw(ArgumentError(
            "workspace element type $(eltype(workspace)) does not match operand type $T"))
        typeof(get_backend(workspace.storage)) === typeof(get_backend(A)) ||
            throw(ArgumentError("workspace and operands must use the same backend"))
        workspace
    else
        throw(ArgumentError(
            "workspace must be an integer byte count or DenseGemmWorkspace"))
    end

    return ws, GemmArena(view(ws.storage, :), 1), sizeof(ws)
end
