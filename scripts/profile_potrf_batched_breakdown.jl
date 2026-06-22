using CUDA
using KernelAbstractions
using LinearAlgebra
using Statistics
using NextLA

const T = Float32

_env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function make_spd_batch_host(n::Int, batch::Int, ::Type{T}) where {T}
    A = Array{T,3}(undef, n, n, batch)
    for b in 1:batch
        X = randn(T, n, n)
        A[:, :, b] = X * X' + T(n) * I
    end
    return A
end

function timed!(f, timer::Base.RefValue{Float64})
    t = @elapsed begin
        f()
        CUDA.synchronize()
    end
    timer[] += t
    return nothing
end

function profile_once!(A, status, ::Val{OB}, ::Val{IB}, ::Val{RB}, ::Val{MR}) where {OB,IB,RB,MR}
    backend = KernelAbstractions.get_backend(A)
    W = NextLA._val_parameter(NextLA.SUBGROUP_SIZE(typeof(backend)))
    n = NextLA._potrf_validate!(A, status)
    trans = eltype(A) <: Real ? 'T' : 'C'
    alpha = one(eltype(A))
    host_status = NextLA._potrf_status_buffer(A)
    batch_count = ndims(A) == 2 ? 1 : size(A, 3)
    uplo = 'L'
    uplo_val = Val(:L)
    fill!(status, Int32(0))

    fused_tile_kernel = NextLA._potrf_fused_tile_kernel!(backend, (512, 1, 1))
    partial_kernel = NextLA._potf2_partial_kernel!(backend, (256, 1, 1))

    fused_kernel_t = Ref(0.0)
    fused_sync_status_t = Ref(0.0)
    internal_syrk_t = Ref(0.0)
    internal_trsm_t = Ref(0.0)
    partial_kernel_t = Ref(0.0)
    partial_sync_status_t = Ref(0.0)
    global_trsm_t = Ref(0.0)
    global_syrk_t = Ref(0.0)
    final_status_t = Ref(0.0)

    total_t = @elapsed begin
        @inbounds for k0 in 1:OB:n
            k1 = min(k0 + OB - 1, n)
            diag_tile = NextLA._potrf_tile_view(A, k0, k1)
            outer_n = size(diag_tile, 1)
            active_before = iszero.(host_status)
            fused_n = (outer_n ÷ IB) * IB
            partial_n = outer_n - fused_n

            if fused_n > 0
                fused_tile = NextLA._potrf_tile_view(diag_tile, 1, fused_n)
                for k0_fused in 1:IB:fused_n
                    timed!(fused_kernel_t) do
                        fused_tile_kernel(fused_tile, status, k0_fused, uplo_val, Val(fused_n), Val(IB), Val(RB), Val(MR), Val(W); ndrange=(512, 1, batch_count))
                    end
                    t = @elapsed begin
                        copyto!(host_status, status)
                    end
                    fused_sync_status_t[] += t
                    all(x -> !iszero(x), host_status) && break

                    kt = k0_fused + IB
                    if kt <= fused_n
                        panel = NextLA._potrf_panel_view(uplo_val, fused_tile, k0_fused, kt - 1, fused_n)
                        trailing = NextLA._potrf_trailing_view(fused_tile, kt - 1, fused_n)
                        timed!(internal_syrk_t) do
                            NextLA._potrf_syrk_dispatch!(trailing, panel, uplo, 'N', alpha)
                        end
                    end
                end

                final_copy = @elapsed copyto!(host_status, status)
                fused_sync_status_t[] += final_copy
                if all(x -> !iszero(x), host_status)
                    NextLA._potrf_translate_status!(host_status, active_before, k0) && copyto!(status, host_status)
                    break
                end
            end

            if partial_n > 0
                if fused_n > 0
                    fused_tile = NextLA._potrf_tile_view(diag_tile, 1, fused_n)
                    panel = NextLA._potrf_panel_view(uplo_val, diag_tile, 1, fused_n, outer_n)
                    trailing = NextLA._potrf_trailing_view(diag_tile, fused_n, outer_n)
                    timed!(internal_trsm_t) do
                        NextLA.trsm!('R', 'L', trans, 'N', fused_tile, panel, alpha)
                    end
                    timed!(internal_syrk_t) do
                        NextLA._potrf_syrk_dispatch!(trailing, panel, 'L', 'N', alpha)
                    end
                end

                partial_tile = NextLA._potrf_tile_view(diag_tile, fused_n + 1, outer_n)
                timed!(partial_kernel_t) do
                    partial_kernel(partial_tile, status, fused_n + 1, partial_n, uplo_val, Val(IB), Val(RB), Val(W); ndrange=(256, 1, batch_count))
                end
                t = @elapsed begin
                    copyto!(host_status, status)
                end
                partial_sync_status_t[] += t
            end

            NextLA._potrf_translate_status!(host_status, active_before, k0) && copyto!(status, host_status)
            all(!iszero, host_status) && break
            k1 == n && continue

            panel = NextLA._potrf_panel_view(uplo_val, A, k0, k1, n)
            trailing = NextLA._potrf_trailing_view(A, k1, n)
            timed!(global_trsm_t) do
                NextLA.trsm!('R', 'L', trans, 'N', diag_tile, panel, alpha)
            end
            timed!(global_syrk_t) do
                NextLA._potrf_syrk_dispatch!(trailing, panel, 'L', 'N', alpha)
            end
        end

        final_status_t[] += @elapsed copyto!(status, host_status)
    end

    return (
        total = total_t,
        fused_kernel = fused_kernel_t[],
        fused_sync_status = fused_sync_status_t[],
        internal_trsm = internal_trsm_t[],
        internal_syrk = internal_syrk_t[],
        partial_kernel = partial_kernel_t[],
        partial_sync_status = partial_sync_status_t[],
        global_trsm = global_trsm_t[],
        global_syrk = global_syrk_t[],
        final_status = final_status_t[],
    )
end

function summarize(label, values)
    println(label)
    println("  mean=$(mean(values))")
    println("  median=$(median(values))")
    println("  best=$(minimum(values))")
end

function main()
    n = _env_int("NEXTLA_POTRF_BATCH_N", 512)
    batch = _env_int("NEXTLA_POTRF_BATCH_COUNT", 100)
    warmup = _env_int("NEXTLA_POTRF_WARMUP", 1)
    iters = _env_int("NEXTLA_POTRF_ITERS", 10)
    ob = _env_int("NEXTLA_POTRF_OB", 64)
    ib = _env_int("NEXTLA_POTRF_IB", 64)
    rb = _env_int("NEXTLA_POTRF_RB", 8)
    mr = _env_int("NEXTLA_POTRF_MR", 16)

    println("potrf batched breakdown")
    println("  n=$n batch=$batch")
    println("  config=", (ob, ib, rb, mr))
    println("  warmup=$warmup iters=$iters")

    Aref = CuArray(make_spd_batch_host(n, batch, T))
    Awork = similar(Aref)
    status = similar(Aref, Int32, batch)

    for _ in 1:warmup
        copyto!(Awork, Aref)
        profile_once!(Awork, status, Val(ob), Val(ib), Val(rb), Val(mr))
    end

    results = NamedTuple[]
    for _ in 1:iters
        copyto!(Awork, Aref)
        push!(results, profile_once!(Awork, status, Val(ob), Val(ib), Val(rb), Val(mr)))
    end

    for field in (:total, :fused_kernel, :fused_sync_status, :internal_trsm, :internal_syrk,
                  :partial_kernel, :partial_sync_status, :global_trsm, :global_syrk, :final_status)
        summarize(string(field), getfield.(results, field))
    end

    best = results[argmin(getfield.(results, :total))]
    println("best iteration fractions")
    accounted = best.fused_kernel + best.fused_sync_status + best.internal_trsm + best.internal_syrk +
                best.partial_kernel + best.partial_sync_status + best.global_trsm + best.global_syrk + best.final_status
    for field in (:fused_kernel, :fused_sync_status, :internal_trsm, :internal_syrk,
                  :partial_kernel, :partial_sync_status, :global_trsm, :global_syrk, :final_status)
        value = getfield(best, field)
        println("  $field=$(value) frac_total=$(value / best.total) frac_accounted=$(value / accounted)")
    end
    println("  unaccounted=$(best.total - accounted)")
end

main()
