export rademacher_word, rademacher_sign

const RademacherSeed = UInt64

@inline function _splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

@inline function _rademacher_word(k::Int,
                                  word_block::Int,
                                  batch::Int,
                                  seed::RademacherSeed,
                                  ::Val{true})
    counter = seed
    counter ⊻= UInt64(batch - 1) * 0x8cb92baa2df21a8d
    counter ⊻= UInt64(k - 1) * 0xd6e8feb86659fd93
    counter ⊻= UInt64(word_block - 1) * 0xa5a3564e27f88691
    return UInt32(_splitmix64(counter) & 0xffffffff)
end

@inline function _rademacher_word(k::Int,
                                  word_block::Int,
                                  batch::Int,
                                  seed::RademacherSeed,
                                  ::Val{false})
    counter = seed
    counter ⊻= UInt64(k - 1) * 0xd6e8feb86659fd93
    counter ⊻= UInt64(word_block - 1) * 0xa5a3564e27f88691
    return UInt32(_splitmix64(counter) & 0xffffffff)
end

"""
    rademacher_word(k, word_block, seed; batch=1, use_batch_in_rng=true)

Return one bit-packed `UInt32` word containing 32 deterministic Rademacher
signs. When `use_batch_in_rng=true`, the tile batch index participates in the
counter; otherwise the same implicit `Ω` is reused for every tile.
"""
@inline function rademacher_word(k::Integer,
                                 word_block::Integer,
                                 seed::Integer;
                                 batch::Integer=1,
                                 use_batch_in_rng::Bool=true)
    k >= 1 || throw(ArgumentError("k must be positive"))
    word_block >= 1 || throw(ArgumentError("word_block must be positive"))
    batch >= 1 || throw(ArgumentError("batch must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    return _rademacher_word(Int(k), Int(word_block), Int(batch), UInt64(seed), Val(use_batch_in_rng))
end

"""
    rademacher_sign(::Type{T}, k, s, seed; batch=1, use_batch_in_rng=true)

Return a single scaled Rademacher entry from the exact same stateless bit-packed
generator used by the sampling kernels.
"""
@inline function rademacher_sign(::Type{T},
                                 k::Integer,
                                 s::Integer,
                                 seed::Integer;
                                 batch::Integer=1,
                                 use_batch_in_rng::Bool=true) where {T}
    s >= 1 || throw(ArgumentError("s must be positive"))
    word_block = ((Int(s) - 1) >>> 5) + 1
    bit = (Int(s) - 1) & 31
    word = rademacher_word(k, word_block, seed; batch, use_batch_in_rng)
    sign = ((word >> bit) & UInt32(1)) == UInt32(1) ? one(T) : -one(T)
    return sign
end

@kernel function sample_range_rademacher_kernel!(Y::AbstractArray{T,3},
                                                 A::AbstractMatrix{T},
                                                 layout::Layout,
                                                 S::Int,
                                                 seed::RademacherSeed,
                                                 ::Val{ROWS_CTA},
                                                 ::Val{BK},
                                                 ::Val{SBLK},
                                                 ::Val{CPW},
                                                 ::Val{THREADS_ROW},
                                                 ::Val{ROWS_PER_THREAD},
                                                 ::Val{USE_BATCH_IN_RNG},
                                                 ::Val{OFFDIAG_ONLY}) where {T,Layout<:TileMap,ROWS_CTA,BK,SBLK,CPW,THREADS_ROW,ROWS_PER_THREAD,USE_BATCH_IN_RNG,OFFDIAG_ONLY}
    cta_row_block, sblk_block, batch = @index(Group, NTuple)
    thread_row, thread_s_chunk = @index(Local, NTuple)
    tid = @index(Local, Linear)

    @uniform group_nthreads = @groupsize()[1] * @groupsize()[2]
    @uniform n_omega_words = SBLK >>> 5

    linear_batch = OFFDIAG_ONLY ? offdiag_linear_index(layout, batch) : batch
    tile_i, tile_j = inverse_tile_index(layout, linear_batch)
    p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
    tile_m, tile_n = tile_sizes(layout, tile_i, tile_j)
    cta_row_start = (cta_row_block - 1) * ROWS_CTA
    sblk_start = (sblk_block - 1) * SBLK + 1
    sblk_valid = min(SBLK, S - sblk_start + 1)
    full_sblk = sblk_valid == SBLK
    valid_word_count = cld(sblk_valid, 32)
    global_word_base = ((sblk_start - 1) >>> 5) + 1
    sreg_local = (thread_s_chunk - 1) * CPW + 1
    valid_s_cols = sreg_local <= sblk_valid ? min(CPW, sblk_valid - sreg_local + 1) : 0
    full_s_chunk = valid_s_cols == CPW
    full_k_end = (tile_n ÷ BK) * BK
    tail_k = tile_n - full_k_end

    Asmem = @localmem T (ROWS_CTA, BK)
    OmegaBits = @localmem UInt32 (BK, SBLK >>> 5)
    acc = @private T (ROWS_PER_THREAD * CPW,)
    row_local = @private Int (ROWS_PER_THREAD,)
    row_in_tile_local = @private Int (ROWS_PER_THREAD,)
    row_active = @private Bool (ROWS_PER_THREAD,)
    sign_bits_cache = @private UInt32 (BK,)

    if sblk_start <= S
        @unroll for row_slot in 1:ROWS_PER_THREAD
            local_row = thread_row + (row_slot - 1) * THREADS_ROW
            row_in_tile = cta_row_start + local_row
            row_local[row_slot] = local_row
            row_in_tile_local[row_slot] = row_in_tile
            row_active[row_slot] = row_in_tile <= tile_m
            acc_base = (row_slot - 1) * CPW
            @unroll for c in 1:CPW
                @inbounds acc[acc_base + c] = zero(T)
            end
        end

        for kblk_start in 1:BK:full_k_end
            for linear_idx in tid:group_nthreads:(ROWS_CTA * BK)
                local_row = ((linear_idx - 1) % ROWS_CTA) + 1
                local_k = ((linear_idx - 1) ÷ ROWS_CTA) + 1
                row_in_tile = cta_row_start + local_row

                if row_in_tile <= tile_m
                    @inbounds Asmem[local_row, local_k] = A[p0 + row_in_tile - 1, q0 + kblk_start + local_k - 2]
                else
                    @inbounds Asmem[local_row, local_k] = zero(T)
                end
            end

            if full_sblk
                for linear_idx in tid:group_nthreads:(BK * n_omega_words)
                    local_k = ((linear_idx - 1) % BK) + 1
                    local_word = ((linear_idx - 1) ÷ BK) + 1
                    @inbounds OmegaBits[local_k, local_word] =
                        _rademacher_word(kblk_start + local_k - 1, global_word_base + local_word - 1, batch, seed, Val(USE_BATCH_IN_RNG))
                end
            else
                for linear_idx in tid:group_nthreads:(BK * n_omega_words)
                    local_k = ((linear_idx - 1) % BK) + 1
                    local_word = ((linear_idx - 1) ÷ BK) + 1

                    if local_word <= valid_word_count
                        @inbounds OmegaBits[local_k, local_word] =
                            _rademacher_word(kblk_start + local_k - 1, global_word_base + local_word - 1, batch, seed, Val(USE_BATCH_IN_RNG))
                    else
                        @inbounds OmegaBits[local_k, local_word] = UInt32(0)
                    end
                end
            end
            @synchronize

            if valid_s_cols > 0
                word_local_idx = ((sreg_local - 1) >>> 5) + 1
                bit_offset = (sreg_local - 1) & 31
                spans_two_words = bit_offset + CPW - 1 >= 32

                @unroll for local_k in 1:BK
                    word_lo = @inbounds OmegaBits[local_k, word_local_idx]
                    word_hi = spans_two_words && word_local_idx < n_omega_words ?
                        @inbounds(OmegaBits[local_k, word_local_idx + 1]) :
                        UInt32(0)
                    @inbounds sign_bits_cache[local_k] =
                        (word_lo >> bit_offset) | (spans_two_words ? (word_hi << (32 - bit_offset)) : UInt32(0))
                end

                if full_s_chunk
                    @unroll for row_slot in 1:ROWS_PER_THREAD
                        if @inbounds row_active[row_slot]
                            local_row = @inbounds row_local[row_slot]
                            acc_base = (row_slot - 1) * CPW
                            @unroll for local_k in 1:BK
                                a = @inbounds Asmem[local_row, local_k]
                                sign_bits = @inbounds sign_bits_cache[local_k]
                                @unroll for c in 1:CPW
                                    @inbounds acc[acc_base + c] += ifelse(((sign_bits >> (c - 1)) & UInt32(1)) == UInt32(1), a, -a)
                                end
                            end
                        end
                    end
                else
                    @unroll for row_slot in 1:ROWS_PER_THREAD
                        if @inbounds row_active[row_slot]
                            local_row = @inbounds row_local[row_slot]
                            acc_base = (row_slot - 1) * CPW
                            @unroll for local_k in 1:BK
                                a = @inbounds Asmem[local_row, local_k]
                                sign_bits = @inbounds sign_bits_cache[local_k]
                                @unroll for c in 1:CPW
                                    if c <= valid_s_cols
                                        @inbounds acc[acc_base + c] += ifelse(((sign_bits >> (c - 1)) & UInt32(1)) == UInt32(1), a, -a)
                                    end
                                end
                            end
                        end
                    end
                end
            end
            @synchronize
        end

        if tail_k > 0
            kblk_start = full_k_end + 1

            for linear_idx in tid:group_nthreads:(ROWS_CTA * BK)
                local_row = ((linear_idx - 1) % ROWS_CTA) + 1
                local_k = ((linear_idx - 1) ÷ ROWS_CTA) + 1
                row_in_tile = cta_row_start + local_row

                if row_in_tile <= tile_m && local_k <= tail_k
                    @inbounds Asmem[local_row, local_k] = A[p0 + row_in_tile - 1, q0 + kblk_start + local_k - 2]
                else
                    @inbounds Asmem[local_row, local_k] = zero(T)
                end
            end

            for linear_idx in tid:group_nthreads:(BK * n_omega_words)
                local_k = ((linear_idx - 1) % BK) + 1
                local_word = ((linear_idx - 1) ÷ BK) + 1

                if local_k <= tail_k && local_word <= valid_word_count
                    @inbounds OmegaBits[local_k, local_word] =
                        _rademacher_word(kblk_start + local_k - 1, global_word_base + local_word - 1, batch, seed, Val(USE_BATCH_IN_RNG))
                else
                    @inbounds OmegaBits[local_k, local_word] = UInt32(0)
                end
            end
            @synchronize

            if valid_s_cols > 0
                word_local_idx = ((sreg_local - 1) >>> 5) + 1
                bit_offset = (sreg_local - 1) & 31
                spans_two_words = bit_offset + CPW - 1 >= 32

                @unroll for local_k in 1:BK
                    if local_k <= tail_k
                        word_lo = @inbounds OmegaBits[local_k, word_local_idx]
                        word_hi = spans_two_words && word_local_idx < n_omega_words ?
                            @inbounds(OmegaBits[local_k, word_local_idx + 1]) :
                            UInt32(0)
                        @inbounds sign_bits_cache[local_k] =
                            (word_lo >> bit_offset) | (spans_two_words ? (word_hi << (32 - bit_offset)) : UInt32(0))
                    else
                        @inbounds sign_bits_cache[local_k] = UInt32(0)
                    end
                end

                if full_s_chunk
                    @unroll for row_slot in 1:ROWS_PER_THREAD
                        if @inbounds row_active[row_slot]
                            local_row = @inbounds row_local[row_slot]
                            acc_base = (row_slot - 1) * CPW
                            @unroll for local_k in 1:BK
                                if local_k <= tail_k
                                    a = @inbounds Asmem[local_row, local_k]
                                    sign_bits = @inbounds sign_bits_cache[local_k]
                                    @unroll for c in 1:CPW
                                        @inbounds acc[acc_base + c] += ifelse(((sign_bits >> (c - 1)) & UInt32(1)) == UInt32(1), a, -a)
                                    end
                                end
                            end
                        end
                    end
                else
                    @unroll for row_slot in 1:ROWS_PER_THREAD
                        if @inbounds row_active[row_slot]
                            local_row = @inbounds row_local[row_slot]
                            acc_base = (row_slot - 1) * CPW
                            @unroll for local_k in 1:BK
                                if local_k <= tail_k
                                    a = @inbounds Asmem[local_row, local_k]
                                    sign_bits = @inbounds sign_bits_cache[local_k]
                                    @unroll for c in 1:CPW
                                        if c <= valid_s_cols
                                            @inbounds acc[acc_base + c] += ifelse(((sign_bits >> (c - 1)) & UInt32(1)) == UInt32(1), a, -a)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            @synchronize
        end

        if valid_s_cols > 0
            if full_s_chunk
                @unroll for row_slot in 1:ROWS_PER_THREAD
                    row_in_tile = @inbounds row_in_tile_local[row_slot]
                    if row_in_tile <= tile_m
                        acc_base = (row_slot - 1) * CPW
                        @unroll for c in 1:CPW
                            s_global = sblk_start + sreg_local + c - 2
                            if s_global <= size(Y, 2)
                                @inbounds Y[row_in_tile, s_global, batch] = acc[acc_base + c]
                            end
                        end
                    end
                end
            else
                @unroll for row_slot in 1:ROWS_PER_THREAD
                    row_in_tile = @inbounds row_in_tile_local[row_slot]
                    if row_in_tile <= tile_m
                        acc_base = (row_slot - 1) * CPW
                        @unroll for c in 1:CPW
                            if c <= valid_s_cols
                                s_global = sblk_start + sreg_local + c - 2
                                if s_global <= size(Y, 2)
                                    @inbounds Y[row_in_tile, s_global, batch] = acc[acc_base + c]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function sample_range_rademacher_reference!(Y::AbstractArray{T,3},
                                            A::AbstractMatrix{T},
                                            layout::TileMap;
                                            S::Int = size(Y, 2),
                                            seed::Integer = 0x123456789abcdef0,
                                            CPW::Int = 4,
                                            use_batch_in_rng::Bool = true,
                                            offdiag_only::Bool = false) where {T}
    S > 0 || throw(ArgumentError("S must be positive"))
    CPW > 0 || throw(ArgumentError("CPW must be positive"))
    size(A) == (layout.m, layout.n) || throw(DimensionMismatch("dense matrix size must match the layout"))
    size(Y, 1) >= layout.tile_m || throw(DimensionMismatch("Y must have at least $(layout.tile_m) rows"))
    size(Y, 2) >= S || throw(DimensionMismatch("Y must have at least $S columns"))
    ntiles = ifelse(offdiag_only, noffdiag_tiles(layout), prod(size(layout)))
    size(Y, 3) >= ntiles ||
        throw(DimensionMismatch("Y must have at least $ntiles batches"))
    fill!(Y, zero(T))

    for batch in 1:ntiles
        linear_batch = offdiag_only ? offdiag_linear_index(layout, batch) : batch
        tile_i, tile_j = inverse_tile_index(layout, linear_batch)
        p0, q0 = tile_origin_coords(layout, tile_i, tile_j)
        tile_m, tile_n = tile_sizes(layout, tile_i, tile_j)

        for row_in_tile in 1:tile_m
            for sreg_start in 1:CPW:S
                acc = zeros(T, CPW)
                for k in 1:tile_n
                    a = A[p0 + row_in_tile - 1, q0 + k - 1]
                    for c in 1:CPW
                        s = sreg_start + c - 1
                        if s <= S
                            acc[c] += rademacher_sign(T, k, s, seed; batch, use_batch_in_rng) * a
                        end
                    end
                end
                for c in 1:CPW
                    s = sreg_start + c - 1
                    if s <= S
                        Y[row_in_tile, s, batch] = acc[c]
                    end
                end
            end
        end
    end

    return Y
end

"""
    sample_range_rademacher!(Y, A, tile_order, mt, nt; S=size(Y,2), seed=..., nthreads=256, ROWS_CTA=128, BK=16, SBLK=32, CPW=4, use_batch_in_rng=true, backend=get_backend(A))

Sample every logical dense tile `A_tile` into `Y[:, :, batch] = A_tile * Ω`
using an implicit Rademacher sketch matrix generated from a stateless bit-packed
counter-based sampler.

On the CPU backend this wrapper falls back to the exact reference
implementation so the same public API remains available in tests and benchmarks.
"""
function sample_range_rademacher!(Y::AbstractArray{T,3},
                                  A::AbstractMatrix{T},
                                  layout::TileMap;
                                  S::Int = size(Y, 2),
                                  seed::Integer = 0x123456789abcdef0,
                                  nthreads::Int = 256,
                                  ROWS_CTA::Int = 128,
                                  BK::Int = 16,
                                  SBLK::Int = 32,
                                  CPW::Int = 4,
                                  use_batch_in_rng::Bool = true,
                                  offdiag_only::Bool = false,
                                  backend = KernelAbstractions.get_backend(A)) where {T}
    S > 0 || throw(ArgumentError("S must be positive"))
    nthreads > 0 || throw(ArgumentError("nthreads must be positive"))
    ROWS_CTA > 0 || throw(ArgumentError("ROWS_CTA must be positive"))
    BK > 0 || throw(ArgumentError("BK must be positive"))
    SBLK > 0 || throw(ArgumentError("SBLK must be positive"))
    CPW > 0 || throw(ArgumentError("CPW must be positive"))
    SBLK % 32 == 0 || throw(ArgumentError("SBLK must be divisible by 32"))
    SBLK % CPW == 0 || throw(ArgumentError("SBLK must be divisible by CPW"))
    size(A) == (layout.m, layout.n) || throw(DimensionMismatch("dense matrix size must match the layout"))
    size(Y, 1) >= layout.tile_m || throw(DimensionMismatch("Y must have at least $(layout.tile_m) rows"))
    size(Y, 2) >= S || throw(DimensionMismatch("Y must have at least $S columns"))
    ntiles = ifelse(offdiag_only, noffdiag_tiles(layout), prod(size(layout)))
    size(Y, 3) >= ntiles ||
        throw(DimensionMismatch("Y must have at least $ntiles batches"))

    if backend isa KernelAbstractions.CPU
        return sample_range_rademacher_reference!(
            Y, A, layout;
            S,
            seed,
            CPW,
            use_batch_in_rng,
            offdiag_only,
        )
    end

    s_chunks = SBLK ÷ CPW
    nthreads % s_chunks == 0 ||
        throw(ArgumentError("nthreads must be divisible by SBLK ÷ CPW"))
    threads_row = nthreads ÷ s_chunks
    threads_row > 0 || throw(ArgumentError("output-stationary launch needs at least one row thread"))
    threads_row <= ROWS_CTA ||
        throw(ArgumentError("nthreads must be <= ROWS_CTA * (SBLK ÷ CPW)"))
    rows_per_thread = cld(ROWS_CTA, threads_row)

    kernel! = sample_range_rademacher_kernel!(backend, (threads_row, s_chunks))
    kernel!(
        Y,
        A,
        layout,
        S,
        RademacherSeed(seed),
        Val(ROWS_CTA),
        Val(BK),
        Val(SBLK),
        Val(CPW),
        Val(threads_row),
        Val(rows_per_thread),
        Val(use_batch_in_rng),
        Val(offdiag_only);
        ndrange=(cld(layout.tile_m, ROWS_CTA) * threads_row, cld(S, SBLK) * s_chunks, ntiles),
    )
    return Y
end

function sample_range_rademacher_reference!(Y::AbstractArray{T,3},
                                            A::AbstractMatrix{T},
                                            tile_order::TileOrder,
                                            mt::Int,
                                            nt::Int;
                                            kwargs...) where {T}
    layout = TileMap(tile_order, mt, nt, size(A, 1), size(A, 2))
    return sample_range_rademacher_reference!(Y, A, layout; kwargs...)
end

function sample_range_rademacher!(Y::AbstractArray{T,3},
                                  A::AbstractMatrix{T},
                                  tile_order::TileOrder,
                                  mt::Int,
                                  nt::Int;
                                  kwargs...) where {T}
    layout = TileMap(tile_order, mt, nt, size(A, 1), size(A, 2))
    return sample_range_rademacher!(Y, A, layout; kwargs...)
end
