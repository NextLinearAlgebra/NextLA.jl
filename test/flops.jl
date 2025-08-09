function mul_trsm(m, n)
    mult = 0.5 * n * m * (m+1.0)
    return mult
end

function add_trsm(m, n)
    adds = 0.5 * n * m * (m-1.0)
    return adds
end

function flops_trsm(::Type{T}, m, n) where T
    adds = add_trsm(m, n)
    mult = mul_trsm(m, n)
    if T <: Complex
        mult = 6 * mult
        adds = 2 * adds
    end
    return mult + adds
end

function mul_potrf(n)
    mult=((n) * (((1.0 / 6.0) * (n) + 0.5) * (n) + (1.0 / 3.0)))
    return mult
end

function add_potrf(n)
    adds=(((n) * (((1. / 6.) * (n)) * (n) - (1. / 6.))))
    return adds
end

function flops_potrf(::Type{T}, n) where T
    mult = mul_potrf(n)
    adds = add_potrf(n)
    if T <: Complex
        mult = 6 * mult
        adds = 2 * adds
    end
    return mult + adds
end

function muls_syrk(n, k)
    return 0.5 * k * n * (n + 1)
end

function adds_syrk(n, k)
    return 0.5 * k * n * (n + 1)
end

function flops_syrk(::Type{T}, n, k) where T
    mult = muls_syrk(n, k)
    adds = adds_syrk(n, k)
    if T <: Complex
        mult = 6 * mult
        adds = 2 * adds
    end
    return mult + adds
end

function calculate_gflops(flops::Real, time_ns::Real)
    if time_ns == 0
        return 0.0
    end
    time_s = time_ns / 1e9
    gflops = (flops / time_s) / 1e9
    return gflops
end