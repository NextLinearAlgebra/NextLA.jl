ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"
# To ensure that the plot doesn't try to open a window

using BenchmarkTools
using Plots
using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasInt, libblastrampoline
using NextLA
using Profile
using PProf

function profile_lasd7()
    T = Float64
    i = 15000
    starting = -T(1e3)
    ending = T(1e3)
    block_size = i ÷ 2
    nl = block_size - 1
    nr = block_size
    sqre = 0
    k = [0]
    # k_copy = [0]
    
    n = nl + nr + 1
    m = n + sqre
    
    A = Bidiagonal((starting .+ (ending - starting) .* rand(T, i, i)), :U)
    B1 = A[1:block_size-1, 1:block_size-1]
    B2 = A[block_size+1:end, block_size+1:end]
    U1, D1, V1 = svd(B1)
    U2, D2, V2 = svd(B2)

    D = [D1; 0 ; D2]
    D_copy = deepcopy(D)
    icompq = 1

    z = zeros(T, n)
    zw = zeros(T, m)
    vf = zeros(T, m)
    vf[1:nl] .= V1[1,:]
    vf[nl+1] = 1.5
    vf[nl+2:m] .= V2[1,:]
    vfw = zeros(T,m)
    vl = zeros(T, m)
    vl[1:nl] .= V1[end,:]
    vl[nl+1] = 0.5
    vl[nl+2:m] .= V2[end,:]
    vlw = zeros(T, m)
    
    alpha = rand(T)
    beta = rand(T)
    dsigma = zeros(T, n)
    
    idx = zeros(Int64, n)
    idxp = zeros(Int64, n)
    idxq = zeros(Int64, n)
    idxq[1:nl] = reverse(Vector(1:nl))
    idxq[nl+2:end] = reverse(Vector(1:nr))
    perm = zeros(Int64, n)
    givptr = [0]
    ldgcol = n
    ldgnum = n
    givnum = zeros(T, ldgnum, 2)
    givcol = zeros(Int64, ldgcol, 2)
    c = [T(0)]
    s = [T(0)]
    info = [0]
    
    k_native = deepcopy(k)
    D_native  = deepcopy(D)
    z_native = deepcopy(z)
    zw_native = deepcopy(zw)
    vf_native = deepcopy(vf)
    vfw_native = deepcopy(vfw)
    vl_native = deepcopy(vl)
    vlw_native = deepcopy(vlw)
    dsigma_native = deepcopy(dsigma)
    idx_native = deepcopy(idx)
    idxp_native = deepcopy(idxp)
    idxq_native = deepcopy(idxq)
    perm_native = deepcopy(perm)
    givptr_native = deepcopy(givptr)
    givcol_native = deepcopy(givcol)
    givnum_native = deepcopy(givnum)
    c_native = deepcopy(c)
    s_native = deepcopy(s)
    info_native = deepcopy(info)

    NextLA.lasd7!(
    icompq, nl, nr, sqre,
    k_native, D_native, z_native, zw_native, vf_native, 
    vfw_native, vl_native, vlw_native,
    alpha, beta, dsigma_native,
    idx_native, idxp_native, idxq_native, perm_native,
    givptr_native, givcol_native, ldgcol,
    givnum_native, ldgnum, c_native, s_native, info_native
    )

    Profile.clear()
    # Profile.init(n=10^12, delay=1e-6)
    function run_many_iterations()

        for _ in 1:5_000
                copyto!(k_native, k)
                copyto!(D_native, D)
                copyto!(z_native, z)
                copyto!(zw_native, zw)
                copyto!(vf_native, vf)
                copyto!(vfw_native, vfw)
                copyto!(vl_native, vl)
                copyto!(vlw_native, vlw)
                copyto!(dsigma_native, dsigma)
                copyto!(idx_native, idx)
                copyto!(idxp_native, idxp)
                copyto!(idxq_native, idxq)
                copyto!(perm_native, perm)
                copyto!(givptr_native, givptr)
                copyto!(givcol_native, givcol)
                copyto!(givnum_native, givnum)
                copyto!(c_native, c)
                copyto!(s_native, s)
                copyto!(info_native, info)

                NextLA.lasd7!(
                icompq, nl, nr, sqre,
                k_native, D_native, z_native, zw_native, vf_native, 
                vfw_native, vl_native, vlw_native,
                alpha, beta, dsigma_native,
                idx_native, idxp_native, idxq_native, perm_native,
                givptr_native, givcol_native, ldgcol,
                givnum_native, ldgnum, c_native, s_native, info_native
                )
        end
    end

    # b = Profile.@profile NextLA.lasd7!(
    #         icompq, nl, nr, sqre,
    #         k, D, z, zw, vf, vfw, vl, vlw,
    #         alpha, beta, dsigma,
    #         idx, idxp, idxq, perm,
    #         givptr, givcol, ldgcol,
    #         givnum, ldgnum, c, s, info
    #     )
    Profile.@profile run_many_iterations()
    Profile.print()
    pprof() 

end


profile_lasd7()
