# One alignment variant, one process. A misaligned pointer HARD-FAULTS on
# tensor-core kernels and the error is sticky, so each variant must be isolated.
# Driven by experiments/run_all_h100.sh.
#
#   S=Float16 OFFA=1 OFFC=0 MODE=default julia --project=experiments experiments/align_one.jl
#
# lda/ldb/ldc are held FIXED across variants (PAD is constant): changing the
# allocation height would also change the leading dimension and confound
# pointer alignment with power-of-two stride aliasing.

using CUDA, NextLA, Printf

const S    = eval(Meta.parse(get(ENV, "S", "Float16")))
const OFFA = parse(Int, get(ENV, "OFFA", "0"))
const OFFC = parse(Int, get(ENV, "OFFC", "0"))
const MODE = get(ENV, "MODE", "default") == "tf32" ? NextLA.TF32() :
             (S === Float64 ? NextLA.GEMMCompute{Float64}() : NextLA.GEMMCompute{Float32}())
const LABEL = get(ENV, "LABEL", string(S))

const M, K, N, NT, PAD = 128, 128, 256, 1024, 8

function main()
    keep, tasks = Any[], NextLA.GroupedGemmTask[]
    for _ in 1:NT
        As = CUDA.zeros(S, M + PAD, K); fill!(As, S(0.01))
        Bs = CUDA.zeros(S, K + PAD, N); fill!(Bs, S(0.01))
        Cs = CUDA.zeros(S, M + PAD, N)
        push!(keep, (As, Bs, Cs))
        push!(tasks, NextLA.GroupedGemmTask('N', 'N', one(S),
            view(As, (1+OFFA):(OFFA+M), :), view(Bs, 1:K, :), zero(S),
            view(Cs, (1+OFFC):(OFFC+M), :)))
    end
    guard = NextLA._grouped_gemm_task_alignment_safe(tasks[1], MODE)
    # `_prepare_...` (underscore) is the raw builder BELOW the guard. The public
    # `prepare_precision_gemm_grouped` would route unsafe members to ordinary
    # GEMMEx and we would be timing the fallback instead of cuBLAS grouped.
    p  = NextLA._prepare_precision_gemm_grouped(tasks, MODE)
    bk = NextLA.get_backend(tasks[1].C)
    submit() = NextLA._with_grouped_host_pointer_mode(bk) do
        # NOTE: bypasses the guard on purpose — we are testing raw cuBLAS.
        NextLA._precision_gemm_grouped_prepared!(p)
    end
    submit(); CUDA.synchronize()
    t = Inf
    for _ in 1:10
        CUDA.synchronize(); t = min(t, CUDA.@elapsed submit())
    end
    @printf("RESULT %-22s offA=%d offC=%d guard_safe=%-5s OK %8.3f ms\n",
            LABEL, OFFA, OFFC, string(guard), t * 1e3)
end

try
    main()
catch e
    msg = sprint(showerror, e)
    kind = occursin("MISALIGNED", uppercase(msg)) ? "FAULT_MISALIGNED" :
           occursin("NOT_SUPPORTED", uppercase(msg)) ? "NOT_SUPPORTED" : "ERROR"
    @printf("RESULT %-22s offA=%d offC=%d guard_safe=?     %s  %s\n",
            LABEL, OFFA, OFFC, kind, replace(msg[1:min(end, 90)], '\n' => ' '))
    exit(1)
end
