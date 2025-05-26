using LinearAlgebra
using BenchmarkTools
using JLD2
using Plots
using StatsPlots

#BLAS.set_num_threads(1) # to make sequential
BLAS.set_num_threads(Threads.nthreads())
t = 40

for T in [Float64, Float32, ComplexF64, ComplexF32]    
    side = 'L'

    for ib in [64,128]
        for trans in ['C', 'N']

            xvals = Float64[]

            y = Float64[] #lapack
            ym = Float64[]

            y0 = Float64[] #julia native
            y0m = Float64[]

            rlvj = Float64[]
            rjvl = Float64[]
            
            for m in [512, 1024, 2048, 4096, 8192, 16384, 32768]
                n = m
                k = m

                push!(xvals, m)

                C = rand(T,m,n)

                if side == 'L'
                    A = rand(T,m,k)
                    lda = m
                    ldw = n
                    work = rand(T,n,n)
                else
                    A = rand(T,n,k)
                    lda = n
                    ldw = m
                    work = rand(T,m,ib)
                end
                
                Tau = rand(T,ib,k)
                Tau1 = rand(T,k)
                A1 = deepcopy(A)
                LinearAlgebra.LAPACK.geqrt!(A, Tau)
                LinearAlgebra.LAPACK.geqrf!(A1, Tau1)
                
                C0 = deepcopy(C)
                C2 = deepcopy(C)
                A2 = deepcopy(A)
                Tau2 = deepcopy(Tau)

                A3 = deepcopy(A1)
                Tau3 = deepcopy(Tau1)
                C3 = deepcopy(C)

                
                if (T == Float64 || T == Float32) && trans == 'C'
                    s = @belapsed unmqr!($T, $side, 'T', $A1, $Tau1, $C) samples = 7 evals = 1
                    sm = @ballocated unmqr!($T, $side, 'T', $A3, $Tau3, $C3) samples = 7 evals = 1
                    push!(y,s)
                    push!(ym, sm / 10^3)
                else
                    s = @belapsed unmqr!($T, $side, $trans, $A1, $Tau1, $C) samples = 7 evals = 1
                    sm = @ballocated unmqr!($T, $side, $trans, $A3, $Tau3, $C3) samples = 7 evals = 1
                    push!(y,s)
                    push!(ym, sm / 10^3)
                end
            
                s0 = @belapsed zunmqrv0($side, $trans, $m, $n, $k, $ib, $A, $lda, $Tau, $ib, $C0, $m, $work, $ldw) samples = 7 evals = 1
                s0m = @ballocated zunmqrv0($side, $trans, $m, $n, $k, $ib, $A2, $lda, $Tau2, $ib, $C2, $m, $work, $ldw) samples = 7 evals = 1
                push!(y0, s0)
                push!(y0m, s0m/10^3)

                push!(rlvj, s/s0)
                push!(rjvl, s0/s)
            end

            """
            #plotting
            
            xvals = Int.(xvals)
            xvals = string.(xvals)

            @save "unmqr type=$T t=$t ib=$ib trans=$trans.jdl2" xvals y ym y0 y0m rlvj rjvl

            p = plot()
            plot!(p, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel="Time (s)", title="unmqr Time Multi-Thread")
            plot!(p, xvals, y, marker=:circle, label="lapack")
            plot!(p, xvals, y0, marker=:star8, label="julia Native")
            savefig(p, "unmqr time type=$T t=$t ib=$ib trans=$trans")

            q = plot()
            plot!(q, legend=:topleft, xlabel="Matrix Size (n x n)", ylabel = "Memory (KB)", title="unmqr Memory Multi-Thread")
            plot!(q, xvals, ym, marker=:circle, label="lapack")
            plot!(q, xvals, y0m, marker=:star8, label="julia Native")
            savefig(q, "unmqr memory type=$T t=$t ib=$ib trans=$trans")
            
            rats = [rlvj; rjvl]
            b1 = minimum(rats) - 0.15
            b2 = maximum(rats) + 0.15

            p1 = groupedbar(xvals, [rlvj rjvl], ylimits=(b1,b2),
                label=["lapack / julia" "julia / lapack"],
                xlabel="Matrix Size (n x n)", ylabel="ratio", title="unmqr time ratio")
            savefig(p1, "unmqr time ratio type=$T t=$t ib=$ib trans=$trans")
            """

        end
    end
end
     
