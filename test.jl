using MatrixDepot
using CUDA
using LinearAlgebra
using NPZ
using DelimitedFiles
using BenchmarkTools

include("mirs_gesv.jl")

function run_irs!(X, A, B; irs_precision, refinement_solver)
    CUDA.synchronize()
    t = @elapsed begin
        X, info, niters, flag = gesv!(X, A, B, 
            irs_precision=irs_precision, 
            refinement_solver=refinement_solver)
    CUDA.synchronize()
    end
    return X, info, niters, flag, t
end



function find_mismatch(name::String, outdir::String; iter::Int = 1000, working_precision::String="R_64F",irs_precision::String="R_16F", refinement_solver::String="CLASSICAL")
    typemap = Dict(
        # "R_8F" => Float8, #not supported
        "R_16F" => Float16,
        # "R_16BF" => BFloat16,
        "R_32F" => Float32,
        "R_64F" => Float64,
        # "R_128F" => Float128, #not supported

    )

    wp_type = typemap[working_precision]

    tol = sqrt(eps(real(wp_type)))

    A = CuArray(convert(Matrix{wp_type}, mdopen(name).A))

    if size(A, 1) != size(A, 2)
        return false, []
    end

    n = size(A, 1)
    
    counter = 0

    mismatch_data = Dict()
    for key in ("B","X_sp","X_mp", "niters_sp", "niters_mp", "flag_sp", "flag_mp", "time_sp", "time_mp")
        mismatch_data[key] = nothing
    end

    found_one = false

    convergence_results = Vector{Tuple{Bool, Bool, Int, Int, Int, Int, Float64, Float64}}()

    #moved outside to limit memory usage
    X_mp = CUDA.zeros(wp_type, n, 1)
    X_sp = CUDA.zeros(wp_type, n, 1)

    B = zeros(wp_type, n, 1)
    B_gpu = CuArray(B)
    for i in 1:iter
        println("Computing test $i")

        B .= randn(wp_type, n, 1) #move this outside
        copyto!(B_gpu, B)    

        #timing mixed precision
        X_mp, info_mp, niters_mp, flag_mp, t_mp = run_irs!(X_mp, A, B_gpu;
            irs_precision=irs_precision,
            refinement_solver=refinement_solver)

        X_sp, info_sp, niters_sp, flag_sp, t_sp = run_irs!(X_sp, A, B_gpu;
            irs_precision=working_precision,
            refinement_solver=refinement_solver)



        dR_sp = B_gpu .- A * X_sp
        dR_mp = B_gpu .- A * X_mp

        rel_norm_sp = norm(dR_sp)/norm(B)
        rel_norm_mp = norm(dR_mp)/norm(B)

        conv_sp = rel_norm_sp <= tol
        conv_mp = rel_norm_mp <= tol

        push!(convergence_results, (conv_sp, conv_mp, niters_sp, niters_mp, flag_sp, flag_mp, t_sp, t_mp))

        if (conv_sp) != (conv_mp)

            println("Found a mismatch")
            found_one = true

            counter += 1

            f = !conv_sp ? "SF_MT" : "ST_MF"

            mismatch_data["B"]    = Array(B)
            mismatch_data["X_sp"] = Array(X_sp)
            mismatch_data["X_mp"] = Array(X_mp)
            mismatch_data["niters_mp"] = niters_mp
            mismatch_data["niters_sp"] = niters_sp
            mismatch_data["flag_sp"] = flag_sp
            mismatch_data["flag_mp"] = flag_mp
            mismatch_data["time_sp"] = t_sp
            mismatch_data["time_mp"] = t_mp
             
            save_dir = joinpath(outdir, name, f)
            mkpath(save_dir)
            outpath = joinpath(save_dir, "$counter.npz")
            pairs = (Symbol(k) => v for (k,v) in mismatch_data)
            npzwrite(outpath; pairs...)
            println("Wrote to $outpath")

            for key in keys(mismatch_data)
                mismatch_data[key] = nothing
            end

        end
        
        # for arr in (A_gpu, B_gpu, X_gpu, A, B, X_mp, A, B, X_sp)
        #     finalize(arr)
        # end

        GC.gc()
        CUDA.reclaim()


            
    end


    return found_one, convergence_results
end


function main()

    #args: "test_32_16" "R_32F" "R_16F" "CLASSICAL_GMRES"
    outdir = joinpath(@__DIR__, ARGS[1])
    mkpath(outdir)

    nm_path = joinpath(@__DIR__, ARGS[1], "no_mismatches.txt")
    
    if !isfile(nm_path)
        open(nm_path, "w") do f 
        end
    end

    for matrix in ("Bai/af23560", "Engwirda/airfoil_2d", 
        "Simon/appu", "vanHeukelum/cage10", 
        "Oberwolfach/chipcool1", "Averous/epb1", 
        "Botonakis/FEM_3D_thermal1", "Oberwolfach/inlet", 
        "Hollinger/jan99jac040sc", "FEMLAB/ns3Da", 
        "FEMLAB/poisson3Da",
        "Wang/wang3", "Wang/wang4", "Zhao/Zhao1", "Zhao/Zhao2"
        )
        
        #decrease number of iterations
        found_mismatch, convergence_results = find_mismatch(matrix, outdir;  iter=10, working_precision=ARGS[2], irs_precision=ARGS[3], refinement_solver=ARGS[4])

        if !found_mismatch
            
            open(nm_path, "a") do f
                write(f, matrix * "\n")
            end
        end

        
        
        mkpath(joinpath(outdir, matrix))

        results_array = reduce(hcat, [[t[i] for t in convergence_results] for i in 1:8])

        writedlm(
            joinpath(outdir, matrix, "convergence_results.csv"),
            [["conv_sp" "conv_mp" "niters_sp" "niters_mp" "flag_sp" "flag_mp" "time_sp" "time_mp"]; results_array],
            ','
        )
        
        println("completed "*matrix)
    end
    
end

main()