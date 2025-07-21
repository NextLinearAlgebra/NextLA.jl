using MatrixDepot
using CUDA
using LinearAlgebra
using NPZ
using DelimitedFiles

include("mirs_gesv.jl")

function find_mismatch(name::String, outdir::String; iter::Int = 1000, working_precision::String="R_64F",irs_precision::String="R_16F")
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

    A = convert(Matrix{wp_type}, mdopen(name).A)

    if size(A, 1) != size(A, 2)
        return false, []
    end

    A_gpu = CuArray(A)

    n = size(A, 1)
    
    counter = 0

    mismatch_data = Dict()
    for key in ("B","X_sp","X_mp", "niters_mp", "niters_sp")
        mismatch_data[key] = nothing
    end

    found_one = false

    convergence_results = Vector{Tuple{Bool, Bool, Int, Int, Int, Int}}()


    for i in 1:iter
        println("Computing iteration $i")

        # B_gpu = CuArray(rand(wp_type, n, 1))
        B_gpu = CuArray(randn(wp_type, n, 1)) #move this outside

        X_gpu = similar(B_gpu)

        A_mp = copy(A_gpu);  B_mp = copy(B_gpu); X_mp = copy(X_gpu)
        A_sp = copy(A_gpu);  B_sp = copy(B_gpu);  X_sp = copy(X_gpu)

       
        X_result_mp, info_mp, niters_mp, flag_mp = gesv!(X_mp, A_mp, B_mp, irs_precision=irs_precision)
        X_result_sp, info_sp, niters_sp, flag_sp = gesv!(X_sp, A_sp, B_sp, irs_precision=working_precision)

        dR_sp = B_sp .- A_sp * X_sp
        dR_mp = B_mp .- A_mp * X_mp

        rel_norm_sp = norm(dR_sp)/norm(B_sp)
        rel_norm_mp = norm(dR_mp)/norm(B_mp)

        conv_sp = rel_norm_sp <= tol
        conv_mp = rel_norm_mp <= tol

        push!(convergence_results, (conv_sp, conv_mp, niters_sp, niters_mp, flag_mp, flag_sp))

        if (conv_sp) != (conv_mp)

            println("Found a mismatch")
            found_one = true

            counter += 1

            f = !conv_sp ? "SF_MT" : "ST_MF"

            mismatch_data["B"]    = Array(B_gpu)
            mismatch_data["X_sp"] = Array(X_sp)
            mismatch_data["X_mp"] = Array(X_mp)
            mismatch_data["niters_mp"] = niters_mp
            mismatch_data["niters_sp"] = niters_sp
            
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
        
        # for arr in (A_gpu, B_gpu, X_gpu, A_mp, B_mp, X_mp, A_sp, B_sp, X_sp)
        #     finalize(arr)
        # end

        GC.gc()
        CUDA.reclaim()


            
    end


    return found_one, convergence_results
end


function main()

    #args: "test_32_16" "R_32F" "R_16F" 
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
        found_mismatch, convergence_results = find_mismatch(matrix, outdir;  iter=10, working_precision=ARGS[2], irs_precision=ARGS[3])

        if !found_mismatch
            
            open(nm_path, "a") do f
                write(f, matrix * "\n")
            end
        end

        
        
        mkpath(joinpath(outdir, matrix))

        results_array = reduce(hcat, [[t[i] for t in convergence_results] for i in 1:6])

        writedlm(
            joinpath(outdir, matrix, "convergence_results.csv"),
            [["conv_sp" "conv_mp" "niters_sp" "niters_mp" "flag_mp" "flag_sp"]; results_array],
            ','
        )
        
        println("completed "*matrix)
    end
    
end

main()