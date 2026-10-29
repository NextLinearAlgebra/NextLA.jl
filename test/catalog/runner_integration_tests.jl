# Exercises the actual supervisor/checkpoint/resume path using a tiny test worker.
# It intentionally does not load CUDA or sTiles.
using Test, TOML
include("run.jl")

@testset "Sweep completion, timeout rows, resume and input fingerprints" begin
    mktempdir() do dir
        data = joinpath(dir,"data")
        mkpath(data)
        for name in ("bcsstk15", "nd3k")
            write(joinpath(data,name*".mtx"), "test worker fixture\n")
        end
        worker = joinpath(dir,"testworker.jl")
        common = repr(joinpath(@__DIR__,"common.jl"))
        script = "include($common)\nusing .CatalogSweep, TOML\n" * raw"""
        config = TOML.parsefile(ARGS[1])
        name, backend, csv = ARGS[2:4]
        if backend == "stiles"
            println("native timeout fixture"); flush(stdout)
            if Sys.iswindows()
                ccall(:Sleep, stdcall, Cvoid, (UInt32,), 120000)
            else
                ccall(:sleep, Cuint, (Cuint,), 120)
            end
        end
        entry = only(filter(r -> r["name"] == name, CatalogSweep.read_csv(config["manifest"])))
        for method in split(ARGS[5], ',')
            row = CatalogSweep.base_row(entry,config,method)
            row["status"] = "OK"
            CatalogSweep.append_row(csv,row)
        end
        """
        write(worker,script)
        out = joinpath(dir,"output")
        withenv("MATRIX_DIR"=>data, "RESULTS_DIR"=>out, "REAL_MATRICES"=>"bcsstk15,nd3k",
                "METHODS"=>"all", "GPU_METHODS"=>"Pure_F32,M1", "GPU_TIMEOUT_SECONDS"=>"20",
                "STILES_TIMEOUT_SECONDS"=>"3", "SWEEP_TIMEOUT_SECONDS"=>"90") do
            main(String[]; worker=worker)
            rows = CatalogSweep.read_csv(joinpath(out,"results.csv"))
            @test length(rows) == 6
            @test count(r -> r["status"]=="OK",rows) == 4
            @test count(r -> r["status"]=="TIMEOUT",rows) == 2
            summary = read(joinpath(out,"cases-summary.csv"),String)
            main(String[]; worker=worker)
            @test length(CatalogSweep.read_csv(joinpath(out,"results.csv"))) == 6
            @test read(joinpath(out,"cases-summary.csv"),String) == summary
            write(joinpath(data,"bcsstk15.mtx"),"changed input")
            @test_throws ErrorException main(String[]; worker=worker)
        end
        withenv("MATRIX_DIR"=>data, "RESULTS_DIR"=>joinpath(dir,"budget"), "REAL_MATRICES"=>"bcsstk15",
                "METHODS"=>"gpu", "GPU_METHODS"=>"M1", "GPU_TIMEOUT_SECONDS"=>"20",
                "SWEEP_TIMEOUT_SECONDS"=>"0.25") do
            main(String[]; worker=worker)
            @test isempty(CatalogSweep.read_csv(joinpath(dir,"budget","results.csv")))
            summary = CatalogSweep.read_csv(joinpath(dir,"budget","cases-summary.csv"))
            @test only(summary)["status"] == "SWEEP_LIMIT"
        end
    end
end
