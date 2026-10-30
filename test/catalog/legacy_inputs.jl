# Locate the old downloads automatically; fetch only missing inputs to scratch.
include("common.jl")
using .CatalogSweep, Downloads, Dates
function header_ok(path,n)
    open(path) do io
        startswith(lowercase(readline(io)),"%%matrixmarket matrix coordinate") || return false
        for line in eachline(io)
            s=strip(line);(isempty(s)||startswith(s,'%')) && continue
            fields=split(s)
            return length(fields)==3 && tryparse(Int,fields[1])==n && tryparse(Int,fields[2])==n
        end
        false
    end
end
function prepare_legacy_inputs()
    destination=abspath(expanduser(get(ENV,"LEGACY_MATRIX_DIR",joinpath(homedir(),"orcd","scratch","nextla-legacy-matrices"))))
    mkpath(destination)
    roots=unique([joinpath(homedir(),"NextLA.jl"),joinpath(homedir(),"NextLA-catalog-run"),joinpath(homedir(),"orcd","scratch","nextla-matrices"),destination,homedir()])
    candidates=Dict{String,Vector{String}}()
    for root in roots
        isdir(root) || continue
        for (dir,dirs,files) in walkdir(root;onerror=e->println(stderr,"Search skipped: ",e))
            filter!(d->!startswith(d,'.') && !(d in ("node_modules","benchmark-results","env","venv")),dirs)
            length(splitpath(relpath(dir,root)))>=4 && empty!(dirs)
            for file in files
                endswith(lowercase(file),".mtx") || continue
                push!(get!(candidates,lowercase(file),String[]),joinpath(dir,file))
            end
        end
    end
    entries=CatalogSweep.read_csv(joinpath(@__DIR__,"legacy-matrices.csv"))
    status=Dict{String,Any}[]
    for entry in entries
        name=entry["name"];n=parse(Int,entry["n"]);target=joinpath(destination,name*".mtx")
        aliases=name=="inla_graph_animal2" ? [name*".mtx","animal2.mtx"] : [name*".mtx"]
        found=nothing
        for alias in aliases,path in get(candidates,alias,String[])
            try
                header_ok(path,n) && (found=path;break)
            catch e
                println(stderr,"Cannot use $path: ",sprint(showerror,e))
            end
        end
        row=Dict{String,Any}("matrix"=>name,"target"=>target,"url"=>entry["url"],"timestamp"=>string(now()))
        if found!==nothing
            if abspath(found)!=target && !ispath(target)
                symlink(found,target)
            end
            row["status"]="FOUND";row["source_path"]=found
            println("FOUND $name: $found")
        else
            temporary=target*".part"
            try
                println("DOWNLOAD $name from ",entry["url"]);flush(stdout)
                if entry["archive"]=="true"
                    archive=target*".tar.gz"
                    Downloads.download(entry["url"],archive;timeout=300)
                    open(temporary,"w") do io
                        run(pipeline(`tar -xOf $archive $name/$name.mtx`;stdout=io))
                    end
                else
                    Downloads.download(entry["url"],temporary;timeout=parse(Float64,get(ENV,"LEGACY_DOWNLOAD_TIMEOUT_SECONDS","600")))
                end
                header_ok(temporary,n) || error("Downloaded input has wrong header or dimensions")
                # Preserve invalid pre-existing targets for diagnosis.
                ispath(target) && error("Target already exists but was not accepted; inspect $target")
                mv(temporary,target)
                row["status"]="DOWNLOADED"
            catch e
                row["status"]="UNAVAILABLE";row["detail"]=sprint(showerror,e)
                println("UNAVAILABLE $name: ",row["detail"])
            end
        end
        push!(status,row)
        CatalogSweep.write_csv(joinpath(destination,"legacy-input-status.csv"),status;
            columns=["matrix","status","source_path","target","url","timestamp","detail"])
    end
    println("Legacy matrix directory: ",destination)
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && prepare_legacy_inputs()
