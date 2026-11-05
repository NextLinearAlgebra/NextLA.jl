using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
if "--stiles" in ARGS
    Pkg.add(PackageSpec(url="https://github.com/esmail-abdulfattah/sTiles", subdir="julia"))
end
Pkg.precompile()
