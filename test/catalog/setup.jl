# A separate environment; does not change the repository's Project.toml.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add(PackageSpec(url="https://github.com/esmail-abdulfattah/sTiles", subdir="julia"))
"--plots" in ARGS && Pkg.add("Plots")
Pkg.precompile()
