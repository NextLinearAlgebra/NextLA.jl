using NextLA
using Documenter

DocMeta.setdocmeta!(NextLA, :DocTestSetup, :(using NextLA); recursive=true)

makedocs(;
    modules=[NextLA],
    authors="Rabab Alomairy",
    sitename="NextLA.jl",
    format=Documenter.HTML(;
        canonical="https://nextlinearalgebra.github.io/NextLA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Mixed precision" => "mixed_precision.md",
    ],
)

deploydocs(;
    repo="github.com/NextLinearAlgebra/NextLA.jl",
    devbranch="main",
)
