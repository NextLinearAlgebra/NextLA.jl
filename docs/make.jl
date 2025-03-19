using NextLA
using Documenter

DocMeta.setdocmeta!(NextLA, :DocTestSetup, :(using NextLA); recursive=true)

makedocs(;
    modules=[NextLA],
    authors="Rabab Alomairy",
    sitename="NextLA.jl",
    format=Documenter.HTML(;
        canonical="https://rabab53.github.io/NextLA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rabab53/NextLA.jl",
    devbranch="main",
)
