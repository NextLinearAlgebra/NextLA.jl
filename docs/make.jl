using NextLA
using NextLA.MixedPrecision
using Documenter

DocMeta.setdocmeta!(NextLA, :DocTestSetup, :(using NextLA); recursive=true)

# Edit links and source links need the git remote. When the docs are built from a
# checkout without .git (e.g. an unpacked archive), turn them off rather than fail.
const IN_GIT = isdir(joinpath(@__DIR__, "..", ".git"))
const REMOTE_KW = IN_GIT ? (; repo=Remotes.GitHub("NextLinearAlgebra", "NextLA.jl")) : (; remotes=nothing)

# Some pages live at the repository root so they also read on GitHub. Copy them into
# the site at build time (the copies are git-ignored), adapting what only works on GitHub:
#   - links between root files point to the matching site page, links into docs/src
#     become relative site links, and files that are not part of the site
#     (CITATION.cff, LICENSE) point to GitHub;
#   - Julia's Markdown parser has no task lists, so checkboxes become ✅ and ⬜;
#   - GitHub's "> [!NOTE]" box becomes a Documenter note admonition;
#   - "Edit on GitHub" opens the root file, not the generated copy.
# README.md becomes the Home page (index.md): see home_page below.
const ROOT_PAGES = [
    "README.md" => "index.md",
    "CITATION.md" => "citing.md",
    "CONTRIBUTING.md" => "dev/contributing.md",
    "test/README.md" => "dev/testing.md",
]
const GITHUB_BLOB = "https://github.com/NextLinearAlgebra/NextLA.jl/blob/main/"
github_note(block) = "!!! note\n" *
    join(("    " * replace(l, r"^> ?" => "") for l in split(chomp(block), "\n")[2:end]), "\n") * "\n"
# Headings get an anchor prefixed with the page name (home-installation, ...), so a
# root page's "Installation" or "Mixed precision" never clashes with a manual heading.
# Lines inside code fences are left alone; a fence closes only on a backtick run at least
# as long as the one that opened it, so ```` blocks may contain ``` lines.
function prefixed_headings(text, prefix)
    fence = ""
    lines = map(split(text, "\n")) do l
        f = match(r"^(`{3,})(.*)$", l)
        if f !== nothing
            if isempty(fence)
                fence = f.captures[1]
            elseif length(f.captures[1]) >= length(fence) && isempty(strip(f.captures[2]))
                fence = ""
            end
            return l
        end
        m = match(r"^(#{1,6}) (.+)$", l)
        (!isempty(fence) || m === nothing) && return l
        slug = strip(replace(lowercase(m.captures[2]), r"[^a-z0-9]+" => "-"), '-')
        return "$(m.captures[1]) [$(m.captures[2])](@id $prefix-$slug)"
    end
    return join(lines, "\n")
end
# The README speaks to someone on GitHub; on the Home page leave out what only makes
# sense there (the badges, the docs link, how to build the docs and run the tests).
const README_ONLY_SECTIONS = ["Documentation", "Running the tests"]
function home_page(text)
    text = replace(text, r"^\[!\[.*\n"m => "")
    text = replace(text, r"\n{3,}" => "\n\n")
    text = replace(text, r"^See the \[documentation\]\(https://nextlinearalgebra\.github\.io/[^)]*\).*\n"m =>
        "Continue with [Getting started](@ref), or go to the [API reference](@ref \"Matrix multiply (GEMM)\").\n")
    for section in README_ONLY_SECTIONS
        text = replace(text, Regex("^## \\Q$section\\E\\n(?:(?!## ).*\\n)*", "m") => "")
    end
    return rstrip(text) * "\n"
end
for (from, to) in ROOT_PAGES
    text = read(joinpath(@__DIR__, "..", from), String)
    to == "index.md" && (text = home_page(text))
    text = prefixed_headings(text, to == "index.md" ? "home" : splitext(basename(to))[1])
    text = replace(text, r"^> \[!NOTE\]\n(?:>.*(?:\n|$))+"m => github_note)
    up = "../"^count(==('/'), to)   # from the page's folder back to docs/src
    text = replace(text,
        r"^- \[x\] "m => "- ✅ ",
        r"^- \[ \] "m => "- ⬜ ",
        "](docs/src/" => "](" * up,
        "](README.md)" => "](" * up * "index.md)",
        "](CITATION.md)" => "](" * up * "citing.md)",
        "](CONTRIBUTING.md)" => "](" * up * "dev/contributing.md)",
        "](test/README.md)" => "](" * up * "dev/testing.md)",
        "](" * GITHUB_BLOB * "test/README.md)" => "](" * up * "dev/testing.md)",
        "](CITATION.cff)" => "](" * GITHUB_BLOB * "CITATION.cff)",
        "](LICENSE)" => "](" * GITHUB_BLOB * "LICENSE)",
    )
    dest = joinpath(@__DIR__, "src", to)
    edit = relpath(joinpath(@__DIR__, "..", from), dirname(dest))
    meta = (to == "index.md" ? "CurrentModule = NextLA\n" : "") * "EditURL = \"$edit\"\n"
    write(dest, "```@meta\n" * meta * "```\n\n" * text)
end

# Local switches (CI sets neither): NEXTLA_DOCS_BUILDDIR builds into another folder, so a
# served docs/build stays complete until the new build is copied over it; NEXTLA_DOCS_DOCTEST=false
# skips the doctests for a quick preview.
makedocs(;
    build=get(ENV, "NEXTLA_DOCS_BUILDDIR", "build"),
    doctest=get(ENV, "NEXTLA_DOCS_DOCTEST", "true") != "false",
    modules=[NextLA, NextLA.MixedPrecision],
    authors="Rabab Alomairy, Evelyne Ringoot, Sophie Xuan, Vicki Carrica, Maxwell Onyango, Julian Samaroo, Felipe Tome and contributors",
    sitename="NextLA.jl",
    REMOTE_KW...,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nextlinearalgebra.github.io/NextLA.jl",
        repolink="https://github.com/NextLinearAlgebra/NextLA.jl",
        edit_link="main",
        assets=["assets/bibtex.css"],
        size_threshold=300 * 1024,
        size_threshold_warn=200 * 1024,
    ),
    # Every exported, documented name must appear somewhere in the manual.
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Manual" => [
            "Backends and GPUs" => "manual/backends.md",
            "Mixed precision" => "manual/mixed_precision.md",
            "Tile low-rank" => "manual/tile_low_rank.md",
            "Benchmarks" => "manual/benchmarks.md",
        ],
        # Mirrors src/: dense/{core,blas,lapack}, mxp/{core,blas,lapack}, tlr/{core,blas}.
        "API reference" => [
            "Dense" => [
                "Core" => "api/dense/core.md",
                "BLAS" => [
                    "GEMM" => "api/dense/blas/gemm.md",
                    "TRSM and TRMM" => "api/dense/blas/trsm.md",
                    "SYRK" => "api/dense/blas/syrk.md",
                    "AXPY and GERC" => "api/dense/blas/axpy_gerc.md",
                ],
                "LAPACK" => [
                    "Cholesky" => "api/dense/lapack/cholesky.md",
                    "LU" => "api/dense/lapack/lu.md",
                    "QR" => "api/dense/lapack/qr.md",
                    "Rotations" => "api/dense/lapack/rotations.md",
                ],
            ],
            "Mixed precision" => [
                "Core" => "api/mxp/core.md",
                "BLAS" => "api/mxp/blas.md",
                "LAPACK" => "api/mxp/lapack.md",
            ],
            "Tile low-rank" => [
                "Core" => "api/tlr/core.md",
                "BLAS" => "api/tlr/blas.md",
            ],
        ],
        "Developer documentation" => [
            "Contributing" => "dev/contributing.md",
            "Running the tests" => "dev/testing.md",
            "NextLA TODO" => "dev/todo.md",
        ],
        "Citing NextLA" => "citing.md",
    ],
)

deploydocs(;
    repo="github.com/NextLinearAlgebra/NextLA.jl",
    devbranch="main",
    push_preview=true,
)
