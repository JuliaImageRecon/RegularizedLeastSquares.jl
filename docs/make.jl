using Documenter, RegularizedLeastSquares

makedocs(
    format = :html,
    modules = [RegularizedLeastSquares],
    sitename = "RegularizedLeastSquares.jl",
    authors = "Tobias Knopp",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Operator Interface" => "operators.md",
    ],
)

deploydocs(repo   = "github.com/tknopp/RegularizedLeastSquares.jl.git",
           julia  = "0.7",
           target = "build",
           deps   = nothing,
           make   = nothing)
