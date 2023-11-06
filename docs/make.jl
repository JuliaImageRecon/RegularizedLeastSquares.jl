using Documenter, RegularizedLeastSquares

makedocs(
    modules = [RegularizedLeastSquares],
    sitename = "RegularizedLeastSquares.jl",
    authors = "Tobias Knopp, Mirco Grosser, Martin Möddel, Niklas Hackelberg",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Solvers" => "solvers.md",
        #"Matrices & Operators" => "operators.md",
        "Regularization" => "regularization.md",
        #"API" => "API.md"
    ],
    pagesonly = true,
    checkdocs = :export
)

deploydocs(repo   = "github.com/tknopp/RegularizedLeastSquares.jl.git")