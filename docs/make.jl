using Documenter, RegularizedLeastSquares, LinearOperatorCollection, Wavelets

makedocs(
    format = Documenter.HTML(prettyurls = false),
    modules = [RegularizedLeastSquares],
    sitename = "RegularizedLeastSquares.jl",
    authors = "Tobias Knopp, Mirco Grosser, Martin Möddel, Niklas Hackelberg",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Solvers" => "solvers.md",
        "Regularization" => "regularization.md",
        "API" => "API.md"
    ],
    pagesonly = true,
    checkdocs = :none
)

deploydocs(repo   = "github.com/tknopp/RegularizedLeastSquares.jl.git")