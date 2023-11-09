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
        "API" => Any["Solvers" => "API/solvers.md",
        "Regularization Terms" => "API/regularization.md"],

    ],
    pagesonly = true,
    checkdocs = :none
)

deploydocs(repo   = "github.com/JuliaImageRecon/RegularizedLeastSquares.jl.git")
