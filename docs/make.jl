using Documenter, RegularizedLeastSquares

makedocs(
    format = :html,
    modules = [RegularizedLeastSquares],
    sitename = "RegularizedLeastSquares.jl",
    authors = "Tobias Knopp, Mirco Grosser, Martin Möddel",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Matrices & Operators" => "operators.md",
        "Regularization" => "regularization.md",
        "Solvers" => "solvers.md",
        "API" => "API.md",
    ],
    html_prettyurls = false, #!("local" in ARGS),
)

deploydocs(repo   = "github.com/tknopp/RegularizedLeastSquares.jl.git",
           target = "build")
