using Documenter, RegularizedLeastSquares, LinearOperatorCollection, Wavelets

makedocs(
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl",
        assets=String[],
    ),
    repo="https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl/blob/{commit}{path}#{line}",
    modules = [RegularizedLeastSquares],
    sitename = "RegularizedLeastSquares.jl",
    authors = "Tobias Knopp, Mirco Grosser, Martin Möddel, Niklas Hackelberg, Andrew Mao, Jakob Assländer",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Solvers" => "solvers.md",
        "Regularization" => "regularization.md",
        "API" => Any["Solvers" => "API/solvers.md",
        "Regularization Terms" => "API/regularization.md"],

    ],
    pagesonly = true,
    checkdocs = :none,
    doctest   = true
)

deploydocs(repo   = "github.com/JuliaImageRecon/RegularizedLeastSquares.jl.git", push_preview = true)