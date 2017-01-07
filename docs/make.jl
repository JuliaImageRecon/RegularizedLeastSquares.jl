using Documenter, LinearSolver

makedocs(
    format = :html,
    modules = [LinearSolver],
    sitename = "LinearSolver.jl",
    authors = "Tobias Knopp",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingStarted.md",
        "Operator Interface" => "operators.md",
    ],
)

deploydocs(repo   = "github.com/tknopp/LinearSolver.jl.git",
           julia  = "0.5",
           deps   = nothing,
           make   = nothing)

