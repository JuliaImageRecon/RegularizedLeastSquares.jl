using Documenter, Literate, RegularizedLeastSquares

# Generate examples
OUTPUT_BASE = joinpath(@__DIR__(), "src/generated")
INPUT_BASE = joinpath(@__DIR__(), "src/literate")
for (_, dirs, _) in walkdir(INPUT_BASE)
    for dir in dirs
        OUTPUT = joinpath(OUTPUT_BASE, dir)
        INPUT = joinpath(INPUT_BASE, dir)
        for file in filter(f -> endswith(f, ".jl"), readdir(INPUT))
            Literate.markdown(joinpath(INPUT, file), OUTPUT)
        end
    end
end

makedocs(
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl",
        assets=String[],
        collapselevel=1,
    ),
    repo="https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl/blob/{commit}{path}#{line}",
    modules = [RegularizedLeastSquares],
    sitename = "RegularizedLeastSquares.jl",
    authors = "Tobias Knopp, Mirco Grosser, Martin Möddel, Niklas Hackelberg, Andrew Mao, Jakob Assländer",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "generated/examples/getting_started.md",
        "Examples" => Any["Compressed Sensing" => "generated/examples/compressed_sensing.md", "Computed Tomography" => "generated/examples/computed_tomography.md"],
        "Solvers" => "solvers.md",
        "Regularization" => "generated/explanations/regularization.md",
        "How to" => Any[
            "Weighting" => "generated/howto/weighting.md",
            "Normal Operator" => "generated/howto/normal_operator.md",
            "Multi-Threading" => "generated/howto/multi_threading.md",
            "GPU Acceleration" => "generated/howto/gpu_acceleration.md",
            "Efficient Kaczmarz" => "generated/howto/efficient_kaczmarz.md",
            "Callbacks" => "generated/howto/callbacks.md",
            "Plug-and-Play Regularization" => "generated/howto/plug-and-play.md"
        ],
        "API Reference" => Any["Solvers" => "API/solvers.md",
        "Regularization Terms" => "API/regularization.md"],

    ],
    pagesonly = true,
    checkdocs = :none,
    doctest   = false,
    doctestfilters = [r"(\d*)\.(\d{4})\d+"]
    )

deploydocs(repo   = "github.com/JuliaImageRecon/RegularizedLeastSquares.jl.git", push_preview = true)