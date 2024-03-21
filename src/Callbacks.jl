using ProgressMeter


export CompareSolutionCallback
mutable struct CompareSolutionCallback{T, F}
  ref::Vector{T}
  cmp::F
  results::Vector{Float64}
end

"""
    CompareSolutionCallback(ref, cmp)

Callback that compares the solvers current `solution` with the given reference via `cmp(ref, solution)` per iteration. Results are stored in the `results` field.
"""
CompareSolutionCallback(ref::Vector{T}, cmp = nrmsd) where T = CompareSolutionCallback(ref, cmp, Vector{Float64}())

function (cb::CompareSolutionCallback)(solver::AbstractLinearSolver, _)
  x = solversolution(solver)
  push!(cb.results, cb.cmp(cb.ref, x))
end

export StoreSolutionCallback
mutable struct StoreSolutionCallback{T}
  solutions::Vector{Vector{T}}
end
"""
    StoreSolutionCallback(T)

Callback that accumlates the solvers `solution` per iteration. Results are stored in the `solutions` field.
"""
StoreSolutionCallback(T::Type=Number) = StoreSolutionCallback(Vector{Vector{T}}())
function (cb::StoreSolutionCallback)(solver::AbstractLinearSolver, _)
  x = solversolution(solver)
  push!(cb.solutions, deepcopy(x))
end

export StoreConvergenceCallback
mutable struct StoreConvergenceCallback
  convMeas::Dict{Symbol, Any}
end
"""
    StoreConvergenceCallback()

Callback that accumlates the solvers convergence metrics per iteration. Results are stored in the `convMeas` field.
"""
StoreConvergenceCallback() = new(Dict{Symbol, Any}())
function (cb::StoreConvergenceCallback)(solver::AbstractLinearSolver, _)
  meas = solverconvergence(solver)
  for key in keys(meas)
    values = get(cb.convMeas, key, Vector{typeof(meas[key])}())
    push!(values, meas[key])
    cb.convMeas[key] = values
  end
end


export ProgressBarCallback
"""
    ProgressBarCallback()

Callback that displays a progress bar for a solver.
"""
Base.@kwdef mutable struct ProgressBarCallback
  meter::Union{Progress,Nothing} = nothing
end
ProgressBarCallback(solver::AbstractLinearSolver) = ProgressBarCallback(Progress(solver.iterations))
ProgressBarCallback(iterations::Int) = ProgressBarCallback(Progress(iterations))

"""
  (self::ProgressBarCallback)(solver::AbstractLinearSolver, iter_n::Int)

Initializes the callback when `iter_n` is zero, then updates the progress bar.
"""
function (self::ProgressBarCallback)(solver::AbstractLinearSolver, iter_n::Int)
  if iter_n != 0
    next!(self.meter)
  end

  # lazy init for iter_n = 0
  if iter_n == 0 && isnothing(self.meter)
    self.meter = Progress(solver.iterations)
  end
end
