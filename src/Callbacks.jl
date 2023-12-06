export CompareSolutionCallback
mutable struct CompareSolutionCallback{T, F}
  ref::Vector{T}
  cmp::F
  results::Vector{Float64}
end

CompareSolutionCallback(ref::Vector{T}, cmp = nrmsd) where T = CompareSolutionCallback(ref, cmp, Vector{Float64}())

function (cb::CompareSolutionCallback)(solver::AbstractLinearSolver, _)
  x = solversolution(solver)
  push!(cb.results, cb.cmp(cb.ref, x))
end

export StoreSolutionCallback
mutable struct StoreSolutionCallback{T}
  solutions::Vector{Vector{T}}
end
StoreSolutionCallback(T::Type=Number) = StoreSolutionCallback(Vector{Vector{T}}())
function (cb::StoreSolutionCallback)(solver::AbstractLinearSolver, _)
  x = solversolution(solver)
  push!(cb.solutions, deepcopy(x))
end

export StoreConvergenceCallback
mutable struct StoreConvergenceCallback
  convMeas::Dict{Symbol, Any}
  StoreConvergenceCallback() = new(Dict{Symbol, Any}())
end
function (cb::StoreConvergenceCallback)(solver::AbstractLinearSolver, _)
  meas = solverconvergence(solver)
  for key in keys(meas)
    values = get(cb.convMeas, key, Vector{typeof(meas[key])}())
    push!(values, meas[key])
    cb.convMeas[key] = values
  end
end

export MultipleCallbacks
mutable struct MultipleCallbacks
  callbacks::Vector{Any}
end
MultipleCallbacks(args...) = MultipleCallbacks([args...])
function (cb::MultipleCallbacks)(solver, iteration)
  for callback in cb.callbacks
    callback(solver, iteration)
  end
end