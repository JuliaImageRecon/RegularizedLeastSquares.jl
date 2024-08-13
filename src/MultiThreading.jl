export SequentialState, MultiThreadingState, prepareMatrixStates
abstract type AbstractMatrixSolverState{S} <: AbstractSolverState{S} end
mutable struct SequentialState{S, ST <: AbstractSolverState{S}} <: AbstractMatrixSolverState{S}
  states::Vector{ST}
  active::Vector{Bool}
  SequentialState(states::Vector{ST}) where {S, ST <: AbstractSolverState{S}} = new{S, ST}(states, fill(true, length(states)))
end

mutable struct MultiThreadingState{S, ST <: AbstractSolverState{S}} <: AbstractMatrixSolverState{S}
  states::Vector{ST}
  active::Vector{Bool}
  MultiThreadingState(states::Vector{ST}) where {S, ST <: AbstractSolverState{S}} = new{S, ST}(states, fill(true, length(states)))
end

function init!(solver::AbstractLinearSolver, state::AbstractSolverState, b::AbstractMatrix; scheduler = SequentialState, kwargs...)
  states = prepareMatrixStates(solver, state, b)
  multiState = scheduler(states)
  solver.state = multiState
  init!(solver, multiState, b; kwargs...)
end
function init!(solver::AbstractLinearSolver, state::AbstractMatrixSolverState, b::AbstractVector; kwargs...)
  singleState = first(state.states)
  solver.state = singleState
  init!(solver, singleState, b; kwargs...)
end

function prepareMatrixStates(solver::AbstractLinearSolver, state::AbstractSolverState, b::AbstractMatrix)
  states = [deepcopy(state) for _ in 1:size(b, 2)]
  return states
end
prepareMatrixStates(solver::AbstractLinearSolver, state::Union{SequentialState, MultiThreadingState}, b::AbstractMatrix) = prepareMatrixStates(solver, first(state.states), b)

function init!(solver::AbstractLinearSolver, state::AbstractMatrixSolverState, b::AbstractMatrix; kwargs...)
  for (i, s) in enumerate(state.states)
    init!(solver, s, b[:, i]; kwargs...)
  end
  state.active .= true
end

function iterate(solver::S, state::AbstractMatrixSolverState) where {S <: AbstractLinearSolver}
  activeIdx = findall(state.active)
  if isempty(activeIdx)
    return nothing
  end
  return iterate(solver, state, activeIdx)
end

function iterate(solver::AbstractLinearSolver, state::SequentialState, activeIdx)
  for i in activeIdx
    res = iterate(solver, state.states[i])
    if isnothing(res)
      state.active[i] = false
    end
  end
  return state.active, state
end

function iterate(solver::AbstractLinearSolver, state::MultiThreadingState, activeIdx)
  Threads.@threads for i in activeIdx
    res = iterate(solver, state.states[i])
    if isnothing(res)
      state.active[i] = false
    end
  end
  return state.active, state
end

solversolution(state::AbstractMatrixSolverState) = mapreduce(solversolution, hcat, state.states)