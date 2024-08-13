# # Multi-Threading
# There are three different kinds of multi-threading in RegularizedLeastSquares.jl, two of which are transparent to the solvers themselves.
# In the following examples we will use the Threads.@threads macro for multi-threading, but the concepts are applicable to other multi-threading options as well.

# ## Solver Based Multi-Threading
# This type of multi-threading is transparent to the solver and is applicable if the total solution is composed of individual solutions that can be solved in parallel.
# In particular, this approach also allows for using solvers with different parameters, such as their operator or regularization parameters.

using RegularizedLeastSquares
As = [rand(32, 16) for _ in 1:4]
xs = [rand(16) for _ in 1:4]
bs = [A*x for (A, x) in zip(As, xs)]

xs_approx = similar(xs)
Threads.@threads for i in 1:4
  solver = createLinearSolver(CGNR, As[i]; iterations=32)
  xs_approx[i] = solve!(solver, bs[i])
end

# ## Operator Based Multi-Threading
# This type of multi-threading involves linear operators or proximal maps that can be implemnted in parallel.
# Examples of this include the proximal map of the TV regularization term, which is based on the multi-threaded GradientOp from LinearOperatorCollection.
# GPU acceleration also falls under this approach.

# ## Measurement Based Multi-Threading
# This level of multi-threading applies the same solver (and its parameters) to multiple measurement vectors or rather a measurement matrix B.
# This is useful in the case of multiple measurements that can be solved in parallel and can reuse the same solver. This approach is not applicable if the operator is stateful.

# To use this approach we first build a measurement matrix B and a corresponding solver:
A = first(As)
B = mapreduce(x -> A*x, hcat, xs)
solver = createLinearSolver(CGNR, A; iterations=32)

# We can then simply pass the measurement matrix to the solver. The result will be the same as if we passed each colument of B seperately:
x_approx = solve!(solver, B)
size(x_approx)

# The previous `solve!` call was still executed sequentially. To execute it in parallel, we have to pass a MultiThreadingState to the `solve!` call:
x_multi = solve!(solver, B; scheduler = MultiThreadingState)
x_approx == x_multi

# It is possible to implement custom scheduling. The following pseudo-code shows how to implement this for the FLoop.jl package:

# Since most solver have conv. criteria, they can finish at different iteration numbers, which we track with the active flags.
mutable struct FloopState{S, ST <: AbstractSolverState{S}} <: AbstractMatrixSolverState{S}
  states::Vector{ST}
  active::Vector{Bool}
  FloopState(states::Vector{ST}) where {S, ST <: AbstractSolverState{S}} = new{S, ST}(states, fill(true, length(states)))
end

# To hook into the existing init! code we only have to supply a method that gets a copyable "vector" state. This will invoke our FloopState constructor with copies of the given state.
prepareMultiStates(solver::AbstractLinearSolver, state::FloopState, b::AbstractMatrix) = prepareMultiStates(solver, first(state.states), b)

# We specialise the iterate function which is called with the idx of still active states
#function iterate(solver::AbstractLinearSolver, state::FloopState, activeIdx)
#  @floop for i in activeIdx
#    res = iterate(solver, state.states[i])
#    if isnothing(res)
#      state.active[i] = false
#    end
#  end
#  return state.active, state
#end

# solver = createLinearSolver(CGNR, A, ...)
# solve!(solver, B; scheduler = FloopState)