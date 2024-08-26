# # Multi-Threading
# There are different ways multi-threading can be used with RegularizedLeastSquares.jl. To use multi-threading in Julia, one needs to start their session with multi-threads, see the [Julia documentation](https://docs.julialang.org/en/v1/manual/multi-threading/) for more information.

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
# GPU acceleration also falls under this approach, see [GPU Acceleration](gpu_acceleration.md) for more information.

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

# The previous `solve!` call was still executed sequentially. To execute it in parallel, we have to specify a multi-threaded `scheduler` as a keyword-argument of the `solve!` call.
# RegularizedLeastSquares.jl provides a `MultiThreadingState` scheduler that can be used for this purpose. This scheduler is based on the `Threads.@threads` macro:
x_multi = solve!(solver, B; scheduler = MultiThreadingState)
x_approx == x_multi

# ## Custom Scheduling
# It is possible to implement custom scheduling. The following code shows how to implement this for the `Threads.@spawn` macro. Usually one this to implement multi-threading with a package such as FLoop.jl or ThreadPools.jl for thread pinning:

# Since most solver have conv. criteria, they can finish at different iteration numbers, which we track this information with flags.
 mutable struct SpawnState{S, ST <: AbstractSolverState{S}} <: RegularizedLeastSquares.AbstractMatrixSolverState{S}
   states::Vector{ST}
   active::Vector{Bool}
   SpawnState(states::Vector{ST}) where {S, ST <: AbstractSolverState{S}} = new{S, ST}(states, fill(true, length(states)))
 end

# To hook into the existing init! code we only have to supply a method that gets a copyable "vector" state. This will invoke our SpawnState constructor with copies of the given state.
 prepareMultiStates(solver::AbstractLinearSolver, state::SpawnState, b::AbstractMatrix) = prepareMultiStates(solver, first(state.states), b);

# We specialise the iterate function which is called with the idx of still active states
function Base.iterate(solver::AbstractLinearSolver, state::SpawnState, activeIdx)
  @sync Threads.@spawn for i in activeIdx
    res = iterate(solver, state.states[i])
    if isnothing(res)
      state.active[i] = false
    end
  end
  return state.active, state
end

# Now we can simply use the SpawnState scheduler in the solve! call:
x_custom = solve!(solver, B; scheduler = SpawnState)
x_approx == x_multi