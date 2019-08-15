export fista

mutable struct FISTA <: AbstractLinearSolver
  A
  regularizer::Regularization
  params
end

"""
    FISTA(A; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates a `FISTA` object for the system matrix `A`.

# Arguments
* `A` - system matrix
* (`reg=nothing`)     - Regularization object
* (`regName=["L1"]`)  - name of the Regularization to use (if reg==nothing)
* (`λ=[0.0]`)         - Regularization paramter
"""
function FISTA(A; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  return FISTA(A,vec(reg)[1],kargs)
end

"""
    solve(solver::FISTA, b::Vector)

solves an inverse problem using FISTA.

# Arguments
* `solver::FISTA`  - the solver containing both system matrix and regularizer
* `b::Vector`     - data vector
"""
function solve(solver::FISTA, b::Vector)
  return fista(solver.A, b, solver.regularizer; solver.params...)
end


"""
    fista(A,b::Vector{T}, reg::Regularization; kargs...) where T

This funtion implements the fista algorithm.
Solve the problem: X = arg min_x 1/2*|| Ax-b||² + λ*g(X) where:
   x: variable (vector)
   b: measured data
   A: a general linear operator
   g(X): a convex but not necessarily a smooth function

# Arguments
* `A`                       - system matrix
* `b::Vector{T}`            - data vector (right-hand side)
* `reg::Regularization`     - regularization object
* (`startVector=nothing`)   - start vector
* (`iterations::Int64=50`)  - maximum number of iterations
* (`ρ::Float64=1.0`)        - step size for gradient step
* (`t::Float64=1.0`)        - step size for predictor-corrector step
* (`relTol::Float64=1.e-4`) - relative tolerance for stopping criterion
* (`solverInfo = nothing`)  - `solverInfo` object used to store convergence metrics
"""
function fista(A, b::Vector{T}, reg::Regularization
                ; startVector=nothing
                , iterations::Int64=50
                , ρ::Float64=1.0
                , t::Float64=1.0
                , relTol::Float64=1.e-4
                , solverInfo = nothing
                , kargs...) where T

  if startVector == nothing
    x = A' * b
  else
    x = startVector
  end
  res = A*x-b

  xᵒˡᵈ = copy(x)

  solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg],residual=res)

  costFunc = 0.5*norm(res)^2+reg.norm(x,reg.λ)

  @showprogress 1 "Computing..." for l=1:iterations
    xᵒˡᵈ[:] = x[:]

    x[:] = x[:] - ρ* (A' * res)

    reg.prox!(x, ρ*reg.λ; reg.params...)

    tᵒˡᵈ = t

    t = (1. + sqrt(1. + 4. * tᵒˡᵈ^2)) / 2.
    x[:] = x + (tᵒˡᵈ-1)/t*(x-xᵒˡᵈ)

    res = A*x-b
    regNorm = reg.norm(x,reg.λ)

    solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg],residual=res)

    # exit if objective functional changes by less then ɛ
    costFuncOld = costFunc
    costFunc = 0.5*norm(res)^2+regNorm
    abs(costFunc-costFuncOld)/costFuncOld < relTol && return x
  end

  return x
end

# alternative implementation allowing for an optimized AHA
# does not contain a stopping condition
# function fista2(A, b::Vector{T}, reg::Regularization
#                 ; AHA=nothing
#                 , startVector=nothing
#                 , iterations::Int64=50
#                 , ρ::Float64=1.0
#                 , t::Float64=1.0
#                 , solverInfo = nothing
#                 , kargs...) where T
#
#   if startVector == nothing
#     x = A' * b
#   else
#     x = startVector
#   end
#
#   # if AHA!=nothing
#     op = AHA
#   # else
#   #   op = A'*A
#   # end
#
#   β = A'*b
#
#   xᵒˡᵈ = copy(x)
#
#   solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])
#
#   costFunc = 0.5*norm(res)^2+norm(reg,x)
#
#   for l=1:iterations
#     xᵒˡᵈ[:] = x[:]
#
#     x[:] = x[:] - ρ*(op*x-β)
#
#     reg.prox!(x, ρ*reg.λ)
#
#     tᵒˡᵈ = t
#
#     t = (1. + sqrt(1. + 4. * tᵒˡᵈ^2)) / 2.
#     x[:] = x + (tᵒˡᵈ-1)/t*(x-xᵒˡᵈ)
#
#     solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])
#   end
#
#   return x
# end
