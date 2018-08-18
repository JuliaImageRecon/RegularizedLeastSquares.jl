export fista

mutable struct FISTA <: AbstractLinearSolver
  A
  regularizer::Regularization
  params
end

FISTA(A, regularization; kargs...) = FISTA(A,regularization,kargs)

function solve(solver::FISTA, b::Vector)
  return fista(solver.A, b, solver.regularizer; solver.params...)
end

function fista(A,b::Vector{T}, reg::Regularization; AHA=nothing, kargs...) where T
  if AHA==nothing
    return fista1(A,b,reg;kargs...)
  else
    return fista2(A,b,reg;kargs...)
  end
end

"""This funtion implements the fista algorithm.
Solve the problem: X = arg min_x 1/2*|| Ax-b||² + λ*g(X) where:
   x: variable (vector)
   b: measured data
   A: a general linear operator
   g(X): a convex but not necessarily a smooth function
"""
function fista1(A, b::Vector{T}, reg::Regularization
                ; sparseTrafo=nothing
                , startVector=nothing
                , iterations::Int64=50
                , ρ::Float64=1.0
                , t::Float64=1.0
                , ɛ::Float64=1.e-4
                , solverInfo = nothing
                , kargs...) where T

  p = Progress(iterations, 1, "FISTA iteration...")

  if startVector == nothing
    x = A' * b
  else
    x = startVector
  end
  res = A*x-b

  solverInfo != nothing && storeInfo(solverInfo,norm(res),norm(x))

  xᵒˡᵈ = copy(x)

  A_mul_B!(reg,ρ)
  costFunc = 0.5*norm(res)^2+norm(reg,x)

  for l=1:iterations
    xᵒˡᵈ[:] = x[:]

    x[:] = x[:] - ρ* (A' * res)

    if sparseTrafo != nothing
      xˢᵖᵃʳˢᵉ = sparseTrafo*x[:]
      prox!(reg, xˢᵖᵃʳˢᵉ)
      x = sparseTrafo\xˢᵖᵃʳˢᵉ[:]
    else
      prox!( reg, x)
    end


    tᵒˡᵈ = t

    t = (1. + sqrt(1. + 4. * tᵒˡᵈ^2)) / 2.
    x[:] = x + (tᵒˡᵈ-1)/t*(x-xᵒˡᵈ)

    res = A*x-b
    regNorm = norm(reg,x)

    solverInfo != nothing && storeInfo(solverInfo,norm(res),regNorm)

    # exit if objective functional changes by less then ɛ
    costFuncOld = costFunc
    costFunc = 0.5*norm(res)^2+regNorm
    abs(costFunc-costFuncOld)/costFuncOld < ɛ && return x

    next!(p)
  end

  return x
end

# alternative implementation allowing for an optimized AHA
# does not contain a stopping condition
function fista2(A, b::Vector{T}, reg::Regularization
                ; AHA=nothing
                , sparseTrafo=nothing
                , startVector=nothing
                , iterations::Int64=50
                , ρ::Float64=1.0
                , t::Float64=1.0
                , solverInfo = nothing
                , kargs...) where T

  p = Progress(iterations, 1, "FISTA iteration...")

  if startVector == nothing
    x = A' * b
  else
    x = startVector
  end

  # if AHA!=nothing
    op = AHA
  # else
  #   op = A'*A
  # end

  β = A'*b

  xᵒˡᵈ = copy(x)

  A_mul_B!(reg,ρ)
  costFunc = 0.5*norm(res)^2+norm(reg,x)

  for l=1:iterations
    xᵒˡᵈ[:] = x[:]

    x[:] = x[:] - ρ*(op*x-β)

    if sparseTrafo != nothing
      xˢᵖᵃʳˢᵉ = sparseTrafo*x[:]
      prox!(reg, xˢᵖᵃʳˢᵉ)
      x = sparseTrafo\xˢᵖᵃʳˢᵉ[:]
    else
      prox!( reg, x)
    end

    tᵒˡᵈ = t

    t = (1. + sqrt(1. + 4. * tᵒˡᵈ^2)) / 2.
    x[:] = x + (tᵒˡᵈ-1)/t*(x-xᵒˡᵈ)

    next!(p)
  end

  return x
end
