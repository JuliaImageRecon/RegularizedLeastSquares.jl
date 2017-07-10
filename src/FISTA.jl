export fista

type FISTA <: AbstractLinearSolver
  A
  regularizer::Regularization
  params
end

FISTA(A, regularization; kargs...) = FISTA(A,regularization,kargs)

function solve(solver::FISTA, b::Vector)
  return fista(solver.A, b, solver.regularizer; solver.params...)
end

@doc "This funtion implements the fista algorithm." ->
function fista{T}(A, b::Vector{T}, reg::Regularization
                ; sparseTrafo=nothing
                , startVector=nothing
                , iterations::Int64=50
                , ρ::Float64=1.0
                , t::Float64=1.0
                , solverInfo = nothing
                , kargs...)

  p = Progress(iterations, 1, "FISTA iteration...")

  if startVector == nothing
    x = Ac_mul_B(A,b)
  else
    x = startVector
  end
  res = A*x-b

  solverInfo != nothing && storeInfo(solverInfo,norm(res),norm(x))

  xᵒˡᵈ = copy(x)

  for l=1:iterations
    xᵒˡᵈ[:] = x[:]

    x[:] = x[:] - ρ*Ac_mul_B(A, res)

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
    solverInfo != nothing && storeInfo(solverInfo,norm(res),norm(x))

    next!(p)
  end

  return x
end
