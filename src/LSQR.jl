
type LSQR <: AbstractLinearSolver
  A
  params
end

LSQR(A; kargs...) = LSQR(A,kargs)

function init(solver::LSQR)
  nothing
end

function deinit(solver::LSQR)
  nothing
end

function solve(solver::LSQR, u::Vector)
  return lsqr_(solver.A, u; solver.params... )
end


function lsqr_(S, u::Vector;
 iterations=10, lambd=0.0, weights=nothing, enforceReal=true, enforcePositive=true, sparseTrafo=nothing, startVector=nothing, solverInfo=nothing, verbose = true ,kargs...)
  T = typeof(real(u[1]))
  lambd = convert(T,lambd)
  weights==nothing ? weights=ones(T,size(S,1)) : nothing #search for positive solution as default
  startVector==nothing ? startVector=zeros(typeof(u[1]),size(S,2)) : nothing

  cl,h = lsqr(S, u, maxiter=iterations)

  # invoke constraints
  A_mul_B!(sparseTrafo, cl)
  enforceReal && enfReal!(cl)
  enforcePositive && enfPos!(cl)
  At_mul_B!(sparseTrafo, cl)

  return cl
end
