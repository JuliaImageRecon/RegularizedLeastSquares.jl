export FISTA, fista

type FISTA <: AbstractLinearSolver
  A
  params
end

FISTA(A; kargs...) = FISTA(A,kargs)

function solve{T}(solver::FISTA, u::Vector{T})
  x = zeros(T, size(solver.A,2))
  return fista(solver.A, u, x; solver.params... )
end

"""
 A Fast Iterative Shrinkage-Thresholding Algorithm for
 Linear Inverse Problems.
 Solve the problem: X = arg min_X F(X) = f(X) + lambda*g(X) where:
    X: variable, can be a matrix.
    f(X): a smooth convex function with continuously differentiable
       with Lipschitz continuous gradient `L(f)` (Lipschitz constant of
       the gradient of `f`).
    g(X): a convex but not necessarily a smooth function
"""
function fista(A, b, x; W=Base.I
                , Mask=Base.I, iterations::Int64=50, rho::Float64=1.0
                , lambda::Float64=1e-6, t::Float64=1.0, slices::Int64=1
                , verbose::Bool=true, verboseResid::Bool=false, kargs...)

  if verbose == true
  p = Progress(iterations*4+1, 1, "Solve linear equation...")
  end

  if verbose==true
    next!(p)
  end
  if verboseResid == true
    println("Residuum: ", norm(A*x-b))
    #println("Norm(b): ", norm(b))
  end

  xOld = copy(x)

  #lambdDecreaseFactor=1.0/1.5
  #alpha *= norm(x,Inf) / (lambdDecreaseFactor^(iterations-1) )

  for l=1:iterations

      xOld[:] = x[:]

      x[:] = Mask*x
      if verbose==true
        next!(p)
      end

      res = A*x-b

      x[:] = x[:] - rho*At_mul_B(A,res)

      if verbose==true
        next!(p)
      end


      #x[:] = W\softThreshold(W*x, 2*rho*alpha)
      x[:] = W\softThreshold(W*x, lambda,slices)
      if verbose==true
        next!(p)
      end

      tOld = t

      t = (1. + sqrt(1. + 4. * tOld^2)) / 2.
      #println(size(x))
      #println(size(xOld))
      x[:] = x + (tOld-1)/t*(x-xOld)
      if verbose==true
        next!(p)
      end


      #alpha *= lambdDecreaseFactor
      if verboseResid == true
        println("Residuum: ", norm(A*x-b))
      end

  end
  return x
end
