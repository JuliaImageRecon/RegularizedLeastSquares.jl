export cg

type CG <: AbstractLinearSolver
  A
  params
end

CG(A; kargs...) = FISTA(A,kargs)

function solve(solver::CG, b::Vector)
  return cg(solver.A, b; solver.params...)
end

"""
  Simple conjugate gradient algorithm.
  The system matrix contained in AbstractLinearTrafo MUST be symmetrical and
  and positive definite
"""
function cg(A
            , x::Vector
            , b::Vector
            ; iterations::Int = 30
            , verbose::Bool=true
            , solverInfo = nothing)

  r = b-A_mul_B(A,x)
  p = r
  rsold = BLAS.dot(r,r)
  rsnew = rsold

  if verbose==true
    #progr = Progress(iterations, 1, "Iterating Conjugate Gradient...")
    progr = Progress(iterations,dt=0.1,desc="Doing CG...";barglyphs=BarGlyphs("[=> ]"),barlen=50);
  end

  for i=1:iterations
    Ap = A_mul_B(A,p)
    bla = BLAS.dot(p,Ap)

    alpha = rsold/ bla
    #x = x+alpha*p;
    BLAS.axpy!(alpha,p,x)
    # BLAS.axpy!(a,X,Y) overwrites Y with a*X + Y
    BLAS.axpy!(-alpha,Ap,r)
    #r = r-alpha*Ap
    rsnew = BLAS.dot(r,r)
    if sqrt(abs(rsnew))<1e-10
      break
    end

    beta = rsnew/rsold
    p = r+beta*p
    rsold = rsnew

    if verbose==true
      next!(progr)
    end
  end

  solverInfo != nothing && storeResidual(solverInfo, sqrt(abs(rsnew)) )

  return x
end
