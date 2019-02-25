"""
  Simple conjugate gradient algorithm.
  The system matrix contained in AbstractLinearTrafo MUST be symmetrical and
  and positive definite
"""
function cg(A
            , x::Vector
            , b::Vector
            ; iterations::Int = 30
            , relTol = 1.e-3
            , solverInfo = nothing
            , storeIterations::Bool=false )

  r = b-A*x
  p = r
  rsold = BLAS.dotc(r,r)
  rsnew = rsold
  r0= norm(b)
  iter_conv = iterations

  for i=1:iterations
    Ap = A*p
    bla = BLAS.dotc(p,Ap)

    alpha = rsold/ bla
    #x = x+alpha*p;
    BLAS.axpy!(alpha,p,x)
    # BLAS.axpy!(a,X,Y) overwrites Y with a*X + Y
    BLAS.axpy!(-alpha,Ap,r)
    #r = r-alpha*Ap
    rsnew = BLAS.dotc(r,r)
    if sqrt(abs(rsnew))/r0<relTol
      iter_conv = i
      break
    end

    beta = rsnew/rsold
    p = r+beta*p
    rsold = rsnew
  end

  if solverInfo != nothing
    if storeIterations
      storeIter(solverInfo,iter_conv)
    else
      storeInfo(solverInfo,A,b,x;residual=rsnew)
    end
  end
  return x
end

"""
  Preconditionned conjugate gradient algorithm.
"""
function cg(A
            , x::Vector
            , b::Vector
            , M
            ; iterations::Int = 30
            , relTol = 1.e-3
            , solverInfo = nothing
            , storeIterations::Bool=false )

  r = b-A*x
  z = M*r
  p = z
  rsold = BLAS.dotc(z,r) # r^T * z
  rsnew = rsold
  r0= norm(b)
  iter_conv = iterations

  for i=1:iterations
    Ap = A*p
    bla = BLAS.dotc(p,Ap)

    alpha = rsold/ bla
    #x = x+alpha*p;
    BLAS.axpy!(alpha,p,x)
    # BLAS.axpy!(a,X,Y) overwrites Y with a*X + Y
    # r = r-alpha*Ap
    BLAS.axpy!(-alpha,Ap,r)
    z = M*r

    rsnew = BLAS.dotc(z,r)
    if sqrt(abs(rsnew))/r0<relTol
      iter_conv = i
      break
    end

    beta = rsnew/rsold
    p = z+beta*p
    rsold = rsnew
  end

  if solverInfo != nothing
    if storeIterations
      storeIter(solverInfo,iter_conv)
    else
      storeInfo(solverInfo,A,b,x;residual=rsnew)
    end
  end

  return x
end
