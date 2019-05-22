BLAS.dot(a::Vector{Complex{T}}, b::Vector{Complex{T}}) where T = BLAS.dotc(a,b)

"""
    cg(A, x::Vector, b::Vector; iterations::Int = 30, relTol = 1.e-3
      , solverInfo = nothing, storeIterations::Bool=false )

Simple conjugate gradient algorithm.
The system matrix contained in AbstractLinearTrafo MUST be symmetrical and
and positive definite

# Arguments
* `A`                           - system matrix
* `x::Vector`                   - solution vector with the initial guess
* `b::Vector`                   - data vector (right hand side of equation)
* `iterations::Int = 30`        - maximum number of iterations
* `relTol = 1.e-3`              - stopping criteria for the relativ residual change
* `solverInfo = nothing`        - `solverInfo` object used to store convergence metrics
* `storeIterations::Bool=false` - if true, the number of iterations until convergence are stored
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
  rsold = BLAS.dot(r,r)
  rsnew = rsold
  r0= norm(b)
  iter_conv = iterations

  for i=1:iterations
    Ap = A*p
    bla = BLAS.dot(p,Ap)

    alpha = rsold/ bla
    #x = x+alpha*p;
    BLAS.axpy!(alpha,p,x)
    # BLAS.axpy!(a,X,Y) overwrites Y with a*X + Y
    BLAS.axpy!(-alpha,Ap,r)
    #r = r-alpha*Ap
    rsnew = BLAS.dot(r,r)
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
    cg(A, x::Vector, b::Vector, M; iterations::Int = 30, relTol = 1.e-3
      , solverInfo = nothing, storeIterations::Bool=false )

Preconditionned conjugate gradient algorithm.
The system matrix contained in AbstractLinearTrafo MUST be symmetrical and
and positive definite

The arguments are the same as for the non-preconditionned case.
`M` denotes the precondionner.
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
  rsold = BLAS.dot(z,r)
  rsnew = rsold
  r0= norm(b)
  iter_conv = iterations

  for i=1:iterations
    Ap = A*p
    bla = BLAS.dot(p,Ap)

    alpha = rsold/ bla
    #x = x+alpha*p;
    BLAS.axpy!(alpha,p,x)
    # BLAS.axpy!(a,X,Y) overwrites Y with a*X + Y
    # r = r-alpha*Ap
    BLAS.axpy!(-alpha,Ap,r)
    z = M*r

    rsnew = BLAS.dot(z,r)
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
