export proxTV!, normTV

"""
    proxTV!(x::Vector{T}, λ::Float64; kargs...) where T

proximal map for ansitropic TV regularization using the Fast Gradient Projection algorithm.

# Arguments
* `x::Array{T}`             - Vector to apply proximal map to
* `λ::Float64`              - regularization paramter
* `shape::NTuple=[]`        - size of the underlying image
* `iterationsTV::Int64=20`  - number of FGP iterations
* `weights::Array=[]`       - weights to apply to the image gradients
"""
function proxTV!(x::Vector{T}, λ::Float64; shape::NTuple=[], iterationsTV::Int64=20, weights::Array=[], kargs...) where T
  m,n = shape

  # initialize dual variables
  p = zeros(eltype(x), m-1,n)
  q = zeros(eltype(x), m,n-1)
  r = zeros(eltype(x), m-1,n)
  s = zeros(eltype(x), m,n-1)
  pOld = zeros(eltype(x), m-1,n)
  qOld = zeros(eltype(x), m,n-1)
  weights!=[] ? weights=reshape(weights,shape) : weights=ones(shape)

  t = 1
  for i=1:iterationsTV
    pOld[:] = p[:]
    qOld[:] = q[:]

    # gradient projection step for dual variables
    p,q = [r,s] + collect( Φ_hermitian( x-λ*Φ(r,s), shape ) ) ./(8*λ)
    p = restrictMagnitude(p, weights[1:m-1,:])
    q = restrictMagnitude(q, weights[:,1:n-1])

    # form linear combinaion of old and new estimates
    tOld = t
    t = ( 1+sqrt(1+4*tOld^2) )/2.0
    r = p + (tOld-1)/t*(p-pOld)
    s = q + (tOld-1)/t*(q-qOld)
  end

  x[:] = x[:]-λ*vec(Φ(p,q))
end

#
# primal-dual mapping
#
function Φ(p::Matrix{T}, q::Matrix{T}) where T
  m = size(q,1)
  n = size(p,2)
  x = zeros(T, m, n)

  # points in the corners
  x[1,1] = p[1,1]+q[1,1]
  x[m,1] = q[m,1]-p[m-1,1]
  x[1,n] = p[1,n]-q[1,n-1]
  x[m,n] = -p[m-1,n]-q[m,n-1]
  # remaining points with i=1 (first row)
  x[1,2:n-1] = p[1,2:n-1]+q[1,2:n-1]-q[1,1:n-2]
  # remaingin points with j=1 (first column)
  x[2:m-1,1] = p[2:m-1,1]+q[2:m-1,1]-p[1:m-2,1]
  #remaining points with i=m (last row)
  x[m,2:n-1] = q[m,2:n-1]-p[m-1,2:n-1]-q[m,1:n-2]
  #remaingin points with j=n (last column)
  x[2:m-1,n] = p[2:m-1,n]-p[1:m-2,n]-q[2:m-1,n-1]
  #all remaining (inner) matrix elements
  x[2:m-1,2:n-1] = p[2:m-1,2:n-1]+q[2:m-1,2:n-1]-p[1:m-2,2:n-1]-q[2:m-1,1:n-2]

  return vec(x)
end

#
# hermitian conjugate of the primal-dual mapping
#
function Φ_hermitian(x::Vector{T}, shape::NTuple) where T
  m,n = shape
  x = reshape(x,m,n)
  p = zeros(T,m-1,n)
  q = zeros(T,m,n-1)
  p[:,:] = x[1:m-1,1:n]-x[2:m,1:n]
  q[:,:] = x[1:m,1:n-1]-x[1:m,2:n]

  return p,q
end

# restrict x to a number smaller then one
# restrictMagnitude(x::Vector) = x/max(1, abs(x))
function restrictMagnitude(x::Array, w::Array=[])
  w != [] ? maxval = w : maxval = ones(size(x))
  return x./max.(1.0, abs.(x)./maxval)
end

"""
    normTV(x::Vector{T},λ::Float64;shape::NTuple=[],kargs...) where T

returns the value of the ansisotropic TV-regularization term.
Arguments are the same as in `proxTV!`
"""
function normTV(x::Vector{T},λ::Float64;shape::NTuple=[],kargs...) where T
  x = reshape(x,shape)
  tv = norm(vec(x[1:end-1,1:end-1]-x[2:end,1:end-1]),1)
  tv += norm(vec(x[1:end-1,1:end-1]-x[1:end-1,2:end]),1)
  tv += norm(vec(x[1:end-1,end]-x[2:end,end]),1)
  tv += norm(vec(x[end,1:end-1]-x[end,2:end]),1)
  return λ*tv
end
