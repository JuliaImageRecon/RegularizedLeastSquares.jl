using RegularizedLeastSquares, Images, PyPlot, Random

# image
N = 256
I = shepp_logan(N)

# sampling operator
idx = sort( shuffle( collect(1:N^2) )[1:div(N^2,2)] )
A = SamplingOp(idx,(N,N))

# generate undersampled data
y = A*vec(I)

# regularizer
reg = Regularization("TV", 0.01; shape=(N,N))

# solver
solver = createLinearSolver("admm",A; reg=reg, ρ=0.1, iterations=20)

Ireco = solve(solver,y)
Ireco = reshape(Ireco,N,N)
