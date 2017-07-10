export proxL2!


@doc "proximal map for Tikhonov regularization." ->
function proxL2!(reg::Regularization, x)
  proxL2!(x, reg.params[:lambdL2])
end

function proxL2!(x, λ)
  x[:] = 1./(1.+2.*λ)*x
end
