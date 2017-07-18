export proxL1!, proxL1, normL1

@doc "soft-thresholding for the Lasso problem." ->
function proxL1!(reg::Regularization, x)
  proxL1!(x, reg.params[:lambdL1])
end

function proxL1!(x, λ)
  x[:] = [i*max( (abs(i)-λ)/abs(i),0 ) for i in x]
end

@doc "return the value of the L1-regularization term" ->
normL1(reg::Regularization,x) = reg.params[:lambdL1]*norm(x,1)
