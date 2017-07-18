export proxL21!


@doc "group-soft-thresholding for l1/l2-regularization." ->
function proxL21!(reg::Regularization, x)
  proxL21!(x, reg.params[:lambdL1], reg.params[:slices])
end

function proxL21!(x, λ, slices::Int64=1)
  sliceLength = floor(Int, length(x)/slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  x[:] = [ x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0 ) for i=1:length(x)]
end

@doc "return the value of the L21-regularization term" ->
function normL21(reg::Regularization,x)
  sliceLength = floor(Int, length(x)/slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  return norm(groupNorm,1)
end
