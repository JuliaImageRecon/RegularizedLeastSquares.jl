@deprecate createLinearSolver(solver, A, x; kargs...) createLinearSolver(solver, A; kargs...)

function Base.vec(reg::AbstractRegularization)
    Base.depwarn("vec(reg::AbstractRegularization) will be removed in a future release. Use `reg = isa(reg, AbstractVector) ? reg : [reg]` instead.", reg; force=true)
    return AbstractRegularization[reg]
end

function Base.vec(reg::AbstractVector{AbstractRegularization})
    Base.depwarn("vec(reg::AbstractRegularization) will be removed in a future release. Use reg = `isa(reg, AbstractVector) ? reg : [reg]` instead.", reg; force=true)
    return reg
end

export ConstraintTransformedRegularization
function ConstraintTransformedRegularization(args...)
    error("ConstraintTransformedRegularization has been removed. ADMM and SplitBregman now take the regularizer and the transform as separat inputs.")
end