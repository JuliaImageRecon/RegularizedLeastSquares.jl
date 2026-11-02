const LO = LinearOperatorCollection.LinearOperators

# CPU runs check a same-backend (Array) and a cross-backend (JLArray) target, GPU runs their own arrayType.
adaptTargets = arrayType === Array ? (Array, JLArray) : (arrayType,)

onTarget(x::AbstractArray, target) = parent(x) isa target
onTarget(op, target) = LO.storage_type(op) <: target

cgnrVectors(s::RegularizedLeastSquares.CGNRState) = (s.x, s.x₀, s.pl, s.vl)
cgnrScalars(s::RegularizedLeastSquares.CGNRState) = (s.αl, s.βl, s.ζl, s.iteration, s.relTol, s.z0)
snapshot(s::RegularizedLeastSquares.CGNRState) = (map(v -> copy(Array(v)), cgnrVectors(s)), cgnrScalars(s))
snapshot(s::Union{SequentialState, MultiThreadingState}) = (map(snapshot, s.states), copy(s.active))
function stateOn(s::RegularizedLeastSquares.CGNRState, target)
  all(v -> v isa target, cgnrVectors(s))
end
stateOn(s::Union{SequentialState, MultiThreadingState}, target) = s.states isa Vector && s.active isa Vector{Bool} && all(c -> stateOn(c, target), s.states)
function sharesVectors(a::RegularizedLeastSquares.CGNRState, b::RegularizedLeastSquares.CGNRState)
  any(v -> any(w -> v === w, cgnrVectors(b)), cgnrVectors(a))
end
sharesVectors(a, b) = any(sharesVectors(x, y) for x in a.states, y in b.states)

# Deterministic, well-conditioned fixtures
# seeded so the tolerance checks see the same data on every run
const adaptRng = MersenneTwister(27)

function adaptFixture(T, m = 12, n = 6; seed = 1234)
  rng = MersenneTwister(seed)
  A = T.(randn(rng, T, m, n) .+ 2 * vcat(Matrix(I, n, n), zeros(m - n, n)))
  return A, randn(rng, T, m), randn(rng, T, n), rand(rng, real(T), m) .+ real(T)(0.5)
end

struct NoopAdaptor end
struct FloatOnlyAdaptor end   # moves floating point arrays only
Adapt.adapt_storage(::FloatOnlyAdaptor, x::AbstractArray{<:AbstractFloat}) = JLArray(x)
struct RealToComplexAdaptor end   # moves complex arrays, but also converts real arrays to complex
Adapt.adapt_storage(::RealToComplexAdaptor, x::AbstractArray{ComplexF32}) = JLArray(x)
Adapt.adapt_storage(::RealToComplexAdaptor, x::AbstractArray{Float32}) = JLArray(ComplexF32.(x))
struct Float32ToFloat64Adaptor end
Adapt.adapt_storage(::Float32ToFloat64Adaptor, x::AbstractArray{ComplexF32}) = JLArray(x)
Adapt.adapt_storage(::Float32ToFloat64Adaptor, x::AbstractArray{Float32}) = JLArray(Float64.(x))

struct CustomCGNRState <: RegularizedLeastSquares.AbstractSolverState{CGNR} end
struct CustomProjection <: RegularizedLeastSquares.AbstractProjectionRegularization end

# Operator that keeps constructor inputs to rebuild itself, like LinearOperatorCollection's NFFTOp
struct ReplayOp{T, V <: AbstractVector{T}} <: LO.AbstractLinearOperator{T}
  Mv::V
  plan_args::Dict{Symbol, Any}
end
LO.storage_type(op::ReplayOp) = typeof(op.Mv)
Adapt.adapt_structure(to, op::ReplayOp) = ReplayOp(Adapt.adapt(to, op.Mv),
  Dict{Symbol, Any}(:nodes => Adapt.adapt(to, op.plan_args[:nodes]), :kwargs => Dict{Symbol, Any}(:m => 4)))

function testCGNRAdaptSolve(target, T, mode)
  A, b, x0, w = adaptFixture(T)
  operator = mode == :weighted ? ProdOp(WeightingOp(w), A) : A
  source = if mode == :normal
    CGNR(; AHA = A' * A, reg = L2Regularization(real(T)(0.2)), normalizeReg = NoNormalization(), iterations = 6)
  else
    CGNR(operator; reg = L2Regularization(real(T)(0.2)), iterations = 6)
  end
  rhs = mode == :normal ? A' * b : b
  rhs_copy = copy(rhs)
  if mode == :weighted
    v = randn(adaptRng, T, 6)
    @test source.AHA isa NormalOp
    @test source.AHA * v ≈ (operator' * operator) * v rtol = 1e-5
  end

  # fresh solver: adapt -> init! -> iterate -> solve!
  fresh = Adapt.adapt(target, source)
  @test stateOn(fresh.state, target) && onTarget(fresh.AHA, target)
  @test isnothing(fresh.A) == isnothing(source.A)
  isnothing(fresh.A) || @test onTarget(fresh.A, target)
  mode == :weighted && @test fresh.AHA.parent === fresh.A
  state = fresh.state
  init!(fresh, target(rhs); x0 = target(x0))
  @test fresh.state === state
  @test iterate(fresh) !== nothing
  @test Array(solve!(fresh, target(rhs); x0 = target(x0))) ≈ solve!(deepcopy(source), rhs; x0) rtol = 2e-4

  # initialized, mid-iteration solver keeps its progress
  init!(source, rhs; x0)
  iterate(source)
  before = snapshot(source.state)
  λ_before = λ(source.L2)
  moved = Adapt.adapt(target, source)
  @test moved !== source && moved.state !== source.state && !sharesVectors(moved.state, source.state)
  @test stateOn(moved.state, target)
  @test eltype(moved.state.x) == T && cgnrScalars(moved.state) == cgnrScalars(source.state)
  @test moved.iterations == source.iterations && moved.normalizeReg === source.normalizeReg
  @test λ(moved.L2) == λ_before
  reference = deepcopy(source)
  @test iterate(moved) !== nothing
  @test iterate(reference) !== nothing
  @test moved.state.iteration == reference.state.iteration == 2
  @test norm(Array(moved.state.x) - reference.state.x) <= 2e-4 * norm(reference.state.x)
  @test snapshot(source.state) == before

  # round trip and solve
  back = Adapt.adapt(Array, moved)
  @test stateOn(back.state, Array) && snapshot(back.state) == snapshot(moved.state)
  result = solve!(moved, target(rhs); x0 = target(x0))
  @test result isa target
  @test Array(result) ≈ solve!(reference, rhs; x0) rtol = 2e-4
  @test snapshot(source.state) == before && λ(source.L2) == λ_before && rhs == rhs_copy
end

function testCGNRAdaptErrors(target)
  A, b, _, w = adaptFixture(Float32)
  solver = CGNR(A; iterations = 3)
  for bad in (Int, AbstractArray, NoopAdaptor(), target{Float64})
    @test_throws ArgumentError Adapt.adapt(bad, solver)
  end
  err = try Adapt.adapt(Int, solver) catch e e end
  @test occursin("unsupported Adapt target", sprint(showerror, err))

  P = ProdOp(WeightingOp(w), A)
  @test_throws ArgumentError Adapt.adapt(target, CGNR(P; AHA = P' * P))                 # opaque product
  @test_throws ArgumentError Adapt.adapt(target, CGNR(A; AHA = LinearOperator(A' * A)))  # closure operator
  err = try Adapt.adapt(target, CGNR(P; AHA = P' * P)) catch e e end
  @test occursin("no Adapt rule", sprint(showerror, err))

  # unsupported operators nested in supported ones, as A, AHA and transform
  n = 6
  nested() = ProdOp(WeightingOp(rand(adaptRng, Float32, n) .+ 0.5f0), opEye(Float32, n))
  B = A[1:n, :]
  @test_throws ArgumentError Adapt.adapt(target, CGNR(nested(); AHA = B' * B))
  err = try Adapt.adapt(target, CGNR(nested(); AHA = B' * B)) catch e e end
  # same backend: the opaque child is caught by the graph check; cross backend: LinearOperators throws while rebuilding
  @test occursin(target === Array ? "no Adapt rule" : "could not be adapted", sprint(showerror, err))
  @test_throws ArgumentError Adapt.adapt(target, CGNR(B; AHA = nested()))
  @test_throws ArgumentError Adapt.adapt(target, CGNR(B; reg = [L2Regularization(0.1f0), TransformedRegularization(RealRegularization(), nested())]))

  # unsupported state and regularization terms
  custom = CGNR(A; iterations = 3)
  custom.state = CustomCGNRState()
  @test_throws ArgumentError Adapt.adapt(target, custom)
  s = CGNR(A; iterations = 3)
  @test_throws ArgumentError Adapt.adapt(target, CGNR(s.A, s.AHA, s.L2, [CustomProjection()], s.normalizeReg, s.iterations, s.state))
  @test_throws ArgumentError Adapt.adapt(target, CGNR(s.A, s.AHA, L2Regularization(fill(0.1f0, n)), s.constr, s.normalizeReg, s.iterations, s.state))

  # an operator shared in a nested position cannot be split into separate copies
  Psq = ProdOp(WeightingOp(rand(adaptRng, Float32, n) .+ 0.5f0), B)
  shared = CGNR(Psq; reg = [L2Regularization(0.1f0), TransformedRegularization(RealRegularization(), Psq.A)])
  err = try Adapt.adapt(target, shared) catch e e end
  @test err isa ArgumentError && occursin("shared", sprint(showerror, err))
end

# Adaptors that only move or convert part of the graph (CPU only, they target JLArray)
function testCGNRPartialAdaptors()
  solver = CGNR(Int[1 0; 0 2]; AHA = Float32[1 0; 0 4], iterations = 2)
  @test_throws ArgumentError Adapt.adapt(FloatOnlyAdaptor(), solver)
  moved = Adapt.adapt(JLArray, solver)
  @test moved.A isa JLArray{Int} && moved.AHA isa JLArray{Float32}
  @test Array(solve!(moved, JLArray(Float32[1, 2]))) ≈ solve!(deepcopy(solver), Float32[1, 2]) rtol = 2e-4

  mixed = ProdOp(WeightingOp(rand(adaptRng, Float32, 4) .+ 0.5f0), WeightingOp(cis.(2f0 * pi * rand(adaptRng, Float32, 4)) .* (rand(adaptRng, Float32, 4) .+ 0.5f0)))
  solver = CGNR(mixed; iterations = 4)
  moved = Adapt.adapt(JLArray, solver)
  @test LO.storage_type(moved.A) <: JLArray{ComplexF32} && LO.storage_type(moved.A.A) <: JLArray{Float32}
  b = randn(adaptRng, ComplexF32, 4)
  @test Array(solve!(moved, JLArray(b))) ≈ solve!(deepcopy(solver), b) rtol = 2e-4
  @test_throws ArgumentError Adapt.adapt(RealToComplexAdaptor(), solver)

  replay = ReplayOp(zeros(ComplexF32, 4), Dict{Symbol, Any}(:nodes => rand(adaptRng, Float32, 2, 4), :kwargs => Dict{Symbol, Any}(:m => 4)))
  solver = CGNR(replay; AHA = Matrix{ComplexF32}(I, 4, 4), iterations = 2)
  @test Adapt.adapt(JLArray, solver).A.plan_args[:nodes] isa JLArray{Float32}
  err = try Adapt.adapt(Float32ToFloat64Adaptor(), solver) catch e e end
  @test err isa ArgumentError && occursin("nodes", sprint(showerror, err))
end

function testCGNRAdaptSharing(target)
  M, b, _, w = adaptFixture(ComplexF32)
  v = randn(adaptRng, ComplexF32, 6)
  matvec(op, x) = Array(op * target(x))

  # identical top-level components stay identical
  D = Float32[2 0; 0 3]
  moved = Adapt.adapt(target, CGNR(D; AHA = D, iterations = 2))
  @test moved.AHA === moved.A && moved.A isa target
  @test Array(solve!(moved, target(Float32[1, 1]))) ≈ solve!(CGNR(D; AHA = D, iterations = 2), Float32[1, 1]) rtol = 2e-4

  # default normal operator follows the adapted forward operator
  P = ProdOp(WeightingOp(copy(w)), M)
  source = CGNR(P; iterations = 4)
  source_action = source.AHA * v
  moved = Adapt.adapt(target, source)
  @test moved.AHA.parent === moved.A
  moved.A.A.weights .*= 2
  @test matvec(moved.AHA, v) ≈ matvec(moved.A', Array(moved.A * target(v))) rtol = 1e-4
  @test source.AHA * v == source_action

  # explicit normalOperator of a weighted product: shares M, keeps its own W'W snapshot
  P = ProdOp(WeightingOp(copy(w)), copy(M))
  source = CGNR(P; AHA = normalOperator(P), iterations = 4)
  source_action = source.AHA * v
  moved = Adapt.adapt(target, source)
  @test moved.AHA.parent === moved.A.B
  @test matvec(moved.AHA, v) ≈ source_action rtol = 1e-4
  if target !== Array   # on the same backend the read-only matrix M may be shared with the source
    moved.A.B .*= 2
    @test matvec(moved.AHA, v) ≈ 4 .* source_action rtol = 1e-4
    moved.A.A.weights .*= 2
    @test matvec(moved.AHA, v) ≈ 4 .* source_action rtol = 1e-4
    @test source.AHA * v == source_action
  end

  # explicit custom normal operators keep their own action
  custom_dense = M' * M + 2I
  moved = Adapt.adapt(target, CGNR(P; AHA = custom_dense, iterations = 4))
  @test matvec(moved.AHA, v) ≈ custom_dense * v rtol = 1e-4
  other = rand(adaptRng, Float32, 12) .+ 0.5f0
  custom_normal = NormalOp(ComplexF32; parent = P.B, weights = WeightingOp(other))
  moved = Adapt.adapt(target, CGNR(P; AHA = custom_normal, iterations = 4))
  @test moved.AHA.parent === moved.A.B
  @test matvec(moved.AHA, v) ≈ custom_normal * v rtol = 1e-4

  # wrapped dense arrays
  At = Matrix(transpose(M))
  wrapped = adjoint(conj.(At))
  for source in (CGNR(wrapped; iterations = 6), CGNR(M; AHA = adjoint(M' * M), iterations = 6))
    moved = Adapt.adapt(target, source)
    @test onTarget(moved.AHA, target)
    @test Array(solve!(moved, target(b))) ≈ solve!(deepcopy(source), b) rtol = 2e-4
  end

  # host container of child operators
  w1, w2 = rand(adaptRng, Float32, 3) .+ 0.5f0, rand(adaptRng, Float32, 3) .+ 0.5f0
  Dg = DiagOp(WeightingOp(w1), WeightingOp(w2))
  source = CGNR(Dg; AHA = NormalOp(Float32; parent = Dg), iterations = 6)
  moved = Adapt.adapt(target, source)
  @test moved.A.ops isa Vector && all(op -> onTarget(op, target), moved.A.ops)
  bd = rand(adaptRng, Float32, 6)
  @test Array(solve!(moved, target(bd))) ≈ solve!(deepcopy(source), bd) rtol = 2e-4
end

function testCGNRAdaptRegularization(target)
  A, b, _, _ = adaptFixture(ComplexF32, 10, 6)
  transform = WeightingOp(rand(adaptRng, Float32, 6) .+ 0.5f0)
  mask = Bool[1, 0, 1, 0, 1, 0]
  for normalization in (NoNormalization(), MeasurementBasedNormalization(), SystemMatrixBasedNormalization())
    auto = AutoScaledRegularization(L2Regularization(0.2f0))
    auto.factor = 0.25f0
    for l2 in (L2Regularization(0.2f0), FixedScaledRegularization(L2Regularization(0.2f0), 0.4f0),
               FixedParameterRegularization(L2Regularization(0.2f0)), auto)
      regs = [l2, TransformedRegularization(RealRegularization(), transform), MaskedRegularization(PositiveRegularization(), mask)]
      source = CGNR(A; reg = regs, normalizeReg = normalization, iterations = 3)
      init!(source, b)
      moved = Adapt.adapt(target, source)
      @test λ(moved.L2) == λ(source.L2)
      @test moved.constr isa Vector && moved.constr !== source.constr
      @test moved.constr[2].mask isa Vector{Bool} && moved.constr[2].mask == mask && moved.constr[2].mask !== source.constr[2].mask
      @test onTarget(moved.constr[1].trafo, target)
      inner(reg) = reg isa NormalizedRegularization ? reg.reg : reg
      if inner(moved.L2) isa AutoScaledRegularization
        @test inner(moved.L2) !== inner(source.L2) && inner(moved.L2).factor == 0.25f0
        inner(moved.L2).factor = 0.5f0
        @test inner(source.L2).factor == 0.25f0
        inner(moved.L2).factor = 0.25f0
      end
      @test Array(solve!(moved, target(b))) ≈ solve!(deepcopy(source), b) rtol = 2e-4
    end
  end
end

function testCGNRAdaptFrames(target, scheduler)
  A, _, _, _ = adaptFixture(ComplexF32, 10, 6)
  rhs = randn(MersenneTwister(99), ComplexF32, 10, 3)
  source = CGNR(A; reg = L2Regularization(0.1f0), iterations = 4)
  init!(source, rhs; scheduler)
  source.state.active[2] = false
  iterate(source)
  before = snapshot(source.state)
  moved = Adapt.adapt(target, source)
  @test moved.state isa scheduler && stateOn(moved.state, target)
  @test moved.state.active == source.state.active && moved.state.active !== source.state.active
  @test !sharesVectors(moved.state, source.state)
  reference = deepcopy(source)
  iterate(moved); iterate(reference)
  @test Array(solversolution(moved)) ≈ solversolution(reference) rtol = 2e-4
  @test snapshot(source.state) == before
  # single -> matrix -> single reuse on the target
  @test Array(solve!(moved, target(rhs[:, 1]))) ≈ solve!(reference, rhs[:, 1]) rtol = 2e-4
  @test Array(solve!(moved, target(rhs); scheduler)) ≈ solve!(reference, rhs; scheduler) rtol = 2e-4
  @test moved.state isa scheduler && stateOn(moved.state, target)
  @test Array(solve!(moved, target(rhs[:, 2]))) ≈ solve!(reference, rhs[:, 2]) rtol = 2e-4
  @test stateOn(moved.state, target)
  @test snapshot(source.state) == before
end

@testset "CGNR Adapt" begin
  for target in adaptTargets
    @testset "$target" begin
      for T in (Float32, ComplexF32), mode in (:dense, :normal, :weighted)
        @testset "solve $T $mode" begin testCGNRAdaptSolve(target, T, mode) end
      end
      @testset "errors" begin testCGNRAdaptErrors(target) end
      @testset "sharing" begin testCGNRAdaptSharing(target) end
      @testset "regularization" begin testCGNRAdaptRegularization(target) end
      for scheduler in (SequentialState, MultiThreadingState)
        @testset "frames $scheduler" begin testCGNRAdaptFrames(target, scheduler) end
      end
    end
  end
  arrayType === Array && @testset "partial adaptors" begin testCGNRPartialAdaptors() end
end
