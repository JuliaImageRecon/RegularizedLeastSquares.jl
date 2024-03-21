@testset "ProgressBarCallback" begin
    A = [
        0.831658 0.96717
        0.383056 0.39043
        0.820692 0.08118
    ]
    x = [0.593; 0.269]
    b = A * x

    solver = ADMM(A; iterations=50)

    _ = solve!(solver, b, callbacks=ProgressBarCallback())
    _ = solve!(solver, b, callbacks=ProgressBarCallback(solver))
    _ = solve!(solver, b, callbacks=ProgressBarCallback(50))
end