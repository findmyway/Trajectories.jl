@testset "BatchSampler" begin
    sz = 32
    s = BatchSampler(sz)
    t = Traces(
        state=rand(3, 4, 5),
        action=rand(1:4, 5),
    )

    b = ReinforcementLearningTrajectories.sample(s, t)

    @test keys(b) == (:state, :action)
    @test size(b.state) == (3, 4, sz)
    @test size(b.action) == (sz,)

    e = Episodes() do
        Episode(Traces(state=rand(2, 3, 0), action=rand(0)))
    end

    push!(e, Episode(Traces(state=rand(2, 3, 2), action=rand(2))))
    push!(e, Episode(Traces(state=rand(2, 3, 3), action=rand(3))))

    @test length(e) == 5
    @test size(e[2:4].state) == (2, 3, 3)
    @test size(e[2:4].action) == (3,)
end

@testset "MetaSampler" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=MetaSampler(policy=BatchSampler(3), critic=BatchSampler(5)),
    )

    append!(t, Traces(a=rand(Int, 10), b=rand(Bool, 10)))

    batches = collect(t)

    @test length(batches) == 10
    @test length(batches[1][:policy][:a]) == 3 && length(batches[1][:critic][:b]) == 5
end

@testset "MultiBatchSampler" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=MetaSampler(policy=BatchSampler(3), critic=MultiBatchSampler(BatchSampler(5), 2)),
    )

    append!(t, Traces(a=rand(Int, 10), b=rand(Bool, 10)))

    batches = collect(t)

    @test length(batches) == 10
    @test length(batches[1][:policy][:a]) == 3
    @test length(batches[1][:critic]) == 2 # we sampled 2 batches for critic
    @test length(batches[1][:critic][1][:b]) == 5 #each batch is 5 samples 
end