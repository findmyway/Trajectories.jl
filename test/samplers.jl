using Trajectories, Test

@testset "MetaSampler" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler = MetaSampler(policy = BatchSampler(3), critic = BatchSampler(5)),
        controler = InsertSampleControler(10, 0)
    )

    append!(t; a=[1, 2, 3, 4], b=[false, true, false, true])

    batches = []

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 10
    @test length(batches[1][:policy][:a]) == 3 && length(batches[1][:critic][:b]) == 5    
end

@testset "MultiBatchSampler" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler = MetaSampler(policy = BatchSampler(3), critic = MultiBatchSampler(BatchSampler(5), 2)),
        controler = InsertSampleControler(10, 0)
    )

    append!(t; a=[1, 2, 3, 4], b=[false, true, false, true])

    batches = []

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 10
    @test length(batches[1][:policy][:a]) == 3 
    @test length(batches[1][:critic]) == 2 # we sampled 2 batches for critic
    @test length(batches[1][:critic][1][:b]) == 5 #each batch is 5 samples 
end

@testset "async trajectories" begin
    threshould = 100
    ratio = 1 / 4
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=BatchSampler(3),
        controler=AsyncInsertSampleRatioControler(ratio, threshould)
    )

    n = 100
    insert_task = @async for i in 1:n
        append!(t; a=[i, i, i, i], b=[false, true, false, true])
    end

    s = 0
    sample_task = @async for _ in t
        s += 1
    end
    sleep(1)
    @test s == (n - threshould * ratio) + 1
end