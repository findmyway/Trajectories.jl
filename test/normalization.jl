using Test
using Trajectories
import Trajectories.normalize!
import OnlineStats: fit!, mean, std

#@testset "normalization.jl" begin
    #scalar normalization
    rewards = [1.:10;]
    rn = reward_normalizer()
    fit!(rn, rewards)
    batch_reward = [6.,5.,10.]
    output = normalize!(rn, batch_reward)
    @test batch_reward == output != [6.,5.,10.]
    #vector normalization
    states = reshape([1:50;], 5, 10)
    sn = state_normalizer((5,))
    fit!(sn, eachslice(states; dims = ndims(states)))
    @test [mean(stat) for stat in sn] == [mean((1:5:46) .+i) for i in 0:4]
    batch_states = reshape(repeat(5.:-1:1, 5), 5,5)
    normalize!(sn, batch_states)
    @test all(length(unique(x)) == 1 for x in eachrow(batch_states))
    #array normalization
    states = reshape(1.:250, 5,5,10)
    sn = state_normalizer((5,5))
    fit!(sn, eachslice(states, dims = 3))
    batch_states = collect(states)
    normalize!(sn, batch_states)
    
    #NormalizedTrajectory
    t = Trajectory(
        container=Traces(
            a=Float32[],
            b=Int[]
        ),
        sampler=BatchSampler(30000),
        controler=InsertSampleRatioControler(Inf, 0)
    )
    nt = NormalizedTrajectory(t, :a => reward_normalizer())
    append!(nt, a = [1,2,3], b = [1,2,3])
    push!(nt, a = 2, b = 2)
    @test mean(nt.normalizer[:a]) ≈ 2.
    @test std(nt.normalizer[:a]) ≈ std([1,2,2,3])
    a,b = take!(nt)
    @test eltype(a) == Float32
    @test mean(a) ≈ 0 atol = 0.01
    @test mean(b) ≈ 2 atol = 0.01 #b is not normalized
end