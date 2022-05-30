using ReinforcementLearningTrajectories
using CircularArrayBuffers
using Test

@testset "ReinforcementLearningTrajectories.jl" begin
    include("traces.jl")
    include("common.jl")
    include("samplers.jl")
    include("trajectories.jl")
    include("normalization.jl")
    include("samplers.jl")
end
