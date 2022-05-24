using Trajectories
using CircularArrayBuffers
using Test

@testset "Trajectories.jl" begin
    include("traces.jl")
    include("common.jl")
    include("trajectories.jl")
end
