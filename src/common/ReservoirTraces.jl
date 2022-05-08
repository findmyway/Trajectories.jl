using MacroTools: @forward

mutable struct ReservoirTraces{T,R}
    traces::T
    n::Int
    capacity::Int
    rng::R
end

@forward ReservoirTrajectory.buffer sample, Base.keys, Base.haskey, Base.getindex, Base.view, Base.length, Base.setindex!, Base.lastindex, Base.firstindex

function Base.push!(t::ReservoirTraces, x)
    if t.n < t.capacity
        push!(t.traces, x)
        t.n += 1
    else
        i = rand(t.rng, 1:length(t))
        t.traces[i] = x
    end
end