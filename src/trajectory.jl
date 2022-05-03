export Trajectory

Base.@kwdef struct Trajectory{T,S}
    traces::T
    sampler::S
end

Base.rand(t::Trajectory) = rand(t.sampler, t.traces)

Base.push!(t::Trajectory; kw...) = push!(t.traces; kw...)
Base.append!(t::Trajectory; kw...) = append!(t.traces; kw...)

Base.getindex(t::Trajectory, k) = getindex(t.traces, k)
Base.setindex!(t::Trajectory, v, ks...) = setindex!(t.traces, v, ks...)
Base.length(t::Trajectory) = length(t.traces)
