export Trajectory

"""
    Trajectory(container, sampler, controler)

The `container` is used to store experiences. Common ones are [`Traces`](@ref)
or [`Episodes`](@ref). The `sampler` is used to sample experience batches from
the `container`. The `controler` controls whether it is time to sample a batch
or not.

Supported methoes are:

- `push!(t::Trajectory, experience)`, add one experience into the trajectory.
- `append!(t::Trajectory, batch)`, add a batch of experiences into the trajectory.
- `take!(t::Trajectory)`, take a batch of experiences from the trajectory. Note
  that `nothing` may be returned, indicating that it's not ready to sample yet.
"""
Base.@kwdef struct Trajectory{C,S,T}
    container::C
    sampler::S
    controler::T
end

function Base.push!(t::Trajectory, x)
    n_pre = length(t)
    push!(t.container, x)
    n_post = length(t)
    on_insert!(t.controler, n_post - n_pre)
end

function Base.append!(t::Trajectory, x)
    n_pre = length(t)
    append!(t.container, x)
    n_post = length(t)
    on_insert!(t.controler, n_post - n_pre)
end

function Base.take!(t::Trajectory)
    res = on_take!(t.controler)
    if isnothing(res)
        nothing
    else
        rand(t.sampler, t.container)
    end
end

function Base.iterate(t::Trajectory)
    x = take!(t)
    if isnothing(x)
        nothing
    else
        x, true
    end
end

Base.iterate(t::Trajectory, state) = iterate(t)

#####

mutable struct InsertSampleRatioControler
    n_inserted::Int
    n_sampled::Int
    threshold::Int
    ratio::Float64
end

function on_insert!(c::InsertSampleRatioControler, n::Int)
    if n > 0
        c.n_inserted += n
    end
end

function on_take!(c::InsertSampleRatioControler)
    if c.n_inserted >= c.threshold
        if c.n_sampled <= c.n_inserted * c.ratio
            c.n_sampled += 1
            true
        end
    end
end