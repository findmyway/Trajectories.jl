export Trajectory, InsertSampleRatioControler

using Base.Threads


#####

mutable struct InsertSampleRatioControler
    ratio::Float64
    threshold::Int
    n_inserted::Int
    n_sampled::Int
end

"""
    InsertSampleRatioControler(ratio, threshold)

Used in [`Trajectory`](@ref). The `threshold` means the minimal number of
insertings before sampling. The `ratio` balances the number of insertings and
the number of samplings.
"""
InsertSampleRatioControler(ratio, threshold) = InsertSampleRatioControler(ratio, threshold, 0, 0)

function on_insert!(c::InsertSampleRatioControler, n::Int)
    if n > 0
        c.n_inserted += n
    end
end

function on_sample!(c::InsertSampleRatioControler)
    if c.n_inserted >= c.threshold
        if c.n_sampled < (c.n_inserted - n.threshold) * c.ratio
            c.n_sampled += 1
            true
        end
    end
end

#####

mutable struct AsyncInsertSampleRatioControler
    ratio::Float64
    threshold::Int
    n_inserted::Int
    n_sampled::Int
    ch_in::Channel
    ch_out::Channel
end

function AsyncInsertSampleRatioControler(
    ratio,
    threshold,
    ; ch_in_sz=1,
    ch_out_sz=1,
    n_inserted=0,
    n_sampled=0
)
    AsyncInsertSampleRatioControler(
        ratio,
        threshold,
        n_inserted,
        n_sampled,
        Channel(ch_in_sz),
        Channel(ch_out_sz)
    )
end

#####

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

    Trajectory(c::C, s::S, t::T) where {C,S,T} = new{C,S,T}(c, s, t)

    function Trajectory(container::C, sampler::S, controler::T) where {C,S,T<:AsyncInsertSampleRatioControler}
        t = Threads.@spawn while true
            for msg in controler.in
                if msg.f === Base.push! || msg.f === Base.append!
                    n_pre = length(trajectory)
                    msg.f(trajectory, msg.args...; msg.kw...)
                    n_post = length(trajectory)
                    controler.n_inserted += n_post - n_pre
                else
                    msg.f(trajectory, msg.args...; msg.kw...)
                end

                if controler.n_inserted >= controler.threshold
                    if controler.n_sampled < (controler.n_inserted - controler.threshold) * controler.ratio
                        batch = sample(sampler, container)
                        put!(controler.ch_out, batch)
                        controler.n_sampled += 1
                    end
                end
            end
        end

        bind(controler.in, t)
        bind(controler.out, t)
        new{C,S,T}(container, sampler, controler)
    end
end


Base.push!(t::Trajectory; kw...) = push!(t, values(kw))

function Base.push!(t::Trajectory, x)
    n_pre = length(t.container)
    push!(t.container, x)
    n_post = length(t.container)
    on_insert!(t.controler, n_post - n_pre)
end

struct CallMsg
    f::Any
    args::Tuple
    kw::Any
end

Base.push!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioControler}, args...; kw...) = put!(t.controler.ch_in, CallMsg(Base.push!, args, kw))
Base.append!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioControler}, args...; kw...) = append!(t.controler.ch_in, CallMsg(Base.push!, args, kw))

Base.append!(t::Trajectory; kw...) = append!(t, values(kw))

function Base.append!(t::Trajectory, x)
    n_pre = length(t.container)
    append!(t.container, x)
    n_post = length(t.container)
    on_insert!(t.controler, n_post - n_pre)
end

function Base.take!(t::Trajectory)
    res = on_sample!(t.controler)
    if isnothing(res)
        nothing
    else
        sample(t.sampler, t.container)
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

Base.iterate(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioControler}, args...) = iterate(t.controler.ch_out, args...)
Base.take!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioControler}) = take!(t.controler.ch_out)