export Trajectory

using Base.Threads


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
            for msg in controler.ch_in
                if msg.f === Base.push! || msg.f === Base.append!
                    n_pre = length(container)
                    msg.f(container, msg.args...; msg.kw...)
                    n_post = length(container)
                    controler.n_inserted += n_post - n_pre
                else
                    msg.f(container, msg.args...; msg.kw...)
                end

                if controler.n_inserted >= controler.threshold
                    if controler.n_sampled <= (controler.n_inserted - controler.threshold) * controler.ratio
                        batch = sample(sampler, container)
                        put!(controler.ch_out, batch)
                        controler.n_sampled += 1
                    end
                end
            end
        end

        bind(controler.ch_in, t)
        bind(controler.ch_out, t)
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
Base.append!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioControler}, args...; kw...) = put!(t.controler.ch_in, CallMsg(Base.append!, args, kw))

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