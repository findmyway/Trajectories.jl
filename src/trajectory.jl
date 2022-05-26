export Trajectory, TrajectoryStyle, SyncTrajectoryStyle, AsyncTrajectoryStyle

using Base.Threads

struct AsyncTrajectoryStyle end
struct SyncTrajectoryStyle end

"""
    Trajectory(container, sampler, controller)

The `container` is used to store experiences. Common ones are [`Traces`](@ref)
or [`Episodes`](@ref). The `sampler` is used to sample experience batches from
the `container`. The `controller` controls whether it is time to sample a batch
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
    controller::T = InsertSampleRatioController()

    Trajectory(c::C, s::S, t::T) where {C,S,T} = new{C,S,T}(c, s, t)

    function Trajectory(container::C, sampler::S, controller::T) where {C,S,T<:AsyncInsertSampleRatioController}
        t = Threads.@spawn while true
            for msg in controller.ch_in
                if msg.f === Base.push! || msg.f === Base.append!
                    n_pre = length(container)
                    msg.f(container, msg.args...; msg.kw...)
                    n_post = length(container)
                    controller.n_inserted += n_post - n_pre
                else
                    msg.f(container, msg.args...; msg.kw...)
                end

                if controller.n_inserted >= controller.threshold
                    if controller.n_sampled <= (controller.n_inserted - controller.threshold) * controller.ratio
                        batch = sample(sampler, container)
                        put!(controller.ch_out, batch)
                        controller.n_sampled += 1
                    end
                end
            end
        end

        bind(controller.ch_in, t)
        bind(controller.ch_out, t)
        new{C,S,T}(container, sampler, controller)
    end
end

TrajectoryStyle(::Trajectory) = SyncTrajectoryStyle()
TrajectoryStyle(::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}) = AsyncTrajectoryStyle()

Base.bind(::Trajectory, ::Task) = nothing

function Base.bind(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, task)
    bind(t.controler.ch_in, task)
    bind(t.controler.ch_out, task)
end

function Base.push!(t::Trajectory, x)
    n_pre = length(t.container)
    push!(t.container, x)
    n_post = length(t.container)
    on_insert!(t.controller, n_post - n_pre)
end

struct CallMsg
    f::Any
    args::Tuple
    kw::Any
end

Base.push!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, args...; kw...) = put!(t.controller.ch_in, CallMsg(Base.push!, args, kw))
Base.append!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, args...; kw...) = put!(t.controller.ch_in, CallMsg(Base.append!, args, kw))

function Base.append!(t::Trajectory, x)
    n_pre = length(t.container)
    append!(t.container, x)
    n_post = length(t.container)
    on_insert!(t.controller, n_post - n_pre)
end

function Base.take!(t::Trajectory)
    res = on_sample!(t.controller)
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

Base.iterate(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}, args...) = iterate(t.controller.ch_out, args...)
Base.take!(t::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}) = take!(t.controller.ch_out)

Base.IteratorSize(::Trajectory{<:Any,<:Any,<:AsyncInsertSampleRatioController}) = Base.IsInfinite()
Base.IteratorSize(::Trajectory) = Base.SizeUnknown()