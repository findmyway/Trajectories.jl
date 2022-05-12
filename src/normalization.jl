import OnlineStats: OnlineStats, Group, Moments, fit!, OnlineStat, Weight, EqualWeight, mean, std
export state_normalizer, reward_normalizer, NormalizedTrajectory, Normalizer
using MacroTools

"""
    Normalizer(::OnlineStat)

Wraps an OnlineStat to be used by a [`NormalizedTrajectory`](@ref).
"""
struct Normalizer{OS<:OnlineStat}
    os::OS
end

MacroTools.@forward Normalizer.os OnlineStats.mean, OnlineStats.std, Base.iterate, normalize!, Base.length

function OnlineStats.fit!(n::Normalizer, y)
    for yi in y
        fit!(n, yi)
    end
    n
end

function OnlineStats.fit!(n::Normalizer, data::AbstractArray)
    fit!(n.os, vec(data))
    n
end

function OnlineStats.fit!(n::Normalizer, data::Number)
    fit!(n.os, data)
    n
end

"""
    reward_normalizer(;weights = OnlineStats.EqualWeight())

Returns preconfigured normalizer for scalar rewards. By default, all rewards have equal weights.
See the [OnlineStats documentation](https://joshday.github.io/OnlineStats.jl/stable/weights/) to use variants such as exponential weights.
"""
reward_normalizer(; weight::Weight = EqualWeight()) = Normalizer(Moments(weight = weight))

"""
    state_normalizer([state_size::Tuple{Int}]; weights = OnlineStats.EqualWeight())

Returns preconfigured normalizer for scalar or array states. 
For Array states, state_size is a tuple containing the dimension sizes of a state. E.g. `(10,)` for a 10-elements vector, or `(252,252)` for a square image.
For scalar states, do not provide a state_size information.
By default, all states have equal weights.
See the [OnlineStats documentation](https://joshday.github.io/OnlineStats.jl/stable/weights/) to use variants such as exponential weights to favor the most recent observations.
"""
state_normalizer(; weight::Weight = EqualWeight()) = Normalizer(Moments(weight = weight))

state_normalizer(state_size; weight::Weight = EqualWeight()) = Normalizer(Group([Moments(weight = weight) for _ in 1:prod(state_size)]))


"""
    NormalizedTrajectory(trajectory, normalizer::Dict{Symbol, Normalizer})
    NormalizedTrajectory(trajectory, normalizer::Pair{Symbol, Normalizer}...)

Wraps a `Trajectory` and a [`Normalizer`](@ref). When pushing new elements of `:trace` to trajectory, a `NormalizedTrajectory` will first update `normalizer[:trace]`, an online estimate of the mean and variance of :trace.
When sampling `:trace` from a normalized trajectory, it will first normalize the samples using `normalizer[:trace]`, if `:trace` is in the keys of `normalizer`, according to its current estimate.

Use a `Normalizer(Moments())` estimate for scalar traces, `Normalizer(Group([Moments() for el in trace]))` for Array estimates. 
Predefined constructors are provide for scalar rewards (see [`reward_normalizer`](@ref)) and states (see [`state_normalizer`](@ref))

#Example
NormalizedTrajectory(
    my_trajectory,
    :state => state_normalizer((5,5)),
    :reward => reward_normalizer(weight = OnlineStats.ExponentialWeight)
)

"""
struct NormalizedTrajectory{T, N}
    trajectory::T
    normalizer::Dict{Symbol, N}
end 

NormalizedTrajectory(traj::Trajectory, pairs::Pair{<:Symbol, <:Normalizer}...) = NormalizedTrajectory(traj, Dict(pairs))

function Base.push!(nt::NormalizedTrajectory; x...)
    for (key, value) in x
        if key in keys(nt.normalizer)
            fit!(nt.normalizer[key], value)
        end
    end
    push!(nt.trajectory; x...)
end

function Base.append!(nt::NormalizedTrajectory; x...)
    for (key, value) in x
        if key in keys(nt.normalizer)
            fit!(nt.normalizer[key], value)
        end
    end
    append!(nt.trajectory; x...)
end

"""
    normalize!(os::Moments, x)

Given an Moments estimate of x, a scalar trace or a vector of scalar traces,
normalizes x to zero mean, and unit variance. Works elementwise given a vector. 
"""
function normalize!( os::Moments, x::Number)
    m, s = mean(os), std(os)
    x -= m
    x /= s
end

function normalize!(os::Moments, x::AbstractVector)
    m, s = mean(os), std(os)
    x .-= m
    x ./= s
end


"""
    normalize!(os::Group{<:AbstractVector{<:Moments}}, x)

Given an os::Group{<:Tuple{Moments}}, that is, a multivariate estimator of the moments of each element of x,
normalizes each element of x to zero mean, and unit variance. Treats the last dimension as a batch dimension if `ndims(x) >= 2`.
"""
function normalize!(os::Group{<:AbstractVector{<:Moments}}, x::AbstractVector)
    m = [mean(stat) for stat in os]
    s = [std(stat) for stat in os]
    x .-= m
    x ./= s
end

function normalize!(os::Group{<:AbstractVector{<:Moments}}, x::AbstractArray)
    for slice in eachslice(x, dims = ndims(x))
        normalize!(os, vec(slice))
    end
end

function Base.take!(nt::NormalizedTrajectory)
    x = take!(nt.trajectory)
    if isnothing(x)
        x
    else
        for key in keys(nt.normalizer)
            normalize!(nt.normalizer[key], x[key])
        end
    end
    x
end
