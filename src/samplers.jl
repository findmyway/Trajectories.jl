using Random

abstract type AbstractSampler end

#####
# BatchSampler
#####

export BatchSampler
struct BatchSampler{names} <: AbstractSampler
    batch_size::Int
    rng::Random.AbstractRNG
end

"""
    BatchSampler{names}(;batch_size, rng=Random.GLOBAL_RNG)
    BatchSampler{names}(batch_size ;rng=Random.GLOBAL_RNG)

Uniformly sample a batch of examples for each trace specified in `names`. 
By default, all the traces will be sampled.

See also [`sample`](@ref).
"""
BatchSampler(batch_size; kw...) = BatchSampler(; batch_size=batch_size, kw...)
BatchSampler(; kw...) = BatchSampler{nothing}(; kw...)
BatchSampler{names}(batch_size; kw...) where {names} = BatchSampler{names}(; batch_size=batch_size, kw...)
BatchSampler{names}(; batch_size, rng=Random.GLOBAL_RNG) where {names} = BatchSampler{names}(batch_size, rng)

sample(s::BatchSampler{nothing}, t::AbstractTraces) = sample(s, t, keys(t))
sample(s::BatchSampler{names}, t::AbstractTraces) where {names} = sample(s, t, names)

function sample(s::BatchSampler, t::AbstractTraces, names)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    NamedTuple{names}(t[x][inds] for x in names)
end

#####
# MetaSampler
#####

export MetaSampler

"""
    MetaSampler(::NamedTuple)

Wraps a NamedTuple containing multiple samplers. When sampled, returns a named tuple with a 
batch from each sampler.
Used internally for algorithms that sample multiple times per epoch.

# Example
```
MetaSampler(policy = BatchSampler(10), critic = BatchSampler(100))
```
"""
struct MetaSampler{names,T} <: AbstractSampler
    samplers::NamedTuple{names,T}
end

MetaSampler(; kw...) = MetaSampler(NamedTuple(kw))

sample(s::MetaSampler, t) = map(x -> sample(x, t), s.samplers)

#####
# MultiBatchSampler
#####

export MultiBatchSampler

"""
    MultiBatchSampler(sampler, n)

Wraps a sampler. When sampled, will sample n batches using sampler. Useful in combination 
with MetaSampler to allow different sampling rates between samplers.

# Example
```
MetaSampler(policy = MultiBatchSampler(BatchSampler(10), 3), 
            critic = MultiBatchSampler(BatchSampler(100), 5))
```
"""
struct MultiBatchSampler{S<:AbstractSampler} <: AbstractSampler
    sampler::S
    n::Int
end

sample(m::MultiBatchSampler, t) = [sample(m.sampler, t) for _ in 1:m.n]

#####
# NStepBatchSampler
#####

export NStepBatchSampler

Base.@kwdef mutable struct NStepBatchSampler{traces}
    n::Int # !!! n starts from 1
    γ::Float32
    batch_size::Int = 32
    stack_size::Union{Nothing,Int} = nothing
    rng::Any = Random.GLOBAL_RNG
end

select_last_dim(xs::AbstractArray{T,N}, inds) where {T,N} = @views xs[ntuple(_ -> (:), Val(N - 1))..., inds]
select_last_frame(xs::AbstractArray{T,N}) where {T,N} = select_last_dim(xs, size(xs, N))

consecutive_view(cb, inds; n_stack=nothing, n_horizon=nothing) = consecutive_view(cb, inds, n_stack, n_horizon)
consecutive_view(cb, inds, ::Nothing, ::Nothing) = select_last_dim(cb, inds)
consecutive_view(cb, inds, n_stack::Int, ::Nothing) = select_last_dim(cb, [x + i for i in -n_stack+1:0, x in inds])
consecutive_view(cb, inds, ::Nothing, n_horizon::Int) = select_last_dim(cb, [x + j for j in 0:n_horizon-1, x in inds])
consecutive_view(cb, inds, n_stack::Int, n_horizon::Int) = select_last_dim(cb, [x + i + j for i in -n_stack+1:0, j in 0:n_horizon-1, x in inds])

function sample(s::NStepBatchSampler{names}, ts) where {names}
    valid_range = isnothing(s.stack_size) ? (1:(length(ts)-s.n+1)) : (s.stack_size:(length(ts)-s.n+1))# think about the exteme case where s.stack_size == 1 and s.n == 1
    inds = rand(s.rng, valid_range, s.batch_size)
    sample(s, ts, Val(names), inds)
end

function sample(s::NStepBatchSampler, ts, ::Val{SSART}, inds)
    s = consecutive_view(ts[:state], inds; n_stack=s.stack_size)
    s′ = consecutive_view(ts[:next_state], inds .+ (s.n - 1); n_stack=s.stack_size)
    a = consecutive_view(ts[:action], inds)
    t_horizon = consecutive_view(ts[:terminal], inds; n_horizon=s.n)
    r_horizon = consecutive_view(ts[:reward], inds; n_horizon=s.n)

    @assert ndims(t_horizon) == 2
    t = any(t_horizon, dims=1)

    @assert ndims(r_horizon) == 2
    r = map(eachcol(r_horizon), eachcol(t_horizon)) do r⃗, t⃗
        foldr((init, (rr, tt)) -> rr + f.γ * init * (1 - tt), zip(r⃗, t⃗); init=0.0f0)
    end

    NamedTuple{names}(s, s′, a, r, t)
end

function sample(s::NStepBatchSampler, ts, ::Val{SSLART}, inds)
    s, s′, a, r, t = sample(s, ts, Val(SSART), inds),
    l = consecutive_view(ts[:legal_actions_mask], inds)
    NamedTuple{SSLART}(s, s′, l, a, r, t)
end