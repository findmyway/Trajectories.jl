export BatchSampler, MetaSampler, MultiBatchSampler

using MLUtils: batch

using Random

abstract type AbstractSampler end

struct BatchSampler <: AbstractSampler
    batch_size::Int
    rng::Random.AbstractRNG
    transformer::Any
end

"""
    BatchSampler(batch_size; rng=Random.GLOBAL_RNG, transformer=identity)

Uniformly sample a batch of examples for each trace.

See also [`sample`](@ref).
"""
BatchSampler(batch_size; rng=Random.GLOBAL_RNG, transformer=batch) = BatchSampler(batch_size, rng, transformer)

function sample(s::BatchSampler, t::AbstractTraces)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    map(s.transformer, t[inds])
end

"""
    MetaSampler(::NamedTuple)

Wraps a NamedTuple containing multiple samplers. When sampled, returns a named tuple with a batch from each sampler.
Used internally for algorithms that sample multiple times per epoch.

# Example

MetaSampler(policy = BatchSampler(10), critic = BatchSampler(100))
"""
struct MetaSampler{names,T} <: AbstractSampler
    samplers::NamedTuple{names,T}
end

MetaSampler(; kw...) = MetaSampler(NamedTuple(kw))

sample(s::MetaSampler, t) = map(x -> sample(x, t), s.samplers)


"""
    MultiBatchSampler(sampler, n)

Wraps a sampler. When sampled, will sample n batches using sampler. Useful in combination with MetaSampler to allow different sampling rates between samplers.

# Example

MetaSampler(policy = MultiBatchSampler(BatchSampler(10), 3), critic = MultiBatchSampler(BatchSampler(100), 5))
"""
struct MultiBatchSampler{S<:AbstractSampler} <: AbstractSampler
    sampler::S
    n::Int
end

sample(m::MultiBatchSampler, t) = [sample(m.sampler, t) for _ in 1:m.n]
