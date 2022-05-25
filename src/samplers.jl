export BatchSampler

using MLUtils: batch

using Random

struct BatchSampler
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
