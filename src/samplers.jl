export BatchSampler

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
BatchSampler(batch_size; rng=Random.GLOBAL_RNG, transformer=identity) = BatchSampler(batch_size, rng, identity)
