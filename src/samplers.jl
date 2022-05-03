export BatchSampler

using Random

struct BatchSampler
    batch_size::Int
    rng::Random.AbstractRNG
end

"""
    BatchSampler(batch_size; rng=Random.GLOBAL_RNG)

Uniformly sample a batch of examples for each trace.
"""
BatchSampler(batch_size; rng=Random.GLOBAL_RNG) = BatchSampler(batch_size, rng)

"""
    MinLengthSampler(min_length, sampler)

A wrapper of `sampler`. When the `length` of traces is less than `min_length`,
`nothing` is returned. Otherwise, apply the `sampler` to the traces.
"""
struct MinLengthSampler{S}
    min_length::Int
    sampler::S
end

function Random.rand(s::MinLengthSampler, t)
    if length(t) < s.min_length
        nothing
    else
        rand(s.sampler, t)
    end
end