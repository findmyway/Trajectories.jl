using Random

struct BatchSampler
    batch_size::Int
    rng::Random.AbstractRNG
end

BatchSampler(batch_size::Int; rng=Random.GLOBAL_RNG) = BatchSampler(batch_size, rng)

function Random.rand(s::BatchSampler, t::Trajectory)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    map(x -> x[inds], t.traces)
end
