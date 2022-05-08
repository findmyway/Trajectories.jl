using ElasticArrays

const ElasticSARTTraces = Traces{
    SART,
    <:Tuple{
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
    }
}

function ElasticSARTTraces(;
    state=Int => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    Traces(
        state=ElasticArray{state_eltype}(state_size..., 0),
        action=ElasticArray{action_eltype}(action_size..., 0),
        reward=ElasticArray{reward_eltype}(reward_size..., 0),
        terminal=ElasticArray{terminal_eltype}(terminal_size..., 0),
    )
end

function Random.rand(s::BatchSampler, t::ElasticSARTTraces)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    inds′ = inds .+ 1
    (
        state=t[:state][inds],
        action=t[:action][inds],
        reward=t[:reward][inds],
        terminal=t[:terminal][inds],
        next_state=t[:state][inds′],
        next_action=t[:state][inds′]
    ) |> s.transformer
end
