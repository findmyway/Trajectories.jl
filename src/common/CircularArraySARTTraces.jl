export CircularArraySARTTraces

const CircularArraySARTTraces = Traces{
    SART,
    <:Tuple{
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer}
    }
}


function CircularArraySARTTraces(;
    capacity::Int,
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
        state=CircularArrayBuffer{state_eltype}(state_size..., capacity + 1),  # !!! state is one step longer
        action=CircularArrayBuffer{action_eltype}(action_size..., capacity + 1),  # !!! action is one step longer
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

function Random.rand(s::BatchSampler, t::CircularArraySARTTraces)
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

function Base.push!(t::CircularArraySARTTraces, x::NamedTuple{SA})
    if length(t[:state]) == length(t[:terminal]) + 1
        pop!(t[:state])
        pop!(t[:action])
    end
    push!(t[:state], x[:state])
    push!(t[:action], x[:action])
end
