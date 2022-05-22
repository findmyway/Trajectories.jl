export CircularArraySARTTraces

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

    MultiplexTraces{(:state, :next_state)}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
    MultiplexTraces{(:action, :next_action)}(CircularArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
    Traces(
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end
