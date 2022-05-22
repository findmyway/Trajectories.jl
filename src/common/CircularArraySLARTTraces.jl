export CircularArraySLARTTraces

function CircularArraySLARTTraces(;
    capacity::Int,
    state=Int => (),
    legal_actions_mask=Bool => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    legal_actions_mask_eltype, legal_actions_mask_size = legal_actions_mask
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{(:state, :next_state)}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
    MultiplexTraces{(:legal_actions_mask, :next_legal_actions_mask)}(CircularArrayBuffer{legal_actions_mask_eltype}(legal_actions_mask_size..., capacity + 1)) +
    MultiplexTraces{(:action, :next_action)}(CircularArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
    Traces(
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end