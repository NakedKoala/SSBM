# functions to create payloads for consistent formats on create.

from collections import namedtuple

# payload that runner sends to experience processor.
# use a single payload that contains all required data.
ExpPayload = namedtuple(
    'ExpPayload',
    [
        'init_states',  # list of window_size - 1 game states used to determine the first action
        'states',       # list of delayed game states (includes player inputs)
        'actions',      # model actions taken when given a delayed game state (so states and actions are aligned)
        'final_state',  # final state of this payload (1 more state frame, including the action taken); if None, then env finished.
        'rewards'       # env rewards - aligned with states/actions.
    ]
)

# payload that experience processor sends to the trainer.
# represents a single training example.
# use a list of these payloads - one payload per frame/training example.
ProcExpPayload = namedtuple(
    'ProcExpPayload',
    [
        'stale_states',     # delayed window of full game states
        # 'recent_inputs',    # recent model actions between delay and current time
        'next_state',       # next state after action was taken - if None, then assume done.
        'action',           # action for next frame model decided to take
        'reward',           # immediate reward (or maybe reward from `frame_delay` frames ago?) for executing the action
    ]
)
