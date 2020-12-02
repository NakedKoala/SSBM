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
# represents a sequence of states/actions/rewards that a runner observed.
ProcExpPayload = namedtuple(
    'ProcExpPayload',
    [
        'states_input',     # delayed windows of full game states
        'action_input',     # recent model actions between delay and current time
        'final_state',      # same as in ExpPayload
        'actions',          # same as in ExpPayload
        'rewards',          # same as in ExpPayload
    ]
)

AdversaryParamPayload = namedtuple(
    'AdversaryParamPayload',
    [
        'state_dict',       # parameters for the model to use
    ]
)

AdversaryInputPayload = namedtuple(
    'AdversaryInputPayload',
    [
        'state',            # state input for the adversary to process
        'behavior',         # action head behavior to use
    ]
)
