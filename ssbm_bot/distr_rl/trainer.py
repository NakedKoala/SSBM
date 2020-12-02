# handles communication for the A3C trainer
# takes in an actor-critic model to train
# continually:
#   - requests workers for training data
#   - trains the model on the received data
#   - push parameters to workers

from .a3c import A3CTrainer
from .communication import *
from .input_manager import InputManager
from .payloads import ProcExpPayload
from ..data.common_parsing_logic import align

import multiprocessing as mp
import time
import sys

import torch

def process_exps_loop(
    exp_port,
    return_port,
    window_size,
    frame_delay,
):
    exp_socket = PullSocket(None, exp_port, bind=True)
    process_exp_socket = PushSocket(None, return_port)

    # get example from the runners, process it,
    # then push to the trainer.
    while True:
        experiences = exp_socket.recv()
        # print("process_exp received payload")

        if len(state_queue) != window_size-1:
            sys.stderr.write("process_exp received payload with initial state "
                             "length != window_size-1. Skipping.\n")
            continue

        if not (
            len(experiences.states) == len(experiences.actions) == len(experiences.rewards)
        ):
            sys.stderr.write("process_exp received payload with unequal state/action/reward "
                             "lengths. Skipping.\n")
            continue

        input_manager = InputManager(window_size, frame_delay)
        for state in experiences.init_states:
            input_manager.get(state, None)

        states_inputs = []
        action_inputs = []
        for i in range(len(experiences.states)):
            if i > 0:
                state_t, action_t = input_manager.get(experiences.states[i], experiences.actions[i-1])
            else:
                state_t, action_t = input_manager.get(experiences.states[i], None)
            states_inputs.append(state_t)
            action_inputs.append(action_t)

        payload = ProcExpPayload(
            states_input=torch.stack(states_inputs),
            action_input=torch.stack(action_inputs),
            final_state=experiences.final_state,
            actions=torch.cat(experiences.actions, dim=0),
            rewards=experiences.rewards
        )
        process_exp_socket.send(payload)


def _save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


_MOVING_AVG_FACTOR = 0.99
def train_loop(
    trainer,                # A3C trainer with model
    optimizer,              # optimizer to use
    exp_port,               # port for experience socket
    param_port,             # port for parameters socket
    exp_process_port,       # port for experience processing socket
    window_size,            # window size of full game state
    frame_delay,            # frame delay between state occurrence and observation
    save_path,              # location to save weights
    send_param_every=10,    # frequency of sending parameters to runners, in seconds.
    max_steps=None,         # maximum number of batches to train on
    output_loss_every=None, # frequency of outputting loss, in steps.
    save_every=None,        # frequecy of saving model weights, in steps.
    spawn_proc=True,        # should the trainer spawn the exp processing process?
):
    process_exp_socket = PullSocket(None, exp_process_port, bind=True)
    param_socket = PubSocket(None, param_port, bind=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using device:", device)
    trainer.model.to(device)

    if spawn_proc:
        process_exps_proc = mp.Process(target=process_exps_loop, args=(
            exp_port,
            exp_process_port,
            window_size,
            frame_delay,
        ), daemon=True)
        process_exps_proc.start()

    # number of training steps done
    cur_step = 1
    avg_loss = None
    # time since last parameter send
    last_param_send_time = time.perf_counter()
    while True:
        if max_steps and cur_step >= max_steps:
            break

        # get training data
        # don't block - runner might be waiting for model params.
        exp = process_exp_socket.recv(block=False)
        if exp is None:
            # avoid excessive busy waiting
            time.sleep(0.5)
        else:
            # print("training", cur_step)

            states_t = exp.states_input.to(device)
            actions_t = exp.actions.to(device)
            rewards = exp.rewards
            next_state = exp.final_state
            if next_state is not None:
                next_state = next_state.to(device)
            done = (next_state is None)

            loss = trainer.optimize(optimizer, done, next_state, states_t, actions_t, rewards, 0.9)
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = _MOVING_AVG_FACTOR * avg_loss + (1.0 - _MOVING_AVG_FACTOR) * loss

            if output_loss_every and cur_step % output_loss_every == 0:
                print("step:", cur_step, "avg loss:", avg_loss)

            # checkpoint
            if save_every and cur_step % save_every == 0:
                _save_model(trainer.model, save_path)

            cur_step += 1

        # send new parameters to runners
        if send_param_every is not None and time.perf_counter() >= \
                last_param_send_time + send_param_every:
            # print("send param update")
            trainer.model.to('cpu')
            param_socket.send(trainer.model.state_dict())
            trainer.model.to(device)
            last_param_send_time = time.perf_counter()

    _save_model(trainer.model, save_path)

    # daemonic process_exps_proc expected to exit
