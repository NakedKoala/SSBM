# handles communication for the A3C trainer
# takes in an actor-critic model to train
# continually:
#   - requests workers for training data
#   - trains the model on the received data
#   - push parameters to workers

from .a3c import A3CTrainer
from .communication import *
from .payloads import ProcExpPayload
from ..data.common_parsing_logic import align
from ..data.infra_adaptor import FrameContext

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

        if len(experiences.init_states) != window_size-1:
            sys.stderr.write("process_exp received payload with initial state "
                             "length != window_size-1. Skipping.\n")
            continue

        if not (
            len(experiences.states) == len(experiences.actions) == len(experiences.rewards)
        ):
            sys.stderr.write("process_exp received payload with unequal state/action/reward "
                             "lengths. Skipping.\n")
            continue

        last_action = None
        frame_ctx = FrameContext(window_size, frame_delay)
        for state, action in zip(experiences.init_states, experiences.init_actions):
            frame_ctx.push_tensor(state, last_action)
            last_action = action[0]

        states_inputs = []
        action_inputs = []
        for i in range(len(experiences.states)):
            state_t, action_t = frame_ctx.push_tensor(experiences.states[i], last_action)
            last_action = experiences.actions[i][0]
            states_inputs.append(state_t)
            action_inputs.append(action_t)

        if experiences.final_state is None:
            final_state_t, final_action_t = None, None
        else:
            final_state_t, final_action_t = frame_ctx.push_tensor(experiences.final_state, last_action)

        if frame_delay == 0:
            action_input_t = None
        else:
            action_input_t = torch.cat(action_inputs, dim=0)

        payload = ProcExpPayload(
            states_input=torch.cat(states_inputs, dim=0),
            action_input=action_input_t,
            final_state=final_state_t,
            final_action=final_action_t,
            actions=torch.cat(experiences.actions, dim=0),
            rewards=experiences.rewards
        )
        process_exp_socket.send(payload)


def _save_model(model, ckpt_path, step):
    ckpt_filename = ckpt_path + '_' + str(step) + '.pt'
    torch.save(model.state_dict(), ckpt_filename)


_MOVING_AVG_FACTOR = 0.99
def train_loop(
    trainer,                # A3C trainer with model
    optimizer,              # optimizer to use
    gamma,                  # gamma for RL
    exp_port,               # port for experience socket
    param_port,             # port for parameters socket
    exp_process_port,       # port for experience processing socket
    window_size,            # window size of full game state
    frame_delay,            # frame delay between state occurrence and observation
    ckpt_path,              # location to save weights
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
            recent_actions_t = exp.action_input
            if recent_actions_t is not None:
                recent_actions_t = recent_actions_t.to(device)
            input_t = (states_t, recent_actions_t)

            if exp.final_state is None:
                done = True
                next_state = (None, None)
            else:
                done = False
                if exp.final_action is None:
                    next_state = (exp.final_state.to(device), None)
                else:
                    next_state = (exp.final_state.to(device), exp.final_action.to(device))

            actions_t = exp.actions.to(device)
            rewards = exp.rewards

            loss = trainer.optimize(optimizer, done, next_state, input_t, actions_t, rewards, gamma)
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = _MOVING_AVG_FACTOR * avg_loss + (1.0 - _MOVING_AVG_FACTOR) * loss

            if output_loss_every and cur_step % output_loss_every == 0:
                print("step:", cur_step, "avg loss:", avg_loss)

            # checkpoint
            if save_every and cur_step % save_every == 0:
                _save_model(trainer.model, ckpt_path, cur_step)

            cur_step += 1

        # send new parameters to runners
        if send_param_every is not None and time.perf_counter() >= \
                last_param_send_time + send_param_every:
            # print("send param update")
            trainer.model.to('cpu')
            param_socket.send(trainer.model.state_dict())
            trainer.model.to(device)
            last_param_send_time = time.perf_counter()

    _save_model(trainer.model, ckpt_path, cur_step)

    # daemonic process_exps_proc expected to exit
