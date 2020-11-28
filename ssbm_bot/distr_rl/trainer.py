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

import multiprocessing as mp
import time
import sys

import torch

def process_exps_loop(
    exp_port,
    return_port,
    window_size,
    # frame_delay,
    exp_batch_size,
):
    exp_socket = PullSocket(None, exp_port, bind=True)
    process_exp_socket = PushSocket(None, return_port)

    # get example from the runners, process it,
    # then push to the trainer.
    while True:
        experiences = exp_socket.recv()
        print("process_exp received payload")

        # process experiences into training examples
        state_queue = experiences.init_states
        # input_queue = experiences.init_inputs
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

        payloads = []
        for i in range(len(experiences.states)):
            states_window = align(state_queue, window_size, experiences.states[i])
            if i < len(states)-1:
                next_state = experiences.states[i+1]
            else:
                next_state = experiences.final_state
            payloads.append(
                ProcExpPayload(
                    stale_states=states_window,
                    next_state=next_state,
                    action=experiences.actions[i],
                    reward=experiences.rewards[i]
                )
            )

        process_exp_socket.send(payloads)


def _save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def train_loop(
    trainer,                # A3C trainer with model
    optimizer,              # optimizer to use
    exp_port,               # port for experience socket
    param_port,             # port for parameters socket
    exp_process_port,       # port for experience processing socket
    window_size,            # window size of full game state
    # frame_delay,            # number of frames between window and current frame
    save_path,              # location to save weights
    send_param_every=10,    # frequency of sending parameters to runners, in seconds.
    exp_batch_size=256,     # experience batch size for training
    max_steps=None,         # maximum number of batches to train on
    save_every=None,        # frequecy of saving model weights, in steps.
    spawn_proc=True,        # should the trainer spawn the exp processing process?
):
    process_exp_socket = PullSocket(None, exp_process_port, bind=True)
    param_socket = PubSocket(None, param_port, bind=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if spawn_proc:
        process_exps_proc = mp.Process(target=process_exps_loop, args=(
            exp_port,
            exp_process_port,
            window_size,
            frame_delay,
            exp_batch_size
        ), daemon=True)
        process_exps_proc.start()

    # number of training steps done
    cur_step = 0
    # time since last parameter send
    last_param_send_time = time.perf_counter()
    # unbatched buffer of experiences from the experience processor
    new_exps = []
    while True:
        if max_steps and cur_step >= max_steps:
            break

        # get training data
        imm_fail = False        # immediately failed to get examples?
        while len(new_exps) < exp_batch_size:
            # don't block - runner might be waiting for model params.
            exp = process_exp_socket.recv(block=False)
            print("trainer received", exp)
            if exp is None:
                break
            imm_fail = False
            new_exps.extend(exp)

        # train if possible
        if len(new_exps) >= exp_batch_size:
            batch = new_exps[:exp_batch_size]
            print("training")

            states, inputs, actions, rewards = [], [], [], []
            for payload in batch:
                states.append(payload.stale_states)
                inputs.append(payload.recent_inputs)
                actions.append(payload.action)
                rewards.append(payload.reward)

            states_t = torch.stack(states).to(device)
            actions_t = torch.stack(actions).to(device)
            rewards_t = torch.stack(rewards).to(device)

            trainer.optimize(optimizer, False, next_state, states_t, actions_t, rewards_t, 0.99))

            # checkpoint
            if save_every and cur_step % save_every == 0:
                _save_model(trainer.model, save_path)

            new_exps = new_exps[exp_batch_size:]

            cur_step += 1
        else:
            # avoid excessive busy waiting
            time.sleep(0.5)

        # send new parameters to runners
        if send_param_every and time.perf_counter() >= \
                last_param_send_time + send_param_every:
            print("send param update")
            param_socket.send(trainer.model.state_dict())
            last_param_send_time = time.perf_counter()

    _save_model(trainer.model, save_path)

    # daemonic process_exps_proc expected to exit

if __name__ == '__main__':
    model = SSBM_LSTM_Prob(
        action_embedding_dim = 100, button_embedding_dim = 50, hidden_size = 256,
        num_layers = 1, bidirectional=False, dropout_p=0.2, attention=False
    )
    optim = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_loop(
        model=A3CTrainer(model),
        optimizer=optim
        exp_port=50000,
        param_port=50001,
        exp_process_port=50002,
        window_size=60,
        # frame_delay=15,
        save_path='./test.out',
        send_param_every=3,
        exp_batch_size=16,
        max_steps=None,
        save_every=None,
        spawn_proc=False,
    )
