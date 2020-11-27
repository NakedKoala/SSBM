# runs 'A3C' but only trains on 1 node
# takes in an actor-critic model to train
# continually:
#   - requests workers for training data
#   - trains the model on the received data
#   - push parameters to workers

from communication import *

import multiprocessing as mp
import time

import torch

def process_exps_loop(
    exp_port,
    return_port,
    window_size,
    frame_delay,
    exp_batch_size,
):
    exp_socket = PullSocket(None, exp_port, bind=True)
    process_exp_socket = PushSocket(None, return_port)

    # get example from the runners, process it,
    # then push to the trainer.
    while True:
        example = exp_socket.recv()

        print("process_exp received", example)

        # process it here

        process_exp_socket.send(example)

def _save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def train_loop(
    model,                  # actor-critic model
    exp_port,               # port for experience socket
    param_port,             # port for parameters socket
    exp_process_port,       # port for experience processing socket
    window_size,            # window size of full game state
    frame_delay,            # number of frames between window and current frame
    save_path,              # location to save weights
    send_param_every=10,    # frequency of sending parameters to runners
    exp_batch_size=256,     # experience batch size for training
    # exp_pull_lim=16,        # maximum number of experience messages to process at a time
    max_steps=None,         # maximum number of batches to train on
    save_every=None,        # frequecy of saving model weights
    spawn_proc=True,        # should the trainer spawn the exp processing process?
):
    process_exp_socket = PullSocket(None, exp_process_port, bind=True)
    param_socket = PubSocket(None, param_port, bind=True)

    if spawn_proc:
        process_exps_proc = mp.Process(target=process_exps_loop, args=(
            exp_port,
            exp_process_port,
            window_size,
            frame_delay,
            exp_batch_size
        ), daemon=True)
        process_exps_proc.start()

    cur_step = 0
    new_exps = []
    while True:
        if max_steps and cur_step >= max_steps:
            break

        time.sleep(1)
        print("current step", cur_step)

        # get training data
        while len(new_exps) < exp_batch_size:
            exp = process_exp_socket.recv(block=False)
            print("trainer received", exp)
            if exp is None:
                break
            new_exps.extend(exp)

        # train if possible
        print("cur batch size:", len(new_exps))
        if len(new_exps) >= exp_batch_size:
            batch = new_exps[:exp_batch_size]
            print("training:", batch)

            new_exps = new_exps[exp_batch_size:]

        # send new parameters to runners
        if send_param_every and cur_step % send_param_every == 0:
            # param_socket.send(model.state_dict())
            print("send param update", cur_step)
            param_socket.send(cur_step)

        if save_every and cur_step % save_every == 0:
            # _save_model(model, save_path)
            pass

        cur_step += 1

    # _save_model(model, save_path)

    # daemonic process_exps_proc expected to exit

if __name__ == '__main__':
    train_loop(
        model=None,
        exp_port=50000,
        param_port=50001,
        exp_process_port=50002,
        window_size=60,
        frame_delay=15,
        save_path='./test.out',
        send_param_every=3,
        exp_batch_size=16,
        max_steps=None,
        save_every=None,
        spawn_proc=False,
    )
