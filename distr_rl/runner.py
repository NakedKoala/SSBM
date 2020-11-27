# handles communication for the A3C runner
# first, waits for the trainer to give it a model
# then continually:
#   - generates experiences with the current model
#   - every number of frames, sends experiences to the trainer
#   - every number of frames, checks for model update (might not occur)

from communication import *

import time

def _check_model_updates(model, param_socket, block=False):
    new_model_state = param_socket.recv(block=block)
    print("runner received:", new_model_state)
    if new_model_state is not None:
        # model.load_state_dict(new_model_state)
        pass

def run_loop(
    model,                      # uninitialized actor-critic model
    trainer_ip,                 # IP address of the trainer
    exp_port,                   # port for experience socket
    param_port,                 # port for parameters socket
    send_exp_every=60,          # frequency (frames) of sending experiences
    check_model_upd_every=10,   # frequency (frames) of checking for model updates
    max_episodes=None,          # maximum number of episodes
):
    exp_socket = PushSocket(trainer_ip, exp_port)
    param_socket = SubSocket(trainer_ip, param_port)

    cur_eps = 0
    experiences = []
    while True:
        if max_episodes and cur_eps >= max_episodes:
            break

        # wait for latest model before starting the episode
        _check_model_updates(model, param_socket, block=True)

        cur_frame = 1
        # run the entire episode to completion
        while True:
            time.sleep(1)
            # run the episode here

            experiences.append(cur_frame)

            if send_exp_every and cur_frame % send_exp_every == 0:
                print("runner sending:", experiences)
                exp_socket.send(experiences)
                experiences.clear()

            if check_model_upd_every and cur_frame % check_model_upd_every == 0:
                # don't block
                _check_model_updates(model, param_socket)

            cur_frame += 1


if __name__ == '__main__':
    run_loop(
        model=None,
        trainer_ip=None,
        exp_port=50000,
        param_port=50001,
        send_exp_every=10,
        check_model_upd_every=5,
        max_episodes=None,
    )
