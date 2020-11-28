# handles communication for the A3C runner
# first, waits for the trainer to give it a model
# then continually:
#   - generates experiences with the current model
#   - every number of frames, sends experiences to the trainer
#   - every number of frames, checks for model update (might not occur)

from .a3c import A3CTrainer
from .communication import *

import time

def _check_model_updates(model, param_socket, block=False):
    new_model_state = param_socket.recv(block=block)
    if new_model_state is not None:
        print("runner received params")
        model.load_state_dict(new_model_state)


def run_loop(
    trainer,                    # A3CTrainer with uninitialized actor-critic model
    trainer_ip,                 # IP address of the trainer
    exp_port,                   # port for experience socket
    param_port,                 # port for parameters socket
    window_size,
    frame_delay,
    send_exp_every=60,          # frequency (frames) of sending experiences
    check_model_upd_every=10,   # frequency (seconds) of checking for model updates
    max_episodes=None,          # maximum number of episodes
):
    if send_exp_every < window_size + frame_delay:
        raise AttributeError(
            "runner.py run_loop requires send_exp_every >= window_size + frame_delay"
        )

    exp_socket = PushSocket(trainer_ip, exp_port)
    param_socket = SubSocket(trainer_ip, param_port)

    cur_eps = 0
    last_model_upd_check = time.perf_counter()
    while True:
        if max_episodes and cur_eps >= max_episodes:
            break

        # wait for latest model before starting the episode
        _check_model_updates(trainer.model, param_socket, block=True)

        cur_frame = 1
        init_states = []
        stale_state_buffer = []
        actions_buffer = []
        # run the entire episode to completion
        while True:
            # run the episode here

            experiences.append(cur_frame)

            if send_exp_every and cur_frame % send_exp_every == 0:
                print("runner sending experiences")
                exp_socket.send(experiences)
                experiences.clear()

            if check_model_upd_every and time.perf_counter() >= \
                    last_model_upd_check + check_model_upd_every:
                # don't block
                _check_model_updates(trainer.model, param_socket)
                last_model_upd_check = time.perf_counter()

            cur_frame += 1


if __name__ == '__main__':
    model = SSBM_LSTM_Prob(
        action_embedding_dim = 100, button_embedding_dim = 50, hidden_size = 256,
        num_layers = 1, bidirectional=False, dropout_p=0.2, attention=False
    )
    run_loop(
        model=A3CTrainer(model),
        trainer_ip=None,
        exp_port=50000,
        param_port=50001,
        window_size=60,
        frame_delay=15,
        send_exp_every=300,
        check_model_upd_every=5,
        max_episodes=None,
    )
