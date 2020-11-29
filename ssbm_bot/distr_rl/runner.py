# handles communication for the A3C runner
# first, waits for the trainer to give it a model
# then continually:
#   - generates experiences with the current model
#   - every number of frames, sends experiences to the trainer
#   - every number of frames, checks for model update (might not occur)

from .communication import *
from .payloads import ExpPayload

from ..data.common_parsing_logic import align
from ..model.action_head import ActionHead

from collections import deque
from itertools import islice
import time

def _check_model_updates(model, param_socket, block=False):
    new_model_state = param_socket.recv(block=block)
    if new_model_state is not None:
        # print("runner received params")
        model.load_state_dict(new_model_state)


def run_loop(
    trainer,                    # A3CTrainer with uninitialized actor-critic model
    environment,                # SSBM environment to train on
    trainer_ip,                 # IP address of the trainer
    exp_port,                   # port for experience socket
    param_port,                 # port for parameters socket
    window_size,
    frame_delay,
    send_exp_every,             # frequency (frames) of sending experiences
    check_model_upd_every=10,   # frequency (seconds) of checking for model updates
    output_reward_every=None,   # frequency (frames) of outputting reward
    max_episodes=None,          # maximum number of episodes
):
    # NOTE frame_delay is currently unused, but once we send recent
    # actions to the model as part of input, we will need frame_delay

    exp_socket = PushSocket(trainer_ip, exp_port)
    param_socket = SubSocket(trainer_ip, param_port)

    cur_eps = 1
    last_model_upd_check = time.perf_counter()
    while True:
        if max_episodes and cur_eps >= max_episodes:
            break

        # wait for latest model before starting the episode
        _check_model_updates(trainer.model, param_socket, block=True)

        cur_frame = 1
        stale_state_buffer = deque()
        stale_state_align = deque()
        actions_buffer = deque()
        rewards_buffer = deque()
        episode_reward = 0.0

        cur_state = environment.reset()
        # wrap in batch dimension
        cur_state_t = align(stale_state_align, window_size, cur_state).unsqueeze(dim=0)

        # add the first window_size-1 zero states for payload init_state.
        # also include the first real state.
        stale_state_buffer.extend(stale_state_align)

        # run the entire episode to completion
        print("runner start episode", cur_eps)
        while True:
            action = trainer.choose_action(cur_state_t, ActionHead.DEFAULT)
            actions_buffer.append(action)
            cur_state, reward, done = environment.step(action)
            episode_reward += reward
            rewards_buffer.append(reward)

            if output_reward_every and cur_frame % output_reward_every == 0:
                print("episode:", cur_eps, "frame:", cur_frame, "reward:", episode_reward)

            # state tensor for next frame
            cur_state_t = align(stale_state_align, window_size, cur_state).unsqueeze(dim=0)

            if done or cur_frame % send_exp_every == 0:
                stale_states = list(islice(
                    stale_state_buffer, window_size-1, len(stale_state_buffer)
                ))
                init_states = list(islice(stale_state_buffer, window_size-1))
                actions = list(actions_buffer)
                rewards = list(rewards_buffer)

                if done:
                    final_state = None
                else:
                    final_state = cur_state

                exp_payload = ExpPayload(
                    init_states=init_states,
                    states=stale_states,
                    actions=actions,
                    final_state=final_state,
                    rewards=rewards
                )

                # print("runner sending experiences")
                exp_socket.send(exp_payload)

            # add next state after possibly sending data
            stale_state_buffer.append(cur_state)

            if check_model_upd_every and time.perf_counter() >= \
                    last_model_upd_check + check_model_upd_every:
                # don't block
                _check_model_updates(trainer.model, param_socket)
                last_model_upd_check = time.perf_counter()

            if done:
                break

            while len(stale_state_buffer) >= send_exp_every + window_size:
                # send_exp_every + (window_size-1) + 1 (for next frame)
                stale_state_buffer.popleft()
            while len(actions_buffer) >= send_exp_every:
                actions_buffer.popleft()
            while len(rewards_buffer) >= send_exp_every:
                rewards_buffer.popleft()

            cur_frame += 1

        print("runner done episode", cur_eps)
        print("running got reward", episode_reward)

        cur_eps += 1
