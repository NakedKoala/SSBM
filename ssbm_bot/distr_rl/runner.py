# handles communication for the A3C runner
# first, waits for the trainer to give it a model
# then continually:
#   - generates experiences with the current model
#   - every number of frames, sends experiences to the trainer
#   - every number of frames, checks for model update (might not occur)

from .adversary import adversary_loop
from .communication import *
from .payloads import ExpPayload, AdversaryParamPayload, AdversaryInputPayload

from ..data.common_parsing_logic import align
from ..data.infra_adaptor import FrameContext
from ..model.action_head import ActionHead

from collections import deque
from itertools import islice
import multiprocessing as mp
import random
import time

import torch

def _check_model_updates(model, param_socket, block=False):
    new_model_state = param_socket.recv(block=block)
    if new_model_state is not None:
        # print("runner received params")
        model.load_state_dict(new_model_state)


_REWARD_MOVING_AVG_FACTOR = 0.95
def run_loop(
    trainer,                        # A3CTrainer with uninitialized actor-critic model
    environment,                    # SSBM environment to train on
    trainer_ip,                     # IP address of the trainer
    exp_port,                       # port for experience socket
    param_port,                     # port for parameters socket
    adversary_port,                 # port for adversary socket
    window_size,
    frame_delay,
    send_exp_every,                 # frequency (frames) of sending experiences
    check_model_upd_every=10,       # frequency (seconds) of checking for model updates
    output_reward_every=None,       # frequency (frames) of outputting reward
    output_eps_every=None,          # frequency (episodes) of ouputting total stats
    max_episodes=None,              # maximum number of episodes
    max_old_agents=10,              # maximum number of old agents
    save_agent_every=5,             # frequency (episodes) of saving the current agent as an adversary
):
    exp_socket = PushSocket(trainer_ip, exp_port)
    param_socket = SubSocket(trainer_ip, param_port)

    # initialize adversary
    adversary_socket = PairSocket(None, adversary_port, bind=True)
    adversary_proc = mp.Process(
        target=adversary_loop,
        args=(
            adversary_port,
            window_size,
            frame_delay
        ),
        daemon=True
    )
    adversary_proc.start()
    adversary_socket.send(trainer.model, block=True)

    old_agents = []

    cur_eps = 1
    avg_reward = None
    last_model_upd_check = time.perf_counter()
    while True:
        if max_episodes and cur_eps >= max_episodes:
            break

        # wait for latest model before starting the episode
        _check_model_updates(trainer.model, param_socket, block=True)

        if len(old_agents) == 0:
            old_agents.append(trainer.model.state_dict())

        # initialize adversary
        model_dict = old_agents[random.randrange(len(old_agents))]
        adversary_socket.send(AdversaryParamPayload(state_dict=model_dict), block=True)

        cur_frame = 1
        frame_ctx = FrameContext(window_size, frame_delay)
        stale_state_buffer = deque()
        actions_buffer = deque()
        rewards_buffer = deque()
        episode_reward = 0.0

        cur_state, adv_state = environment.reset()

        # add the first window_size-1 zero states for payload init_state.
        # also include the first real state.
        for _ in range(window_size - 1):
            stale_state_buffer.append(torch.zeros_like(cur_state))
            actions_buffer.append(torch.zeros(1, 8))
        stale_state_buffer.append(cur_state)

        # run the entire episode to completion
        if output_eps_every is None or cur_eps % output_eps_every == 0:
            print("runner start episode", cur_eps)
        while True:
            # tell adversary to get action
            adversary_socket.send(AdversaryInputPayload(
                state=adv_state,
                behavior=ActionHead.DEFAULT,
            ), block=True)

            # get runner action
            last_action = None if len(actions_buffer) == 0 else actions_buffer[-1][0]  # remove action batch dimension
            state_t, action_t = frame_ctx.push_tensor(cur_state, last_action)
            action = trainer.choose_action((state_t, action_t), ActionHead.DEFAULT)
            # experience processor expects batch dimension - don't reshape
            actions_buffer.append(action)

            # wait for adversary to finish
            adversary_action = adversary_socket.recv()

            # unwrap batch dimension for environment
            cur_state, adv_state, reward, done = environment.step((action[0], adversary_action[0]))
            episode_reward += reward
            rewards_buffer.append(reward)

            if output_reward_every and cur_frame % output_reward_every == 0:
                print("episode:", cur_eps, "frame:", cur_frame, "reward:", episode_reward)

            if done or cur_frame % send_exp_every == 0:
                stale_states = list(islice(
                    stale_state_buffer, window_size-1, len(stale_state_buffer)
                ))
                init_states = list(islice(stale_state_buffer, window_size-1))
                actions = list(islice(
                    actions_buffer, window_size-1, len(actions_buffer)
                ))
                init_actions = list(islice(actions_buffer, window_size-1))
                rewards = list(rewards_buffer)

                if done:
                    final_state = None
                else:
                    final_state = cur_state

                exp_payload = ExpPayload(
                    init_states=init_states,
                    init_actions=init_actions,
                    states=stale_states,
                    actions=actions,
                    final_state=final_state,
                    rewards=rewards
                )

                # print("runner sending experiences")
                exp_socket.send(exp_payload)

            # add next state after possibly sending data
            stale_state_buffer.append(cur_state)

            if check_model_upd_every is not None and time.perf_counter() >= \
                    last_model_upd_check + check_model_upd_every:
                # don't block
                _check_model_updates(trainer.model, param_socket)
                last_model_upd_check = time.perf_counter()

            if done:
                break

            while len(stale_state_buffer) >= send_exp_every + window_size:
                # send_exp_every + (window_size-1) + 1 (for next frame)
                stale_state_buffer.popleft()
            while len(actions_buffer) >= send_exp_every + (window_size - 1):
                # action added to buffer before sending to trainer
                actions_buffer.popleft()
            while len(rewards_buffer) >= send_exp_every:
                rewards_buffer.popleft()

            cur_frame += 1


        if avg_reward == None:
            avg_reward = episode_reward
        else:
            avg_reward = _REWARD_MOVING_AVG_FACTOR * avg_reward + \
                (1.0 - _REWARD_MOVING_AVG_FACTOR) * episode_reward

        if output_eps_every is None or cur_eps % output_eps_every == 0:
            print("runner done episode", cur_eps)
            print("runner got reward", episode_reward)
            print("runner reward moving average", avg_reward)

        if save_agent_every is not None and cur_eps % save_agent_every == 0:
            if len(old_agents) < max_old_agents:
                old_agents.append(trainer.model.state_dict())
            else:
                replace_idx = random.randrange(1, len(old_agents))
                old_agents[replace_idx] = trainer.model.state_dict()

        cur_eps += 1
