from collections import deque
import slippi
import torch

from ..data.infra_adaptor import convert_frame_to_input_tensor
from ..data.common_parsing_logic import align

class BaseEnvironment(object):
    def __init__(self, frame_delay):
        raise NotImplementedError()

    # resets the environment and returns an initial state.
    def reset(self):
        raise NotImplementedError()

    # executes action immediately and returns delayed state/reward/done.
    def step(self, action):
        raise NotImplementedError()

# reads environment from slippi file
# not tested yet...
class SLPEnvironment(BaseEnvironment):
    # NOTE convention is: frame_delay == 0 means agent acts ASAP;
    # equivalent to shifting by 1 when forming the dataset.
    def __init__(self, frame_delay, slp_filename, device):
        self.frame_delay = frame_delay
        self.cur_frame = 0
        self.slp = slippi.Game(slp_filename)

        dummy_features = convert_frame_to_input_tensor(self.slp.frames[0], char_id=2, opponent_id=1)
        self.state_shape = dummy_features.shape

        self.frame_buffer = deque()
        self.reward_buffer = deque()

        # not worth making a deque since alignment requires reading all frames
        self.recent_buffer = []

        self.device = device

    def reset(self):
        self.cur_frame = 0
        self.frame_buffer.clear()
        self.recent_buffer.clear()
        return torch.zeros(*self.state_shape, device=self.device)

    # ignore action and pretend the agent input the action specified by the current frame.
    def step(self, action):
        if self.cur_frame < len(self.slp.frames):
            frame = self.slp.frames[self.cur_frame]

            # compute new state for now and save it
            new_state = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)
            recent_buffer.append(new_state)

            # compute reward function for now and save it
            reward = 0
            if self.cur_frame > 0:
                prev_frame = self.slp.frames[self.cur_frame-1]
                for i, port in enumerate(frame.ports):
                    if port:
                        # hardcode captain falcon as agent
                        # other character is not falcon
                        reward_change = -1 if port.leader.post.character == 2 else 1
                        if port.leader.post.stock < prev_frame.ports[i].leader.post.stock:
                            reward += reward_change

            # is the game done?
            if self.cur_frame == len(self.slp.frames)-1:
                reward_buffer.append((reward, True))
            else:
                reward_buffer.append((reward, False))

        # not enough recent states - return 0
        if len(recent_buffer) <= self.frame_delay:
            self.cur_frame += 1
            return torch.zeros(*self.state_shape, device=self.device), 0, False

        # get current delayed state and update recent states buffer
        delayed_state_t = align(self.frame_buffer, self.window_size, self.recent_buffer[0]).to(self.device)
        self.recent_buffer.popleft()

        delayed_reward, delayed_done = reward_buffer.popleft()

        return delayed_state_t, delayed_reward, delayed_done
