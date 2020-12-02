# handles full game state and recent action input alignment generation
# also handles conversion of action output (categories) back into input

from ..data.common_parsing_logic import align
from .. import controller_indices as c_idx

from collections import deque

import torch

class InputManager(object):
    def __init__(self, window_size, frame_delay):
        self.window_size = window_size
        self.frame_delay = frame_delay
        self.state_queue = deque()
        self.action_queue = deque()

    def get(self, cur_state, last_action):
        cur_state_t = align(self.state_queue, self.window_size, cur_state).unsqueeze(dim=0)
        if self.frame_delay > 0:
            action_t = torch.zeros(7).float()
            if last_action is not None:
                stick_x, stick_y = c_idx.stick.to_stick(*last_action[1:4])
                cstick_x, cstick_y = c_idx.stick.to_stick(*last_action[4:7])
                trigger = c_idx.trigger.to_trigger(last_action[7])
                # recent action format is:
                # buttons, stick x, stick y, cstick x, cstick y, trigger x, trigger y
                action_t = torch.Tensor([last_action[0], stick_x, stick_y, cstick_x, cstick_y, trigger, 0.0]).float()
            action_align_t = align(self.action_queue, self.frame_delay, action_t).unsqueeze(dim=0)
        else:
            action_align_t = None

        return cur_state_t, action_align_t
