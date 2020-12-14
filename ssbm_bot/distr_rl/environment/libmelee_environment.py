from .environment import BaseEnvironment
from ..infrastructure import *
import torch

class LibmeleeEnvironment(BaseEnvironment):
    def __init__(self, frame_delay, iso_path):
        self.agent = None

        self.frame_delay = frame_delay
        self.buffer = []
        self.state_shape = None

        self.done = False

        self.iso_path = iso_path

    # resets the environment and returns an initial state.
    def reset(self):
        if self.agent is not None:
            self.agent.shutdown()
        else:
            self.agent = MeleeAI(action_frequence=None, window_size=60, frame_delay=self.frame_delay, include_opp_input=False, multiAgent=True, model=None, iso_path=self.iso_path)

        self.agent.start()
        cur_state, _, _, _ = self.agent.step()

        self.state_shape = cur_state.shape[1:]

        self.done = False
        return torch.zeros(self.state_shape, device='cpu'), torch.zeros(self.state_shape, device='cpu')

    # executes action immediately and returns delayed state/reward/done.
    def step(self, action):
        self.agent.preform_action(action)

        if not self.done:
            cur_state, adv_state, reward, done = self.agent.step()
            self.done = done
            cur_state = cur_state[0]
            adv_state = adv_state[0]
            self.buffer.append((cur_state, adv_state, reward, done))

        if len(self.buffer) > 0 and (len(self.buffer) > self.frame_delay or self.done):
            return self.buffer.pop(0)

        return torch.zeros(self.state_shape, device='cpu'), torch.zeros(self.state_shape, device='cpu'), 0, False

