from .environment import BaseEnvironment

class SSBMEnvironment(BaseEnvironment):
    def __init__(self, frame_delay, device):
        self.frame_delay = frame_delay
        self.device = device

    def reset(self):
        pass

    def step(self, action):
        agent_action = action[0]
        adversary_action = action[1]
