import gym
import torch

from .environment import BaseEnvironment

class CartPoleEnvironment(BaseEnvironment):
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        pass

    def reset(self):
        return torch.from_numpy(self.env.reset()).float()

    def step(self, action):
        agent_action = action[0]
        state, reward, done, _ = self.env.step(agent_action[0].item())
        return torch.from_numpy(state).float(), reward, done
