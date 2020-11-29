import gym
import torch

from .environment import BaseEnvironment

class CartPoleEnvironment(BaseEnvironment):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        pass

    def reset(self):
        return torch.from_numpy(self.env.reset()).float()

    def step(self, action):
        state, reward, done, _ = self.env.step(action[0].item())
        return torch.from_numpy(state).float(), reward, done
