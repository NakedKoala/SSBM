import gym
import torch

from .environment import BaseEnvironment

class CartPoleEnvironment(BaseEnvironment):
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        pass

    def reset(self):
        state = torch.from_numpy(self.env.reset()).float()
        return state, state

    def step(self, action):
        agent_action = action[0]
        state, reward, done, _ = self.env.step(agent_action[0].item())
        state = torch.from_numpy(state).float()
        return state, state, reward, done
