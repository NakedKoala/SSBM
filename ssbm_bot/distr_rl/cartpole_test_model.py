import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CartPoleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    # absorb kwargs since other models might use it
    def forward(self, x, **kwargs):
        # get rid of timestep dimension
        x = x.squeeze(dim=1)
        x = F.relu(self.affine1(x))
        action_logits = self.action_head(x)
        state_values = self.value_head(x)
        # also choose actions
        m = Categorical(F.softmax(action_logits, dim=1))
        action = m.sample()
        return [action_logits], action.unsqueeze(dim=0), state_values
