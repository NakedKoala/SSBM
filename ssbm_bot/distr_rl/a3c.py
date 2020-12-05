import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.action_head import ActionHead

class A3CTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs,
        forced_action=None,
        behavior=ActionHead.DEFAULT
    ):
        return self.model(
            inputs,
            forced_action=forced_action,
            behavior=behavior
        )

    def choose_action(self, inputs, behavior):
        self.eval()
        _, choices, _ = self.forward(inputs, behavior=behavior)
        return choices

    def loss_func(self, inputs, action, returns):
        self.train()
        batch_size = inputs[0].shape[0]
        logits, _, values = self.forward(inputs, forced_action=action)
        td = returns - values
        critic_loss = td.pow(2)

        log_prob = torch.zeros(1, batch_size).to(device=logits[0].device)
        for i, lgt in enumerate(logits):
            probs = F.softmax(lgt, dim=1)
            m = torch.distributions.Categorical(probs)
            log_prob += m.log_prob(action[:, i])

        actor_loss = -log_prob * td.detach()
        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def optimize(self, optimizer, done_episode, next_state, inputs, actions, rewards, gamma):
        if done_episode:
            value_est = 0
        else:
            # bootstrap
            value_est = self.forward(next_state)[-1][0].item()

        returns = []
        for rwd in rewards[::-1]:
            value_est = rwd + gamma * value_est
            returns.append(value_est)
        returns.reverse()
        returns = torch.Tensor(returns).to(inputs[0].device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        loss = self.loss_func(inputs, actions, returns)

        # sometimes adjust gradient randomly for exploration?
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
