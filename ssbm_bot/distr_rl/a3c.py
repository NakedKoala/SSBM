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
        stale_state,
        # recent_input,
        forced_action=None,
        behavior=ActionHead.DEFAULT
    ):
        return self.model(
            stale_state, # recent_input,
            forced_action=forced_action,
            behavior=behavior
        )

    def choose_action(self, stale_state, behavior):
        self.eval()
        # could add exploration strategy here...
        # or in the runner
        _, choices, _ = self.forward(stale_state, behavior=behavior)
        return choices

    def loss_func(self, stale_state, action, returns):
        self.train()
        batch_size = stale_state.shape[0]
        logits, _, values = self.forward(stale_state, forced_action=action)
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

    def optimize(self, optimizer, done_episode, next_state, stale_states, actions, rewards, gamma):
        if done_episode:
            value_est = 0
        else:
            # bootstrap
            last_state_input = stale_states[-1]
            next_input = torch.cat((last_state_input[1:], next_state.unsqueeze(dim=0)), dim=0).unsqueeze(dim=0)
            value_est = self.model.forward(next_input)[-1][0].item()

        returns = []
        for rwd in rewards[::-1]:
            value_est = rwd + gamma * value_est
            returns.append(value_est)
        returns.reverse()
        returns = torch.Tensor(returns).to(stale_states.device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        loss = self.loss_func(stale_states, actions, returns)

        # sometimes adjust gradient randomly for exploration?
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
