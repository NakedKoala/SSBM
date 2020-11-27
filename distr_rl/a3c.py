import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        stale_state,
        recent_input,
        forced_action=None,
        behavior=behavior
    ):
        return self.model(
            stale_state, recent_input,
            forced_action=forced_action,
            behavior=behavior
        )

    def choose_action(self, stale_state, recent_input, behavior):
        self.eval()
        # could add exploration strategy here...
        # or in the runner
        _, choices, _ = self.forward(stale_state, recent_input, behavior=behavior)
        return choices

    def loss_func(self, stale_state, recent_input, action, returns):
        self.train()
        batch_size = stale_state[0].shape
        logits, _, values = self.forward(stale_state, recent_input, forced_action=action)
        td = returns - values
        critic_loss = td.pow(2)

        log_prob = torch.zeros(1, batch_size).to(device=logits.device)
        for i, lgt in enumerate(logits):
            probs = F.softmax(lgt, dim=1)
            m = torch.distributions.Categorical(probs)
            log_prob += m.log_prob(action[i, :])

        exp_v = log_prob * td.detach().squeeze()
        actor_loss = -exp_v
        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def optimize(self, optimizer, done_episode, next_state, state_buffer, actions, rewards, gamma):
        if done_episode:
            value_est = 0
        else:
            # bootstrap
            _, _, value_est = self.model.forward(next_state[0], next_state[1])[-1][0].item()

        returns = []
        for rwd in rewards[::-1]:
            value_est = rwd + gamma * value_est
            returns.append(value_est)
        returns.reverse()
        returns = torch.Tensor(returns).to(state_buffer[0].device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        loss = self.loss_func(state_buffer[0], state_buffer[1], actions, returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

