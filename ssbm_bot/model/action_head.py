import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, Sequential, ReLU, Dropout, ModuleList
from torch.nn.functional import softmax
from torch.distributions import Categorical, Uniform

from .. import controller_indices as c_idx

class ActionHead(nn.Module):
    def __init__(self, in_features,
                 # buttons, stick coarse, stick fine, stick magn, cstick coarse, cstick fine, cstick magn, trigger
                 out_embedding_dims=[64, 128, 32, 128, 64, 16, 64, 32],
                 out_output_sizes=[
                     c_idx.button.NUM_INDICES,
                     c_idx.stick.COARSE_N,
                     c_idx.stick.PRECISE_N,
                     c_idx.stick.MAGN_N,
                     c_idx.stick.COARSE_N,
                     c_idx.stick.PRECISE_N,
                     c_idx.stick.MAGN_N,
                     c_idx.trigger.NUM_INDICES
                 ],
                 out_hidden_sizes=[
                     [64, 32], # buttons
                     [256, 128], # stick coarse - NOTE - actually has 129 outputs
                     [32, 16], # stick fine
                     [128, 128], # stick magn
                     [256, 128], # cstick coarse - NOTE - actually has 129 outputs
                     [16, 16], # cstick fine
                     [128, 128], # cstick magn
                     [256, 128], # trigger
                 ],
                 buttons_emb=None
    ):
        super().__init__()

        assert(len(out_embedding_dims) == len(out_output_sizes) == len(out_hidden_sizes))
        # save for reference only
        self.out_output_sizes = out_output_sizes
        # regular list won't register layers properly
        self.output_emb_layers = ModuleList()
        self.output_lin_layers = ModuleList()
        total_in_size = in_features
        # connect linear layers and embedding layers together
        for emb_dim, out_size, hidden_sizes in zip(out_embedding_dims, out_output_sizes, out_hidden_sizes):
            self.output_emb_layers.append(
                Embedding(num_embeddings=out_size, embedding_dim=emb_dim)
            ) # NOTE: we don't actually use the last embedding layer
            last_in_size = total_in_size
            hidden_layers = []
            for size in hidden_sizes:
                hidden_layers.extend((
                    Linear(in_features=last_in_size, out_features=size),
                    ReLU(),
                    Dropout(p=0.2)
                ))
                last_in_size = size
            hidden_layers.append(Linear(in_features=last_in_size, out_features=out_size))

            self.output_lin_layers.append(Sequential(*hidden_layers))
            total_in_size += emb_dim


    DEFAULT = 0 # choose action based on probability logits
    MAX = 1 # choose action based on maximum logit
    RANDOM = 2 # choose action randomly at uniform
    # behavior for use in RL experience generation only - not supervised/RL training!
    # use default for normal RL experience generation
    # use max for evaluating entire model output for evaluation
    # use random in RL experience generation occassionally
    # forced_action required for supervised learning and for RL learning
    def forward(self, x, forced_action=None, behavior=1):
        if forced_action is not None:
            assert(forced_action.shape[0] == x.shape[0])
            assert(forced_action.shape[1] == len(self.output_lin_layers))
        batch_size = x.shape[0]
        embeddings = None
        outputs = []
        action_idx_chosen = None
        for i in range(len(self.output_lin_layers)):
            if i > 0:
                # update embeddings with the most recent determine action category to take
                cur_emb = self.output_emb_layers[i-1](action_idx_chosen[:,-1]).reshape(batch_size, -1)
                if embeddings is None:
                    embeddings = cur_emb
                else:
                    embeddings = torch.cat((embeddings, cur_emb), 1)
            # embeddings is empty on the first iteration
            if embeddings is None:
                combined_in = x
            else:
                combined_in = torch.cat((x, embeddings), 1)

            # get distribution of next action category given state + previous categories chosen
            category_logits = self.output_lin_layers[i](combined_in)
            category_probs = softmax(category_logits, dim=1)

            # sample distribution
            # NOTE during RL training, occassionally randomly select an action to take
            # for exploration
            if forced_action is not None:
                category = forced_action[:,i].reshape(batch_size, -1)
            elif behavior == self.DEFAULT:
                m = Categorical(category_probs)
                category = m.sample().reshape(batch_size, -1).to(device=x.device)
            elif behavior == self.MAX:
                _, category = torch.max(category_logits, dim=1)
                category = category.reshape(batch_size, -1)
            elif behavior == self.RANDOM:
                m = Uniform(torch.tensor([0.0]), torch.tensor([1.0])) # U[0, 1)
                u_sample = m.rsample((batch_size,)).to(device=x.device)
                category = torch.floor(u_sample * (self.out_output_sizes[i])).long() # U[0, output_size) in integers
            else:
                raise AttributeError("invalid action selection behaviour specified")

            if action_idx_chosen is None:
                action_idx_chosen = category
            else:
                action_idx_chosen = torch.cat((action_idx_chosen, category), 1)

            # append logits to output - don't softmax, so we can log_softmax later
            outputs.append(category_logits)

        return outputs, action_idx_chosen
