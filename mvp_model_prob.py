import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, Dropout, ReLU, Dropout, Sequential
from action_head import ActionHead
import pdb

# Based on SSBM_MVP, but outputs a (partial) probability distribution rather than an action.
class SSBM_MVP_Prob(nn.Module):
    action_state_dim = 383
    button_combination_dim = 32
    input_dim = 48
    cts_out_dim = 6
    logit_out_dim = 32
    num_embedding_features = 6
    # kwargs contains ActionHead arguments
    def __init__(self, action_embedding_dim, button_embedding_dim,
            main_hidden_sizes = [1024, 512, 256],
            value_hidden_sizes = [256, 128],
            **kwargs):
            super().__init__()

            self.action_state_embedding = Embedding(num_embeddings=self.action_state_dim, \
                                                    embedding_dim=action_embedding_dim)
            self.button_combination_embedding = Embedding(num_embeddings=self.button_combination_dim, \
                                                    embedding_dim=button_embedding_dim)

            in_features_dim = (self.input_dim - self.num_embedding_features) +  ( 4 * action_embedding_dim + 2 * button_embedding_dim )
            self.in_features_dim = in_features_dim

            def create_linear_block(in_features, hidden_sizes):
                value_hidden_layers = []
                last_in_size = in_features
                for hidden_size in hidden_sizes:
                    value_hidden_layers.extend((
                        Linear(in_features=last_in_size, out_features=hidden_size),
                        ReLU(),
                        Dropout(p=0.2)
                    ))
                    last_in_size = hidden_size
                return value_hidden_layers

            self.main_nn = Sequential(
                *create_linear_block(in_features_dim, main_hidden_sizes)
            )
            value_head_layers = create_linear_block(main_hidden_sizes[-1], value_hidden_sizes)
            value_head_layers.append(
                Linear(in_features=value_hidden_sizes[-1], out_features=1)
            )
            self.value_head = Sequential(*value_head_layers)
            self.action_head = ActionHead(main_hidden_sizes[-1], **kwargs)

    def forward(self, x, compute_value=False, forced_action=None, behavior=2):
        batch_size = x.shape[0]
        embed_indices, regular_feat = x[:,0:self.num_embedding_features].long(), x[:,self.num_embedding_features:]

        action_embed_idx = torch.cat([embed_indices[:,0:2], embed_indices[:,3:5]] ,dim=1)
        button_combination_idx = torch.cat([embed_indices[:,2].reshape(-1, 1),embed_indices[:,5].reshape(-1,1)], dim=1)


        action_embed_feat = self.action_state_embedding(action_embed_idx.reshape(-1)).reshape(batch_size,-1)
        button_embed_feat = self.button_combination_embedding(button_combination_idx.reshape(-1)).reshape(batch_size,-1)

        features = torch.cat([action_embed_feat, button_embed_feat, regular_feat], axis=1).float()
       #   import pdb
       #   pdb.set_trace()
        assert(features.shape == (batch_size, self.in_features_dim))

        o = self.main_nn(features)
        logits, choices = self.action_head(o, forced_action=forced_action, behavior=behavior)
        if compute_value:
            value = self.value_head(o)
        else:
            value = torch.zeros(batch_size, 1).to(x.device)

        return logits, choices, value
