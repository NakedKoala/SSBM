import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, Dropout, LSTM
from torch.nn.functional import relu,tanh
import pandas as pd
from dataset import SSBMDataset
from action_head import ActionHead
from torch.utils.data import DataLoader
import pdb


class SSBM_LSTM_Prob(nn.Module):
     action_state_dim = 383
     button_combination_dim = 32
     input_dim = 48
     cts_out_dim = 6
     logit_out_dim = 32
     num_embedding_features = 6
     def __init__(self, action_embedding_dim, button_embedding_dim, hidden_size = 256, num_layers = 1, bidirectional=False, dropout_p=0.2, **kwargs):
            super().__init__()

            self.action_state_embedding = Embedding(num_embeddings=self.action_state_dim, \
                                                    embedding_dim=action_embedding_dim)
            self.button_combination_embedding = Embedding(num_embeddings=self.button_combination_dim, \
                                                    embedding_dim=button_embedding_dim)
            self.in_features_dim = (self.input_dim - self.num_embedding_features) +  (4 * action_embedding_dim + 2 * button_embedding_dim)
            if bidirectional == True:
               self.num_directions = 2
            else:
               self.num_directions = 1
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            # import pdb
            # pdb.set_trace()
            self.LSTM = LSTM(input_size = self.in_features_dim, hidden_size= hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

            self.dropout = Dropout(p=dropout_p)
            self.action_head = ActionHead(hidden_size * self.num_directions, **kwargs)

     def forward(self, x, forced_action=None, behavior=2):
         # x -> (batch, seq_len, feat_dim)

         batch_size = x.shape[0]
         seq_len = x.shape[1]

         embed_indices, regular_feat = x[:,:,0:self.num_embedding_features].long(), x[:,:,self.num_embedding_features:]
         action_embed_idx = torch.cat([embed_indices[:,:,0:2], embed_indices[:,:,3:5]] ,dim=1)

         button_combination_idx = torch.cat([embed_indices[:,:,2].reshape(batch_size, seq_len,1),embed_indices[:,:,5].reshape(batch_size, seq_len, 1)], dim=1)
        #  import pdb
        #  pdb.set_trace()
         action_embed_feat = self.action_state_embedding(action_embed_idx.reshape(-1)).reshape(batch_size, seq_len, -1)
         button_embed_feat = self.button_combination_embedding(button_combination_idx.reshape(-1)).reshape(batch_size, seq_len, -1)

         features = torch.cat([action_embed_feat, button_embed_feat, regular_feat], axis=-1).float()

         assert(features.shape == (batch_size, seq_len, self.in_features_dim))
         # hn -> (1, batch, hidden_dim)
         _, (h_n, c_n) = self.LSTM(features)
         h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
         h_n = h_n[-1]
         o = h_n.permute(1, 0, 2).reshape(batch_size, -1)

         o = self.dropout(o)
        #  import pdb
        #  pdb.set_trace()
         logits, choices = self.action_head(o, forced_action=forced_action, behavior=behavior)
         value = torch.zeros(batch_size, 1).to(x.device)

         return logits, choices, value
