import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, Dropout, LSTM
from torch.nn.functional import relu,tanh
import pandas as pd
from torch.utils.data import DataLoader
import pdb

class SSBM_LSTM(nn.Module):
     action_state_dim = 383
     button_combination_dim = 32
     input_dim = 48
     cts_out_dim = 6
     logit_out_dim = 32
     num_embedding_features = 6
     def __init__(self, action_embedding_dim, button_embedding_dim, hidden_size = 256, num_layers = 1, bidirectional=False, dropout_p=0.2, attention=False):
            super().__init__()

            self.action_state_embedding = Embedding(num_embeddings=SSBM_LSTM.action_state_dim, \
                                                    embedding_dim=action_embedding_dim)
            self.button_combination_embedding = Embedding(num_embeddings=SSBM_LSTM.button_combination_dim, \
                                                    embedding_dim=button_embedding_dim)
            self.in_features_dim = (SSBM_LSTM.input_dim - SSBM_LSTM.num_embedding_features) +  (4 * action_embedding_dim + 2 * button_embedding_dim)
            if bidirectional == True:
               self.num_directions = 2
            else:
               self.num_directions = 1
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.attention = attention
            # import pdb
            # pdb.set_trace()
            self.LSTM = LSTM(input_size = self.in_features_dim, hidden_size= hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            self.attention_proj = Linear(in_features=hidden_size * self.num_directions, out_features=hidden_size * self.num_directions)
            self.dropout = Dropout(p=dropout_p)
            self.cts_out = Linear(in_features=hidden_size * self.num_directions, out_features=SSBM_LSTM.cts_out_dim )
            self.logits_out = Linear(in_features=hidden_size * self.num_directions, out_features=SSBM_LSTM.logit_out_dim)

     def forward(self, x):
         # x -> (batch, seq_len, feat_dim)

         batch_size = x.shape[0]
         seq_len = x.shape[1]

         embed_indices, regular_feat = x[:,:,0:SSBM_LSTM.num_embedding_features].long(), x[:,:,SSBM_LSTM.num_embedding_features:]
         action_embed_idx = torch.cat([embed_indices[:,:,0:2], embed_indices[:,:,3:5]] ,dim=1)

         button_combination_idx = torch.cat([embed_indices[:,:,2].reshape(batch_size, seq_len,1),embed_indices[:,:,5].reshape(batch_size, seq_len, 1)], dim=1)
        #  import pdb
        #  pdb.set_trace()
         action_embed_feat = self.action_state_embedding(action_embed_idx.reshape(-1)).reshape(batch_size, seq_len, -1)
         button_embed_feat = self.button_combination_embedding(button_combination_idx.reshape(-1)).reshape(batch_size, seq_len, -1)

         features = torch.cat([action_embed_feat, button_embed_feat, regular_feat], axis=-1).float()
         assert(features.shape == (batch_size, seq_len, self.in_features_dim))
         # hn -> (1, batch, hidden_dim)

         lstm_output, (h_n, c_n) = self.LSTM(features)
         if self.attention == False:
            lstm_representation = h_n
         else:

            lstm_output_proj = self.attention_proj(lstm_output)
            # (batch, seq_len, hidden * num_layer)
            attention_logits = torch.squeeze(torch.bmm(input=lstm_output_proj, mat2=torch.unsqueeze(torch.squeeze(h_n, dim=0), dim=-1)), dim=-1)
            attention_probs =  nn.functional.softmax(attention_logits, dim=1)
            combined_hidden = torch.squeeze(torch.bmm(input=torch.unsqueeze(attention_probs, dim=1), mat2 = lstm_output), dim=1)
            # (batch, hidden * num_layer )
            lstm_representation = combined_hidden


         lstm_representation = lstm_representation.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
         lstm_representation = lstm_representation[-1]
         o = lstm_representation.permute(1, 0, 2).reshape(batch_size, -1)

         o = self.dropout(o)

         cts_o = torch.tanh(self.cts_out(o))
         logits_o = self.logits_out(o)

         return cts_o, logits_o


