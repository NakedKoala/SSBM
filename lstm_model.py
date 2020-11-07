import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, Dropout, LSTM
from torch.nn.functional import relu,tanh
import pandas as pd
from dataset import SSBMDataset
from torch.utils.data import DataLoader
import pdb


class SSBM_LSTM(nn.Module):
     action_state_dim = 383
     input_dim = 66
     cts_out_dim = 6
     logit_out_dim = 7
     num_embedding_features = 4
     def __init__(self, embedding_dim, hidden_size = 256):
            super().__init__()
            
            self.action_state_embedding = Embedding(num_embeddings=SSBM_LSTM.action_state_dim, \
                                             embedding_dim=embedding_dim)
            in_features_dim = (SSBM_LSTM.input_dim - SSBM_LSTM.num_embedding_features) +  (embedding_dim * SSBM_LSTM.num_embedding_features)
            self.in_features_dim = in_features_dim

            self.LSTM = LSTM(in_features_dim, hidden_size, batch_first=True)
            
            self.dropout = Dropout(p=0.2)
            self.cts_out = Linear(in_features=hidden_size , out_features=SSBM_LSTM.cts_out_dim )
            self.logits_out = Linear(in_features=hidden_size , out_features=SSBM_LSTM.logit_out_dim)
            
     def forward(self, x):
         # x -> (batch, seq_len, feat_dim)
         
         batch_size = x.shape[0]
         seq_len = x.shape[1]

         embed_idx, regular_feat = x[:,:,0:SSBM_LSTM.num_embedding_features].long(), x[:,:,SSBM_LSTM.num_embedding_features:]
         embed_feat = self.action_state_embedding(embed_idx.reshape(-1)).reshape(batch_size, seq_len, -1)
         features = torch.cat([embed_feat, regular_feat], axis=-1).float()
       
         assert(features.shape == (batch_size, seq_len, self.in_features_dim))
         # hn -> (1, batch, hidden_dim)
         _, (h_n, c_n) = self.LSTM(features)
         o = h_n.squeeze(0)
         o = self.dropout(o)
        #  import pdb 
        #  pdb.set_trace()
         cts_o = torch.tanh(self.cts_out(o))
         logits_o = self.logits_out(o)
         
         return cts_o, logits_o
         
        