import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, Dropout
from torch.nn.functional import relu,tanh
import pandas as pd
from dataset import SSBMDataset
from torch.utils.data import DataLoader
import pdb


class SSBM_MVP(nn.Module):
     action_state_dim = 383
     button_combination_dim = 32
     input_dim = 54
     cts_out_dim = 6
     logit_out_dim = 32
#      logit_out_dim = 5
     num_embedding_features = 6

     def __init__(self, action_embedding_dim, button_embedding_dim, hidden_sizes = [1024, 512, 256]):
            super().__init__()
            assert(len(hidden_sizes) == 3)
            self.action_state_embedding = Embedding(num_embeddings=SSBM_MVP.action_state_dim, \
                                                    embedding_dim=action_embedding_dim)
            self.button_combination_embedding = Embedding(num_embeddings=SSBM_MVP.button_combination_dim, \
                                                    embedding_dim=button_embedding_dim)
            in_features_dim = (SSBM_MVP.input_dim - SSBM_MVP.num_embedding_features) +  ( 4 * action_embedding_dim + 2 * button_embedding_dim )
            self.in_features_dim = in_features_dim
            self.dense1 = Linear(in_features=in_features_dim , out_features=hidden_sizes[0])
            self.dense2 = Linear(in_features=hidden_sizes[0] , out_features=hidden_sizes[1])
            self.dense3 = Linear(in_features=hidden_sizes[1] , out_features=hidden_sizes[2])
            self.dropout = Dropout(p=0.2)
            self.cts_out = Linear(in_features=hidden_sizes[-1] , out_features=SSBM_MVP.cts_out_dim )
            self.logits_out = Linear(in_features=hidden_sizes[-1] , out_features=SSBM_MVP.logit_out_dim)
            
     def forward(self, x):
       #   import pdb 
       #   pdb.set_trace()
       #   import pdb 
       #   pdb.set_trace()
         batch_size = x.shape[0]
         embed_indices, regular_feat = x[:,0:SSBM_MVP.num_embedding_features].long(), x[:,SSBM_MVP.num_embedding_features:]
        
         action_embed_idx = torch.cat([embed_indices[:,0:2], embed_indices[:,3:5]] ,dim=1)
         button_combination_idx = torch.cat([embed_indices[:,2].reshape(-1, 1),embed_indices[:,5].reshape(-1,1)], dim=1)
      
        
         action_embed_feat = self.action_state_embedding(action_embed_idx.reshape(-1)).reshape(batch_size,-1)
         button_embed_feat = self.button_combination_embedding(button_combination_idx.reshape(-1)).reshape(batch_size,-1)

         features = torch.cat([action_embed_feat, button_embed_feat, regular_feat], axis=1).float()
       #   import pdb 
       #   pdb.set_trace()
         assert(features.shape == (batch_size, self.in_features_dim))
        
         o =  self.dropout(relu(self.dense1(features)))
         o =  self.dropout(relu(self.dense2(o)))
         o =  self.dropout(relu(self.dense3(o)))
         cts_o = torch.tanh(self.cts_out(o))
         logits_o = self.logits_out(o)
         
         return cts_o, logits_o
         
        