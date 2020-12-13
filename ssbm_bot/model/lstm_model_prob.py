import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, Dropout, LSTM, ReLU, Sequential
import pandas as pd
from torch.utils.data import DataLoader
import pdb

from .action_head import ActionHead

class SSBM_LSTM_Prob(nn.Module):
     action_state_dim = 383
     input_dim = 54
     no_opp_input_dim = 47
     recent_actions_dim = 7
     num_embedding_features = 9
     no_opp_num_emb_feats = 8
     def __init__(self, action_embedding_dim, hidden_size = 256, num_layers = 1, bidirectional=False, dropout_p=0.2,
                  attention=False, recent_actions=False, value_hidden_sizes = [256, 128], character_embedding_dim = 50,
                  stage_embedding_dim = 50, include_opp_input=True, latest_state_reminder=False,
                  own_dropout_p = 0.5, opp_dropout_p = 0.5,
                  **kwargs):
            super().__init__()

            self.recent_actions = recent_actions
            self.latest_state_reminder = latest_state_reminder

            if bidirectional == True:
               self.num_directions = 2
            else:
               self.num_directions = 1

            if recent_actions:
                self.lstm_out_size = hidden_size * 2
            else:
                self.lstm_out_size = hidden_size
            if latest_state_reminder:
               self.action_head = ActionHead(self.lstm_out_size * self.num_directions + 12, **kwargs)
            else:
               self.action_head = ActionHead(self.lstm_out_size * self.num_directions, **kwargs)

            self.action_state_embedding = Embedding(num_embeddings=self.action_state_dim, \
                                                    embedding_dim=action_embedding_dim)


            self.character_embedding = Embedding(num_embeddings=33, embedding_dim=character_embedding_dim)
            self.stage_embedding = Embedding(num_embeddings=33, embedding_dim=stage_embedding_dim)

            # share embedding with action head
            self.button_combination_embedding = self.action_head.output_emb_layers[0]
            button_embedding_dim = self.button_combination_embedding.embedding_dim

            # change dimensions according to include_opp_input
            self.include_opp_input = include_opp_input
            self.in_dim = self.input_dim
            self.num_emb_feats = self.num_embedding_features
            if not include_opp_input:
                self.in_dim = self.no_opp_input_dim
                self.um_emb_feats = self.no_opp_num_emb_feats

            num_btn_embs = 2 if include_opp_input else 1
            self.in_features_dim = (self.in_dim - self.num_emb_feats) +  (4 * action_embedding_dim + num_btn_embs * button_embedding_dim) + character_embedding_dim * 2 + stage_embedding_dim
            self.num_layers = num_layers
            # import pdb
            # pdb.set_trace()
            self.LSTM = LSTM(input_size = self.in_features_dim, hidden_size= hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

            # for recent actions
            if recent_actions:
                lstm2_in_dim = (self.recent_actions_dim - 1) + button_embedding_dim
                self.LSTM2 = LSTM(input_size=lstm2_in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

            self.attention_proj = Linear(in_features=self.lstm_out_size * self.num_directions, out_features=self.lstm_out_size * self.num_directions)
            self.dropout = Dropout(p=dropout_p)
            self.own_input_dropout = Dropout(p=own_dropout_p)
            self.opp_input_dropout = Dropout(p=opp_dropout_p)
            self.attention = attention

            # create value head
            value_hidden_layers = []
            last_in_size = self.lstm_out_size * self.num_directions

            if latest_state_reminder:
               last_in_size += 12

            for v_hidden_size in value_hidden_sizes:
                value_hidden_layers.extend((
                    Linear(in_features=last_in_size, out_features=v_hidden_size),
                    ReLU(),
                    Dropout(p=0.2)
                ))
                last_in_size = v_hidden_size

            value_hidden_layers.append(
                Linear(in_features=value_hidden_sizes[-1], out_features=1)
            )
            self.value_head = Sequential(*value_hidden_layers)

     def forward(self, x, forced_action=None, behavior=1):
         if self.recent_actions:
            x, recent_actions = x
            recent_actions[:,:,1:] = self.own_input_dropout(recent_actions[:,:,1:])
         # x -> (batch, seq_len, feat_dim)

         # position and direction
         if self.latest_state_reminder:
            if self.include_opp_input:
                latest_state = torch.cat([x[:, -1, 10:16], x[:, -1, 26:32]], dim=1)
            else:
                latest_state = torch.cat([x[:, -1, 9:15], x[:, -1, 25:31]], dim=1)

         # import pdb
         # pdb.set_trace()

         batch_size = x.shape[0]
         seq_len = x.shape[1]

         embed_indices = x[:,:,1:1+self.num_emb_feats].long()
         regular_feat = torch.cat([x[:,:,0].unsqueeze(dim=-1), x[:,:,1+self.num_emb_feats:]], axis=-1)
        #  pdb.set_trace()

         if self.include_opp_input:
            regular_feat[:,:,-12:-6] = self.own_input_dropout(regular_feat[:,:,-12:-6])
            regular_feat[:,:,-6:] = self.opp_input_dropout(regular_feat[:,:,-6:])
         else:
            regular_feat[:,:,-6:] = self.own_input_dropout(regular_feat[:,:,-6:])
         
         
         # import pdb; pdb.set_trace()
         # stage | (player) state state character button | (opponent) state state character [button]
         stage_embed_idx = embed_indices[:,:,0]

         action_embed_idx = torch.cat([
             embed_indices[:,:,1:3].reshape(-1, 2),
             embed_indices[:,:,5:7].reshape(-1, 2)
         ], dim=1)

         character_embed_idx = torch.cat([
             embed_indices[:,:,3].reshape(-1, 1),
             embed_indices[:,:,7].reshape(-1, 1)
         ], dim=1)

         if self.include_opp_input:
             button_combination_idx = torch.cat([
                 embed_indices[:,:,4].reshape(-1, 1),
                 embed_indices[:,:,8].reshape(-1, 1)
             ], dim=1)

         else:
             button_combination_idx = embed_indices[:,:,4]


         action_embed_feat = self.action_state_embedding(action_embed_idx.reshape(-1)).reshape(batch_size, seq_len, -1)
         button_embed_feat = self.button_combination_embedding(button_combination_idx.reshape(-1)).reshape(batch_size, seq_len, -1)
         embed_dim = self.button_combination_embedding.embedding_dim
         
         if self.include_opp_input:
            button_embed_feat[:,:,-embed_dim:] = self.opp_input_dropout(button_embed_feat[:,:,-embed_dim:])
            button_embed_feat[:,:,-embed_dim*2:-embed_dim] = self.own_input_dropout(button_embed_feat[:,:,-embed_dim*2:-embed_dim])
         else:
            button_embed_feat = self.own_input_dropout(button_embed_feat)

         
         character_embed_feat = self.character_embedding(character_embed_idx).reshape(batch_size, seq_len, -1)
         stage_embed_feat =  self.stage_embedding(stage_embed_idx).reshape(batch_size, seq_len, -1)

         features = torch.cat([action_embed_feat, button_embed_feat, character_embed_feat, stage_embed_feat, regular_feat], axis=-1).float()

         assert(features.shape == (batch_size, seq_len, self.in_features_dim))
         # hn -> (1, batch, hidden_dim)
         lstm_output, (h_n, c_n) = self.LSTM(features)

         if self.recent_actions and recent_actions is not None:
            # recent_actions_seq_len = recent_actions.shape[1]
            # pdb.set_trace()
            recent_btn_indices, recent_other = recent_actions[:,:,0].long(), recent_actions[:,:,1:]
            recent_btn_embed_feat = self.button_combination_embedding(recent_btn_indices)
            recent_actions_feat = torch.cat([recent_btn_embed_feat, recent_other], axis=-1).float()

            # NOTE correct to send last hidden state to next lstm?
            lstm_output_2, (h_n2, c_n2) = self.LSTM2(recent_actions_feat, (h_n, c_n))
            lstm_output = torch.cat((lstm_output, lstm_output_2), dim=1)
            h_n = torch.cat((h_n, h_n2), dim=2)

         if self.attention == False:
            lstm_representation = h_n
         else:
            # NOTE broken.
            lstm_output_proj = self.attention_proj(lstm_output)
            # (batch, seq_len, hidden * num_layer)
            attention_logits = torch.squeeze(torch.bmm(input=lstm_output_proj, mat2=torch.unsqueeze(torch.squeeze(h_n, dim=0), dim=-1)), dim=-1)
            attention_probs =  nn.functional.softmax(attention_logits, dim=1)
            combined_hidden = torch.squeeze(torch.bmm(input=torch.unsqueeze(attention_probs, dim=1), mat2 = lstm_output), dim=1)
            # (batch, hidden * num_layer )
            lstm_representation = combined_hidden

         lstm_representation = lstm_representation.view(self.num_layers, self.num_directions, batch_size, self.lstm_out_size)
         lstm_representation = lstm_representation[-1]
         lstm_representation = lstm_representation.permute(1, 0, 2).reshape(batch_size, -1)
         if self.latest_state_reminder:
            o = torch.cat([lstm_representation, latest_state], dim=1)
         else:
            o = lstm_representation


         o = self.dropout(o)
        #  import pdb
        #  pdb.set_trace()
         logits, choices = self.action_head(o, forced_action=forced_action, behavior=behavior)

         value = self.value_head(o)

         return logits, choices, value

