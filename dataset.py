import pandas as pd
import numpy as np
import torch
import os
import pdb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader



class SSBMDataset(Dataset):
    enum_to_softmax_index = {
       0: 0,   #NONE
       1: 1,   #DPAD_LEFT
       2: 2,   #DPAD_RIGHT
       4: 3,   #DPAD_DOWN
       8: 4,   #DPAD_UP
       16: 5,  #Z
       32: 6,  #R
       64: 7,  #L
       256: 8, #A
       512: 9, #B
       1024: 10,  #X
       2048: 11,  #Y
       4096: 12,  #START
    }

    def __init__(self, src_dir, filter_char_id, is_dev=False):
        
        self.csv_files = [ os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if '.csv' in fname]
        self.features = []
        self.targets = []
        
        for csv_path in self.csv_files:
            curr_features, curr_targets = self.proc_csv(csv_path, filter_char_id)
            self.features.append(curr_features)
            self.targets.append(curr_targets)
            
        self.features = torch.cat(self.features, dim=0)
        self.targets = torch.cat(self.targets, dim=0)
        if is_dev == True:
            self.features = self.features[:1000]
            self.targets = self.targets[:1000]
       
       
    def __len__(self):
        return len(self.features)
       

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
        

    def proc_csv(self, csv_path, filter_char_id):
        
        df = pd.read_csv(csv_path, index_col="frame_index")
        feat_cols = ['pre_state',  'post_state','pre_position_x', 'pre_position_y',  'post_position_x', 'post_position_y', \
                'post_damage', 'post_state_age', 'post_combo_count', 'post_shield', 'post_last_attack_landed', \
                'post_hit_stun', 'post_last_hit_by', 'post_airborne', 'pre_damage',  'post_direction_1', 'post_direction_-1', \
                'pre_direction_1', 'pre_direction_-1', 'post_stocks_0', 'post_stocks_1', 'post_stocks_2', 'post_stocks_3', \
                'post_stocks_4', 'post_ground_0.0', 'post_ground_1.0','post_ground_2.0',  'post_jumps_0', 'post_jumps_1',  \
                'post_jumps_2']
        target_cols = ['pre_joystick_x', 'pre_joystick_y',  'pre_cstick_x', 'pre_cstick_y', \
                       'pre_triggers_x', 'pre_triggers_y', 'pre_buttons']

        df = df[df['post_character'] == filter_char_id] #12
        df['pre_buttons'] = df['pre_buttons'].apply(lambda x: SSBMDataset.enum_to_softmax_index[x] if x in SSBMDataset.enum_to_softmax_index else 0)
        features_df, targets_df = df[feat_cols].shift(1).fillna(0), df[target_cols] 
       
        features_np, targets_np = features_df.to_numpy(), targets_df.to_numpy()

        scaler = StandardScaler()

        features_np[:,2:] = scaler.fit_transform(features_np[:,2:])

        features_tensor, targets_tensor = torch.from_numpy(features_np), torch.from_numpy(targets_np)
        return features_tensor.float(), targets_tensor.float()
        
