import pandas as pd
import numpy as np
import torch
import os
import pdb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import traceback



class SSBMDataset(Dataset):

   

    val_ratio = 0.2
    button_press_indicator_dim = 12
    frame_delay = 2
    def __init__(self, src_dir, char_id, opponent_id, device, ds_type=None):
        torch.manual_seed(0)
        self.csv_files = [ os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if '.csv' in fname]
        self.features = []
        self.cts_targets = []
        self.bin_cls_targets = []
        
        for csv_path in tqdm(self.csv_files):
            try:
                features, cts_targets, bin_cls_targets = self.proc_csv(csv_path, char_id, opponent_id, SSBMDataset.frame_delay)
                self.features.append(features)
                self.cts_targets.append(cts_targets)
                self.bin_cls_targets.append(bin_cls_targets)
            except:
                print(f'failed to load {csv_path}')
                traceback.print_exc()
        
        self.features = torch.cat(self.features, dim=0)
        self.cts_targets = torch.cat(self.cts_targets, dim=0)
        self.bin_cls_targets = torch.cat(self.bin_cls_targets, dim=0)

        shuffled_idx = torch.randperm(len(self.features))
        self.features = self.features[shuffled_idx]
        self.cts_targets = self.cts_targets[shuffled_idx]
        self.bin_cls_targets = self.bin_cls_targets[shuffled_idx]
        
        self.features = self.features.to(device)
        self.cts_targets =  self.cts_targets.to(device)
        self.bin_cls_targets =  self.bin_cls_targets.to(device)

        if ds_type == 'dev':
            self.features = self.features[:1000]
            self.cts_targets = self.cts_targets[:1000]
            self.bin_cls_targets = self.bin_cls_targets[:1000]

    def __len__(self):
        return len(self.features)
       

    def __getitem__(self, idx):
        return self.features[idx], self.cts_targets[idx], self.bin_cls_targets[idx]
    
    def proc_button_press(self, buttons_values_np):

        def convert_button_value_to_indicator_vector(value):
           
            tmp_val = value
            bits = []
            while value > 0:
                bits.append(value % 2)
                value //= 2 
            bits = list(reversed(bits)) 

            if len(bits) > 12 :
                bits = bits[1:]

            padded_len = SSBMDataset.button_press_indicator_dim - len(bits)

            if padded_len > 0:
                bits = [0] * padded_len + bits
           
            assert(len(bits) == 12 )
          
           
            selected_bits = bits[4:-1]
            assert( len(selected_bits) == 7)
            # Keep valid keys: [d_left, d_right, d_down, d_up, z, r ,l, a, b, x, y, start]
            return np.array(selected_bits)
           
     
        f = np.vectorize(convert_button_value_to_indicator_vector, otypes=[np.ndarray])
        # try:
        bin_cls_targets_np = np.stack(f(buttons_values_np), axis=0) 
        return bin_cls_targets_np


    def proc_csv(self, csv_path, char_id, opponent_id, frame_delay):
        
        df = pd.read_csv(csv_path, index_col="frame_index")
        feat_cols = ['pre_state',  'post_state','pre_position_x', 'pre_position_y',  'post_position_x', 'post_position_y', \
                'post_damage', 'post_state_age', 'post_combo_count', 'post_shield', 'post_last_attack_landed', \
                'post_hit_stun', 'post_last_hit_by', 'post_airborne', 'pre_damage',  'post_direction_1', 'post_direction_-1', \
                'pre_direction_1', 'pre_direction_-1', 'post_stocks', 'post_ground_0.0', 'post_ground_1.0','post_ground_2.0',  'post_jumps_0', 'post_jumps_1',  \
                'post_jumps_2']
        target_cols = ['pre_joystick_x', 'pre_joystick_y',  'pre_cstick_x', 'pre_cstick_y', \
                       'pre_triggers_x', 'pre_triggers_y', 'pre_buttons']
        
        df_char = df[df['post_character'] == char_id]
        df_opp = df[df['post_character'] == opponent_id]
       
        char_features_df, char_targets_df = df_char[feat_cols].shift(frame_delay).fillna(0), df_char[target_cols]
        opp_features_df, opp_targets_df = df_opp[feat_cols].shift(frame_delay).fillna(0), df_opp[target_cols].shift(frame_delay).fillna(0)

        
        features_np = np.concatenate([char_features_df.to_numpy()[:,0:2], opp_features_df.to_numpy()[:,0:2],\
                                      char_features_df.to_numpy()[:,2:], opp_features_df.to_numpy()[:,2:]], axis=1)
        # embedding_feature  reg_features  opp_target  

        
        char_button_values_np = char_targets_df['pre_buttons'].to_numpy().reshape(-1)
        opp_button_values_np = opp_targets_df['pre_buttons'].to_numpy().reshape(-1)
        char_targets_df.drop('pre_buttons', axis=1, inplace=True)
        opp_targets_df.drop('pre_buttons', axis=1, inplace=True)


        # import pdb 
        # pdb.set_trace()
        char_bin_cls_targets_np = self.proc_button_press(char_button_values_np)
        opp_bin_cls_targets_np = self.proc_button_press(opp_button_values_np)

        features_np = np.concatenate([features_np, opp_targets_df.to_numpy()], axis=1)
        scaler = StandardScaler()
        features_np[:,4:] = scaler.fit_transform(features_np[:,4:])
        features_np = np.concatenate([features_np, opp_bin_cls_targets_np], axis=1)

        cts_targets_np = char_targets_df.to_numpy()


        features_tensor, cts_targets_tensor, char_bin_class_targets_tensor = torch.from_numpy(features_np), torch.from_numpy(cts_targets_np), torch.from_numpy(char_bin_cls_targets_np)
        
        assert(features_tensor.shape[1] == 26 * 2 + 7 + 6)
        assert(cts_targets_tensor.shape[1] == 6)
        assert(char_bin_class_targets_tensor.shape[1] == 7)

        return features_tensor.float(), cts_targets_tensor.float(),  char_bin_class_targets_tensor.float()
        
