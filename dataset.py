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
    def __init__(self, src_dir, filter_char_id, device, ds_type=None):
        torch.manual_seed(0)
        self.csv_files = [ os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if '.csv' in fname]
        self.features = []
        self.cts_targets = []
        self.bin_cls_targets = []
        
        for csv_path in tqdm(self.csv_files):
            try:
                features, cts_targets, bin_cls_targets = self.proc_csv(csv_path, filter_char_id)
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
        
        self.features.to(device)
        self.cts_targets.to(device)
        self.bin_cls_targets.to(device)

        if ds_type == 'dev':
            self.features = self.features[:1000]
            self.cts_targets = self.cts_targets[:1000]
            self.bin_cls_targets = self.bin_cls_targets[:1000]

    def __len__(self):
        return len(self.features)
       

    def __getitem__(self, idx):
        return self.features[idx], self.cts_targets[idx], self.bin_cls_targets[idx]
        

    def proc_csv(self, csv_path, filter_char_id):
        
        df = pd.read_csv(csv_path, index_col="frame_index")
        feat_cols = ['pre_state',  'post_state','pre_position_x', 'pre_position_y',  'post_position_x', 'post_position_y', \
                'post_damage', 'post_state_age', 'post_combo_count', 'post_shield', 'post_last_attack_landed', \
                'post_hit_stun', 'post_last_hit_by', 'post_airborne', 'pre_damage',  'post_direction_1', 'post_direction_-1', \
                'pre_direction_1', 'pre_direction_-1', 'post_stocks', 'post_ground_0.0', 'post_ground_1.0','post_ground_2.0',  'post_jumps_0', 'post_jumps_1',  \
                'post_jumps_2']
        target_cols = ['pre_joystick_x', 'pre_joystick_y',  'pre_cstick_x', 'pre_cstick_y', \
                       'pre_triggers_x', 'pre_triggers_y', 'pre_buttons']

        df = df[df['post_character'] == filter_char_id] #12
    
        features_df, targets_df = df[feat_cols].shift(1).fillna(0), df[target_cols] 
        
        button_values_np = targets_df['pre_buttons'].to_numpy().reshape(-1)
        targets_df.drop('pre_buttons', axis=1, inplace=True)
       
        features_np, cts_targets_np = features_df.to_numpy(), targets_df.to_numpy()

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
            try:
                assert(len(bits) == 12 )
            except:
                import pdb 
                pdb.set_trace()
                import sys 
                sys.exit()
           
            selected_bits = bits[4:-1]
            assert( len(selected_bits) == 7)
            # Keep valid keys: [d_left, d_right, d_down, d_up, z, r ,l, a, b, x, y, start]
            return np.array(selected_bits)
           
     
        f = np.vectorize(convert_button_value_to_indicator_vector, otypes=[np.ndarray])
        # try:
        bin_cls_targets_np = np.stack(f(button_values_np), axis=0) 
        # except:
        #     import pdb 
        #     pdb.set_trace()
        

        scaler = StandardScaler()

        features_np[:,2:] = scaler.fit_transform(features_np[:,2:])

        features_tensor, cts_targets_tensor, bin_class_targets_tensor = torch.from_numpy(features_np), torch.from_numpy(cts_targets_np), torch.from_numpy(bin_cls_targets_np)
        
        assert(features_tensor.shape[1] == 26)
        assert(cts_targets_tensor.shape[1] == 6)
        assert(bin_class_targets_tensor.shape[1] == 7)

        return features_tensor.float(), cts_targets_tensor.float(),  bin_class_targets_tensor.float()
        
