import pandas as pd
import numpy as np
import torch
import os
import pdb

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import traceback
from common_parsing_logic import proc_df, align



class SSBMDataset(Dataset):



    val_ratio = 0.2
    button_press_indicator_dim = 12
    frame_delay = 2
    def __init__(self, src_dir, char_id, opponent_id, device, window_size=0, ds_type=None):
        torch.manual_seed(0)
        self.csv_files = [ os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if '.csv' in fname]
        self.features = []
        self.cts_targets = []
        self.bin_cls_targets = []
        self.window_size = window_size

        for csv_path in tqdm(self.csv_files):
            try:
                df = pd.read_csv(csv_path, index_col="frame_index")
                features, cts_targets, bin_cls_targets = proc_df(df, char_id, opponent_id, SSBMDataset.frame_delay, SSBMDataset.button_press_indicator_dim)

                # align features
                if window_size > 0:
                    aligned_features_list = []
                    align_queue = []
                    for example in features:
                        aligned_features_list.append(align(align_queue, window_size, example))
                    aligned_features = torch.stack(aligned_features_list)
                else:
                    # window_size == 0 means don't use windows at all - for ANNs for example
                    aligned_features = features

                self.features.append(aligned_features)
                self.cts_targets.append(cts_targets)
                self.bin_cls_targets.append(bin_cls_targets)
            except:
                print(f'failed to load {csv_path}')
                traceback.print_exc()

        self.features = torch.cat(self.features, dim=0)
        self.cts_targets = torch.cat(self.cts_targets, dim=0)
        self.bin_cls_targets = torch.cat(self.bin_cls_targets, dim=0)

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

