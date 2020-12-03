import pandas as pd
import numpy as np
import torch
import bisect
import os
import pdb

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import traceback

from .common_parsing_logic import proc_df, align, scale


def find_le_idx(a, x):
    'Find index of rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return i-1
    raise ValueError

class SSBMDataset(Dataset):


    val_ratio = 0.2
    button_press_indicator_dim = 12
    def __init__(self, src_dir, char_id, opponent_id, device, window_size=0, frame_delay=15, output_recent_actions=False, ds_type=None, stage_id=32):
        torch.manual_seed(0)
        self.csv_files = [ os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if '.csv' in fname]
        self.features_per_game = []
        self.frame_splits = [0]
        self.cts_targets = []
        self.bin_cls_targets = []
        self.recent_actions = []
        self.window_size = window_size
        self.frame_delay = frame_delay
        self.output_recent_actions=output_recent_actions
        self.device = device
        
     
        for csv_path in tqdm(self.csv_files, position=0, leave=True):
            try:
                df = pd.read_csv(csv_path, index_col="frame_index")
                features, cts_targets, bin_cls_targets, recent_actions = proc_df(df, char_id, opponent_id,  stage_id, self.frame_delay, SSBMDataset.button_press_indicator_dim)

                self.features_per_game.append(features)
               
                # prefix sum on frame_splits for indexing
                self.frame_splits.append(self.frame_splits[-1] + len(features))

                self.cts_targets.append(cts_targets)
                self.bin_cls_targets.append(bin_cls_targets)
                self.recent_actions.append(recent_actions)

                if ds_type == 'dev' and self.frame_splits[-1] > 1000:
                    break
            except:
                print(f'failed to load {csv_path}')
                traceback.print_exc()

        self.cts_targets = torch.cat(self.cts_targets, dim=0)
        self.bin_cls_targets = torch.cat(self.bin_cls_targets, dim=0)
        self.recent_actions = torch.cat(self.recent_actions, dim=0)
      
       

        # move per-game features to device
        for features in self.features_per_game:
            features.to(device)

        self.cts_targets =  self.cts_targets.to(device)
        self.bin_cls_targets =  self.bin_cls_targets.to(device)
        self.recent_actions = self.recent_actions.to(device)

        if ds_type == 'dev':
            max_frames = 1000
            # find which game contains the last frame
            max_game_idx = find_le_idx(self.frame_splits, max_frames - 1)
            if max_game_idx < len(self.frame_splits)-1:
                # too many frames - need to cut
                self.frame_splits = self.frame_splits[:max_game_idx+1]
                max_frame_idx = max_frames - self.frame_splits[-1]
                self.frame_splits.append(max_frames)
                self.features_per_game[max_game_idx] = self.features_per_game[max_game_idx][:max_frame_idx]
            self.cts_targets = self.cts_targets[:max_frames]
            self.bin_cls_targets = self.bin_cls_targets[:max_frames]
            self.recent_actions = self.recent_actions[:max_frames]

    def __len__(self):
        return self.frame_splits[-1]


    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        # find which game and which frame in that game
        game_idx = find_le_idx(self.frame_splits, idx)
        # print(game_idx)
        frame_idx = idx - self.frame_splits[game_idx]
        first_frame = max(0, frame_idx - self.window_size + 1)
        frame_features = self.features_per_game[game_idx][first_frame:frame_idx + 1]

        recent_actions = []
        if self.frame_delay > 0:
            first_recent_action_idx = idx - self.frame_delay
            if frame_idx - self.frame_delay < 0:
                for _ in range(self.frame_delay - frame_idx):
                    recent_actions.append(torch.unsqueeze(torch.zeros_like(self.recent_actions[0]), 0))
                first_recent_action_idx = idx - frame_idx
            recent_actions.append(self.recent_actions[first_recent_action_idx:idx])
            recent_actions = torch.cat(recent_actions)
        else:
            recent_actions = None

        # at least one frame must exist
        output = ()
        if self.window_size > 1:
            assert(frame_features.shape[0] > 0)
            if frame_features.shape[0] < self.window_size:
                # prepend with zeroes
                features_list = []
                for _ in range(self.window_size - frame_features.shape[0]):
                    # add zero tensor of size (1,) + frame_features[0].shape
                    features_list.append(torch.unsqueeze(torch.zeros_like(frame_features[0]), 0))
                features_list.append(frame_features)
                frame_features = torch.cat(features_list)

            output = (frame_features, self.cts_targets[idx], self.bin_cls_targets[idx])
        else:
            # import pdb
            # pdb.set_trace()
            output = ( frame_features.squeeze(0), self.cts_targets[idx], self.bin_cls_targets[idx])

        if self.output_recent_actions:
            output = output + (recent_actions,)

        return output
