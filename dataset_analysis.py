# measures how frequently the targets for the current frame match the last recent actions

from ssbm_bot.data.dataset import SSBMDataset
from ssbm_bot.supervised_train import train_prob

import torch

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
trn_ds = SSBMDataset(src_dir="../test-dataset-csv-2", char_id=2, window_size=60, frame_delay=15, device=device, output_recent_actions=True, include_opp_input=False)

num_ex = 0
num_match = 0
for ex in trn_ds:
    features, cts_targets, btn_target, recent_action = ex
    # look over a range of recent frames to look for one identical to the target
    for i in range(1, 2):
        recent_action_frame = recent_action[-i]
        cts_target_categories = train_prob.convert_cts_to_idx(cts_targets.unsqueeze(dim=0))
        recent_action_cts_categories = train_prob.convert_cts_to_idx(recent_action_frame[1:].unsqueeze(dim=0))
        if btn_target == recent_action_frame[0] and cts_target_categories == recent_action_cts_categories:
            num_match += 1
            break
    num_ex += 1

print(num_match, num_ex)

