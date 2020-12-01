

from .common_parsing_logic import proc_frame, initialize_dataframe_dict, proc_df, data_pre_proc_mvp, align
from .slp_parser import SLPParser
from .dataset import SSBMDataset
from .. import controller_indices as c_idx

import pandas as pd
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import melee
import warnings
warnings.filterwarnings('ignore')
# convert a single frame to an input tensor
def convert_frame_to_input_tensor(frame, char_id, opponent_id):

    dataframe_dict = initialize_dataframe_dict(SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)
    proc_frame(dataframe_dict, frame, SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)
    df = pd.DataFrame.from_dict(dataframe_dict)
    df = data_pre_proc_mvp(df)
    features_tensor, _, _, _ = proc_df(df, char_id, opponent_id, 0, SSBMDataset.button_press_indicator_dim)
    return features_tensor

# used to store a queue of window_size frames during inference
class FrameContext(object):
    def __init__(self, window_size=60):
        if window_size <= 0:
            raise AttributeError(
                "FrameContext window_size must be positive"
            )
        self.window_size = window_size
        self.align_queue = []

    # pushes frame into align queue and returns the entire queue as a tensor for input
    def push_frame(self, frame, char_id, opponent_id):
        frame_features = convert_frame_to_input_tensor(frame, char_id, opponent_id)
        return align(self.align_queue, self.window_size, frame_features[0])

def button_combination_idx_to_bitmap(idx):
    result = []

    while idx > 0:
        result.append(idx % 2)
        idx //= 2
    result = list(reversed(result))
    result = [0] * (5 - len(result)) + result
    return result

def convert_output_tensor_to_command(cts_targets, button_targets, sample_top_n=3):

    button_targets = button_targets.reshape(-1)
    dist = Categorical(logits = button_targets)
    button_combination_idx = dist.sample().item()
    # print(button_combination_idx)
    bitmap = button_combination_idx_to_bitmap(button_combination_idx)

    state = {
        "main_stick": (cts_targets[0,0].item(), cts_targets[0,1].item()),
        "c_stick":  (cts_targets[0,2].item(), cts_targets[0,3].item()),
        "l_shoulder": cts_targets[0,4].item(),
        "r_shoulder": cts_targets[0,5].item(),
        "button": {
            melee.enums.Button.BUTTON_Y: 0,
            melee.enums.Button.BUTTON_X: bitmap[0],
            melee.enums.Button.BUTTON_B: bitmap[1],
            melee.enums.Button.BUTTON_A: bitmap[2],
            melee.enums.Button.BUTTON_L: 0,
            melee.enums.Button.BUTTON_R: bitmap[3],
            melee.enums.Button.BUTTON_Z: bitmap[4]
         }
    }


    return state

# takes in tuple of 8 index integers:
# buttons, stick coarse, stick fine, stick magn, cstick coarse, cstick fine, cstick magn, trigger
def convert_action_state_to_command(idx_state):
    buttons = c_idx.button.to_buttons(idx_state[0].item())
    stick_x, stick_y = c_idx.stick.to_stick(idx_state[1].item(), idx_state[2].item(), idx_state[3].item())
    cstick_x, cstick_y = c_idx.stick.to_stick(idx_state[4].item(), idx_state[5].item(), idx_state[6].item())
    trigger = c_idx.trigger.to_trigger(idx_state[7].item())
    return {
        'main_stick': (stick_x, stick_y),
        'c_stick': (cstick_x, cstick_y),
        'l_shoulder': 0,
        'r_shoulder': trigger,
        'button': {
            melee.enums.Button.BUTTON_Y: 0,
            melee.enums.Button.BUTTON_X: buttons[0],
            melee.enums.Button.BUTTON_B: buttons[1],
            melee.enums.Button.BUTTON_A: buttons[2],
            melee.enums.Button.BUTTON_L: 0,
            melee.enums.Button.BUTTON_R: buttons[3],
            melee.enums.Button.BUTTON_Z: buttons[4]
        }
    }
