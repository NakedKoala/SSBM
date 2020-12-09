

from .common_parsing_logic import proc_frame, initialize_dataframe_dict, proc_df, data_pre_proc_mvp, align
from .slp_parser import SLPParser
from .dataset import SSBMDataset
from .. import controller_indices as c_idx

from collections import deque
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import melee
import warnings
warnings.filterwarnings('ignore')
# convert a single frame to an input tensor
def convert_frame_to_input_tensor(frame, char_port, stage_id, include_opp_input=True):

    dataframe_dict = initialize_dataframe_dict(SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)
    proc_frame(dataframe_dict, frame, stage_id, SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)
    df = pd.DataFrame.from_dict(dataframe_dict)
    df = data_pre_proc_mvp(df)
    features_tensor, _, _, _ = proc_df(df, char_port, 0, SSBMDataset.button_press_indicator_dim, include_opp_input=include_opp_input)
    return features_tensor

# used to store a queue of window_size stale frames for inference
# also holds a queue of frame_delay last recent actions
# NOTE does not do frame delay! must hand in stale tensors.
class FrameContext(object):
    def __init__(self, window_size=60, frame_delay=15, include_opp_input=False):
        if window_size <= 0:
            raise AttributeError(
                "FrameContext window_size must be positive"
            )
        if frame_delay < 0:
            raise AttributeError(
                "FrameContext frame_delay must be non-negative"
            )
        self.window_size = window_size
        self.frame_delay = frame_delay
        self.state_queue = deque()
        self.action_queue = deque()
        self.include_opp_input = include_opp_input

    # NOTE input should not have a batch dimension.
    def push_tensor(self, cur_state, last_action):
        cur_state_t = align(self.state_queue, self.window_size, cur_state, action=False, include_opp_input=self.include_opp_input).unsqueeze(dim=0)
        if self.frame_delay > 0:
            action_t = torch.zeros(7).float()
            if last_action is not None:
                stick_x, stick_y = c_idx.stick.to_stick(*last_action[1:4])
                cstick_x, cstick_y = c_idx.stick.to_stick(*last_action[4:7])
                trigger = c_idx.trigger.to_trigger(last_action[7])
                # recent action format is:
                # buttons, stick x, stick y, cstick x, cstick y, trigger x, trigger y
                action_t = torch.Tensor([last_action[0], stick_x, stick_y, cstick_x, cstick_y, trigger, 0.0]).float()
            action_align_t = align(self.action_queue, self.frame_delay, action_t, action=True, include_opp_input=self.include_opp_input).unsqueeze(dim=0)
        else:
            action_align_t = None

        return cur_state_t, action_align_t

    # pushes frame/last action into queue and returns the entire queue as a tensor for input
    def push_frame(self, frame, char_port, stage_id, include_opp_input, last_action=None):
        frame_features = convert_frame_to_input_tensor(frame, char_port, stage_id, include_opp_input=include_opp_input)[0]
        return self.push_tensor(frame_features, last_action)

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
    top_idx = np.argsort(button_targets.detach().numpy())[::-1][0:sample_top_n]
    # print(button_combination_idx)
    bitmap = button_combination_idx_to_bitmap(button_combination_idx)

    state = {
        "main_stick": (cts_targets[0,0].item(), cts_targets[0,1].item()),
        "c_stick":  (cts_targets[0,2].item(), cts_targets[0,3].item()),
        "l_shoulder": cts_targets[0,4].item(),
        "r_shoulder": cts_targets[0,5].item(),
        "top_idx": top_idx,
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
