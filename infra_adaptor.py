

from common_parsing_logic import proc_frame, initialize_dataframe_dict, proc_df, data_pre_proc_mvp, align
from slp_parser import SLPParser
from dataset import SSBMDataset
import pandas as pd
import torch

# convert a single frame to an input tensor
def convert_frame_to_input_tensor(frame, char_id, opponent_id):
    dataframe_dict = initialize_dataframe_dict(SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)

    proc_frame(dataframe_dict, frame, SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)
    df = pd.DataFrame.from_dict(dataframe_dict)
    df = data_pre_proc_mvp(df)
    features_tensor, _, _ = proc_df(df, char_id, opponent_id, 0, SSBMDataset.button_press_indicator_dim)
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

def convert_output_tensor_to_command(cts_targets, bin_cls_targets, button_press_thres = [0.5] * 7):
    probs = torch.sigmoid(bin_cls_targets)
    #TODO: Nathan please convert this dict to controller state
    # I can't make sure this run smoothly without running together with the infra part
    state = {
        "main_stick": (cts_targets[0,0].item(), cts_targets[0,1].item()),
        "c_stick":  (cts_targets[0,2].item(), cts_targets[0,3].item()),
        "l_shoulder": cts_targets[0,4].item(),
        "r_shoulder": cts_targets[0,5].item(),
        "button": {
            "Y": probs[0,0].item() > button_press_thres[0],
            "X": probs[0,1].item() > button_press_thres[1],
            "B": probs[0,2].item() > button_press_thres[2],
            "A": probs[0,3].item() > button_press_thres[3],
            "L": probs[0,4].item() > button_press_thres[4],
            "R": probs[0,5].item() > button_press_thres[5],
            "Z": probs[0,6].item() > button_press_thres[6]
         }
    }


    return state
