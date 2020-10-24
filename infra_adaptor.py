

from common_parsing_logic import proc_frame, initialize_dataframe_dict, proc_df, data_pre_proc_mvp
from slp_parser import SLPParser
from dataset import SSBMDataset
import pandas as pd
import torch

def convert_frame_to_input_tensor(frame, char_id, opponent_id):
    dataframe_dict = initialize_dataframe_dict(SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)

    proc_frame(dataframe_dict, frame, SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)
    df = pd.DataFrame.from_dict(dataframe_dict)
    df = data_pre_proc_mvp(df)
    features_tensor, _, _ = proc_df(df, char_id, opponent_id, 0, SSBMDataset.button_press_indicator_dim)
    return features_tensor

def convert_output_tensor_to_command(cts_targets, bin_cls_targets, button_press_thres = [0.5] * 7):
    probs = torch.sigmoid(bin_cls_targets)
    #TODO: Nathan please convert this dict to controller state
    # I can't make sure this run smoothly without running together with the infra part
    state = {
        "main_stick": (cts_targets[0,0], cts_targets[0,1]),
        "c_stick":  (cts_targets[0,2], cts_targets[0,3]),
        "l_shoulder": cts_targets[0,4],
        "r_shoulder": cts_targets[0,5],
        "button": {
            "Y": probs[0,0] > button_press_thres[0],
            "X": probs[0,1] > button_press_thres[1],
            "B": probs[0,2] > button_press_thres[2],
            "A": probs[0,3] > button_press_thres[3],
            "L": probs[0,4] > button_press_thres[4],
            "R": probs[0,5] > button_press_thres[5],
            "Z": probs[0,6] > button_press_thres[6]
         }   
    }
    

    return state 