import torch
from torch.utils.data import DataLoader

from slippi import Game

import traceback
from tqdm import tqdm
import time
import numpy as np

from ssbm_bot.data.slp_parser import SLPParser
from ssbm_bot.data.dataset import SSBMDataset
from ssbm_bot.data.infra_adaptor import convert_frame_to_input_tensor, convert_output_tensor_to_command, FrameContext
from ssbm_bot.data.common_parsing_logic import proc_button_press
from ssbm_bot.model.mvp_model import SSBM_MVP
from ssbm_bot.model.mvp_model import SSBM_MVP
from ssbm_bot.model.lstm_model import SSBM_LSTM
from ssbm_bot.model.lstm_model_prob import SSBM_LSTM_Prob

from ssbm_bot.supervised_train.train_prob import train
import pandas as pd

# Sample usage for the parser
# if __name__ == '__main__':
#     parser = SLPParser(src_dir="./debug", dest_dir="./debug")
#     parser()


# Sample usage: training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # # SSBMDataset has a window_size argument for RNNs
trn_ds = SSBMDataset(src_dir="/home/william.wen42/dataset/2020_12_04_falcon_v_fox_fd_2.0.1_csv/train", char_id=2, window_size=1, include_opp_input=True, device=device)
import pdb; pdb.set_trace()
# trn_dl = DataLoader(trn_ds, batch_size=256, shuffle=True, num_workers=0)


# model = SSBM_LSTM(100, 50, hidden_size=256, num_layers=1, bidirectional=False, attention=True)

# model = SSBM_LSTM_Prob(100, 50, hidden_size=256, num_layers=1, bidirectional=False, attention=True)

# # # model = SSBM_MVP(100, 50)
# for batch in trn_dl:
#     feat, cts_targets, button_targets = batch
#     cts_o, logits_o = model(feat)
#     import pdb
#     pdb.set_trace()

# model = SSBM_MVP(100, 50)
# train(model, trn_dl, trn_dl, 20,  5000, device, [1] * 5)
# train(model, trn_dl, trn_dl, 1, 2, 5000, device)


# Sample usage: infra adaptors

# model = SSBM_MVP(100, 50)
# model.load_state_dict(torch.load('./weights/mvp_fit5_EP7_VL0349.pth',  map_location=lambda storage, loc: storage))
# model.eval()

# slp_object = Game("./(YOTB) Fox vs Falcon (MN) [FD] Game_20200222T152806.slp")


# cmd_lst = []
# for frame in tqdm(slp_object.frames):

#     feature_tensor = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)

#     cts_targets, button_targets = model(feature_tensor)

#     cmd_lst.append(convert_output_tensor_to_command(cts_targets, button_targets))







# # aligned version
# slp_object = Game("./(YOTB) Fox vs Falcon (MN) [FD] Game_20200222T152806.slp")
# frames = slp_object.frames[0:20]
# frame_ctx = FrameContext(window_size=10)
# # imitate streaming
# features_list = []
# for frame in frames:
#     features_list.append(frame_ctx.push_frame(frame, char_id=2, opponent_id=1))
#     # cts_targets, bin_cls_targets = model(features_list[-1])
#     # print(convert_output_tensor_to_command(cts_targets, bin_cls_targets))
# # play around with features_list...
# # e.g. print size
# print(features_list[0].shape)

# import pandas as pd
# import numpy as np
# from ssbm_bot.data.slp_parser import SLPParser
# from infrastructure import MeleeAI
# import melee
# from slippi import Game

# # Read all frames using libmelee and use parse_gamestate() to convert to py-slippi equivalent
# console = melee.Console(is_dolphin=False,
#                         allow_old_version=True,
#                         path="./debug/00_00_21 Captain Falcon + Fox (FD).slp"
#                         )
# console.connect()
# agent = MeleeAI()

# all_state = []
# while True:
#     gamestate = console.step()
#     # step() returns None when the file ends
#     if gamestate is None:
#         break
#     all_state.append(gamestate)

# converted_frames = []
# for gamestate in all_state:
#     converted_frames.append(agent.parse_gamestate(gamestate))

# dataframe_dict = {
#     'pre_joystick_x': [],
#     'pre_joystick_y': [],
#     'pre_cstick_x': [],
#     'pre_cstick_y': [],
#     'pre_triggers_x': [],
#     'pre_triggers_y': [],
#     'top1_idx': [],
#     'top2_idx': [],
#     'top3_idx': []
# }

# def record_prediction(dataframe_dict, cmd):
#     dataframe_dict['pre_joystick_x'].append(cmd["main_stick"][0])
#     dataframe_dict['pre_joystick_y'].append(cmd["main_stick"][1])
#     dataframe_dict['pre_cstick_x'].append(cmd["c_stick"][0])
#     dataframe_dict['pre_cstick_y'].append(cmd["c_stick"][1])
#     dataframe_dict['pre_triggers_x'].append(cmd["l_shoulder"])
#     dataframe_dict['pre_triggers_y'].append(cmd["r_shoulder"])
#     dataframe_dict['top1_idx'].append(cmd['top_idx'][0])
#     dataframe_dict['top2_idx'].append(cmd['top_idx'][1])
#     dataframe_dict['top3_idx'].append(cmd['top_idx'][2])

# slp_object = Game("./debug/00_00_21 Captain Falcon + Fox (FD).slp")

# frames = converted_frames

# model = SSBM_MVP(100, 50)
# model.load_state_dict(torch.load('./weights/mvp_fit5_EP7_VL0349.pth',  map_location=lambda storage, loc: storage))
# model.eval()

# for frame in tqdm(frames):

#     feature_tensor = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)

#     cts_targets, button_targets = model(feature_tensor)

#     cmd = convert_output_tensor_to_command(cts_targets, button_targets)
#     record_prediction(dataframe_dict, cmd)

# df = pd.DataFrame.from_dict(dataframe_dict)
# df.to_csv('./debug/melee_ver.csv')

