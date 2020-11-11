from slp_parser import SLPParser
from torch.utils.data import DataLoader
from dataset import SSBMDataset
from mvp_model import SSBM_MVP
from train import train
import torch
from slippi import Game
from infra_adaptor import convert_frame_to_input_tensor, convert_output_tensor_to_command, FrameContext
from common_parsing_logic import proc_button_press
from mvp_model import SSBM_MVP
from lstm_model import SSBM_LSTM
import traceback
from tqdm import tqdm


# Sample usage for the parser
# if __name__ == '__main__':
    # parser = SLPParser(src_dir="./dev_data_slp", dest_dir="./dev_data_csv")
    # parser()


# Sample usage: training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # SSBMDataset has a window_size argument for RNNs
trn_ds = SSBMDataset(src_dir="./", char_id=2, opponent_id=1, window_size=120, device=device)
trn_dl = DataLoader(trn_ds, batch_size=256, shuffle=True, num_workers=0)

model = SSBM_LSTM(100)
# for batch in trn_dl:
#     feat, cts_targets, bin_cls_targets = batch
#     cts_o, logits_o = model(feat)
#     import pdb 
#     pdb.set_trace()

# model = SSBM_MVP(100)
train(model, trn_dl, trn_dl, 5,  5000, device, [1] * 7)


# Sample usage: infra adaptors

"""
model = SSBM_MVP(100)
model.load_state_dict(torch.load('./weights/mvp_fit1_EP8_VL0078.pth',  map_location=lambda storage, loc: storage))
model.eval()

slp_object = Game("./(YOTB) Fox vs Falcon (MN) [FD] Game_20200222T152806.slp")
frame = slp_object.frames[100]
feature_tensor = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)
cts_targets, bin_cls_targets = model(feature_tensor)
print(convert_output_tensor_to_command(cts_targets, bin_cls_targets))
"""

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
