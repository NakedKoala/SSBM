from slp_parser import SLPParser
from torch.utils.data import DataLoader
from dataset import SSBMDataset
from mvp_model import SSBM_MVP
from train import train
import torch
from slippi import Game
from infra_adaptor import convert_frame_to_input_tensor
from common_parsing_logic import proc_button_press


# Sample usage for the parser
# parser = SLPParser(src_dir="./dev_data_slp", dest_dir="./dev_data_csv")


## Sample usage for the SSBM_MVP model and training loop:
# model = SSBM_MVP(100)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Training on device: {device}')
# trn_ds = SSBMDataset(src_dir="./", char_id=2, opponent_id=1, device=device)
# # val_ds = SSBMDataset(src_dir="./dev_data_csv/valid", filter_char_id=2, device=device)
# trn_dl = DataLoader(trn_ds, batch_size=128, shuffle=True, num_workers=0)
# val_dl = DataLoader(val_ds, batch_size=256, shuffle=True, num_workers=0)


# import pdb 
# pdb.set_trace()
# train(model, trn_dl, trn_dl, 100,  5000, device)





if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trn_ds = SSBMDataset(src_dir="./", char_id=2, opponent_id=1, device=device)
    trn_dl = DataLoader(trn_ds, batch_size=256, shuffle=True, num_workers=0)
    model = SSBM_MVP(100)
    train(model, trn_dl, trn_dl, 100,  5000, device, [1] * 7)

    # for batch in trn_dl:
    #     features, cts_targets, bin_cls_targets = batch 
    #     import pdb 
    #     pdb.set_trace()

    # print(proc_button_press(368, SSBMDataset.button_press_indicator_dim))

    # slp_object = Game("./(YOTB) Fox vs Falcon (MN) [FD] Game_20200222T152806.slp")
    # frame = slp_object.frames[100]
    # feature_tensor = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)
    # import pdb 
    # pdb.set_trace()
