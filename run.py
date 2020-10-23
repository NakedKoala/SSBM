from slp_parser import SLPParser
from torch.utils.data import DataLoader
from dataset import SSBMDataset
from mvp_model import SSBM_MVP
from train import train
import torch

# Sample usage for the parser
# parser = SLPParser(src_dir="./dev_data_slp", dest_dir="./dev_data_csv")





## Sample usage for the SSBM_MVP model and training loop:
model = SSBM_MVP(100)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on device: {device}')
trn_ds = SSBMDataset(src_dir="./", char_id=2, opponent_id=1, device=device)
# val_ds = SSBMDataset(src_dir="./dev_data_csv/valid", filter_char_id=2, device=device)
trn_dl = DataLoader(trn_ds, batch_size=128, shuffle=True, num_workers=0)
# val_dl = DataLoader(val_ds, batch_size=256, shuffle=True, num_workers=0)


# import pdb 
# pdb.set_trace()
train(model, trn_dl, trn_dl, 100,  5000, device)





# if __name__ == '__main__':
#     parser = SLPParser(src_dir="./dev_data_slp", dest_dir="./dev_data_csv")
#     parser()