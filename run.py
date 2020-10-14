from slp_parser import SLPParser
from torch.utils.data import DataLoader
from dataset import SSBMDataset
from mvp_model import SSBM_MVP
from train import train

## Sample usage for the parser
# parser = SLPParser(src_dir="./", dest_dir="./")
# parser(["test.slp"])




## Sample usage for the SSBM dataset

# ds = SSBMDataset(src_dir="./", filter_char_id=12)
# dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

# for batch in dl:
#     features, targets = batch
#     print(features)
#     print(features.shape)
#     print(targets)
#     print(targets.shape)
#     break


## Sample usage for the SSBM_MVP model and training loop:

model = SSBM_MVP(30)
ds = SSBMDataset(src_dir="./", filter_char_id=12, is_dev=True)
dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
train(model, dl, 500,  20)





