from slp_parser import SLPParser
from torch.utils.data import DataLoader
from dataset import SSBMDataset


## Sample usage for the parser
parser = SLPParser(src_dir="./", dest_dir="./")
parser(["test.slp"])

## Sample usage for the SSBM dataset

ds = SSBMDataset(src_dir="./", filter_char_id=12)
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

for batch in dl:
    features, targets = batch
    print(features)
    print(targets)
    break




