from slp_parser import SLPParser
from torch.utils.data import DataLoader
from dataset import SSBMDataset
from mvp_model import SSBM_MVP
from train import train
import torch
from slippi import Game
from infra_adaptor import convert_frame_to_input_tensor, convert_output_tensor_to_command
from common_parsing_logic import proc_button_press
from mvp_model import SSBM_MVP
import traceback
from tqdm import tqdm


# Sample usage for the parser
# if __name__ == '__main__':
    # parser = SLPParser(src_dir="./dev_data_slp", dest_dir="./dev_data_csv")


# Sample usage: training 

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# trn_ds = SSBMDataset(src_dir="./dev_data_csv/train", char_id=2, opponent_id=1, device=device)
# import pdb 
# pdb.set_trace()

# trn_dl = DataLoader(trn_ds, batch_size=256, shuffle=True, num_workers=0)
# model = SSBM_MVP(100)
# train(model, trn_dl, trn_dl, 100,  5000, device, [1] * 7)


'''
(tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  7.8693e-02,
        -1.1763e-01,  7.8442e-02, -1.1488e-01, -1.0408e+00, -9.5368e-01,
        -1.8943e+00, -1.3019e+01, -1.2495e+00, -3.6124e-01, -8.3657e-01,
        -1.1087e+00, -1.0408e+00, -1.0592e+00, -9.4360e-01, -1.0576e+00,
        -9.4497e-01, -2.6892e+00,  5.5534e-02,  6.7566e-02,  5.6069e-02,
         7.0975e-02, -1.2931e+00, -1.0150e+00, -9.4655e-01, -5.1376e+01,
        -9.6078e-01, -2.7596e-01,  0.0000e+00, -9.1549e-01, -1.2931e+00,
        -1.0074e+00, -9.9208e-01, -1.0071e+00, -9.9236e-01, -2.2482e+00,
         1.4232e-02,  1.7074e-01, -2.3667e-02,  1.1295e-01, -3.5252e-01,
        -1.9409e-01, -1.7975e-04,  2.2699e-02,  2.8348e-03, -3.3150e-02,
        -2.8865e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00]), tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0071, 0.0000]), tensor([0., 0., 0., 0., 0., 0., 0.]))


        (tensor([ 6.5000e+01,  6.5000e+01,  2.0000e+01,  2.0000e+01,  5.5080e-01,
         1.7786e-01,  5.1701e-01,  2.1476e-01, -3.8788e-01, -5.5532e-01,
         4.4986e-01,  2.4290e-01, -1.6042e-01, -3.6124e-01, -8.3657e-01,
         9.0196e-01, -3.8788e-01, -1.0592e+00,  1.0598e+00, -1.0576e+00,
         1.0582e+00,  1.2290e+00, -3.6928e-01,  6.7569e-02, -3.3652e-01,
         7.0978e-02,  1.7320e-01, -2.9103e-01,  2.0491e+00,  8.1468e-02,
         2.3809e+00, -2.7596e-01,  0.0000e+00, -9.1549e-01,  1.7320e-01,
         9.9265e-01, -9.9208e-01,  9.9294e-01, -9.9236e-01,  1.0790e+00,
         1.4232e-02,  1.7074e-01, -2.3667e-02,  1.1295e-01, -3.5252e-01,
         1.0429e-01,  1.7852e+00,  2.2699e-02,  2.8348e-03, -3.3150e-02,
        -2.8865e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00]), tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0429]), tensor([0., 0., 0., 0., 0., 0., 0.]))

'''



# Sample usage: infra adaptors

# model = SSBM_MVP(100)
# model.load_state_dict(torch.load('./weights/mvp_fit1_EP8_VL0078.pth',  map_location=lambda storage, loc: storage))
# model.eval()

slp_object = Game("./(YOTB) Fox vs Falcon (MN) [FD] Game_20200222T152806.slp")
# frame = slp_object.frames[100]
lst = []
for f in tqdm(slp_object.frames):
    try:
        feature_tensor = convert_frame_to_input_tensor(f, char_id=2, opponent_id=1)
        lst.append(feature_tensor)
    except:
        print(traceback.print_exc())
        import  pdb 
        pdb.set_trace()

# import  pdb 
# pdb.set_trace()




# cts_targets, bin_cls_targets = model(feature_tensor)
# print(convert_output_tensor_to_command(cts_targets, bin_cls_targets))


   
