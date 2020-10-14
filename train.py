import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss
from torch.nn.functional import relu,tanh
import pandas as pd
from dataset import SSBMDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import pdb

def train(model, dl, epoch, print_out_freq):
    
    optim = Adam(model.parameters(), lr=0.0001)
    mse_crit, cross_entrophy_crit = MSELoss(reduction='mean'), CrossEntropyLoss(reduction='mean')
    
    for i in range(epoch):
        iter_num = 0 
        epoch_loss = 0
        for batch in dl:
            iter_num += 1
            optim.zero_grad()
            features, targets = batch
            cts_o, logits_o = model(features)
            mse_loss = mse_crit(cts_o, targets[:,:-1])
            cross_entrophy_loss = cross_entrophy_crit(logits_o, targets[:,-1].long())
            total_loss = mse_loss + cross_entrophy_loss
            total_loss.backward()
            epoch_loss += total_loss.item() 
            optim.step()
            
            if iter_num % print_out_freq == 0:
                print(f'loss: {epoch_loss / iter_num}')
                
        print(f'end of {i}th epoch loss: {epoch_loss / iter_num}')
                
        
            
            
            