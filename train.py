import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import relu,tanh,sigmoid
import pandas as pd
from dataset import SSBMDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import pdb
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def top_n_accuracy(preds, targets, n):
    best_n = torch.argsort(preds, dim=1)[:, -n:]

    correct = 0
    for i in range(targets.shape[0]):
        if targets[i] in best_n[i,:]:
            correct += 1 
    return correct / targets.shape[0]

def eval(model, val_dl, device, bce_crit):
    total_mse_loss = 0
    total_loss = 0
    button_preds = []
    button_targets = []
    num_batch = 0
    mse_crit = MSELoss(reduction='mean')
    model.eval()
    
    for batch in tqdm(val_dl):
        num_batch += 1
        features, cts_targets, bin_cls_targets = batch
        with torch.no_grad():
            cts_o, logits_o = model(features)
            mse_loss = mse_crit(cts_o, cts_targets)
            total_mse_loss += mse_loss
            total_loss +=  bce_crit(logits_o, bin_cls_targets) + mse_loss
        button_targets.append(bin_cls_targets)
        button_preds.append(sigmoid(logits_o))
    
    print(f'val_loss: {total_loss.item()/num_batch} val_mse: {total_mse_loss.item()/num_batch}')
    button_preds = torch.cat(button_preds, dim=0).to('cpu').detach().numpy()
    button_targets = torch.cat(button_targets, dim=0).to('cpu').detach().numpy()  
   
    report_button_cls_metrics(button_preds,button_targets)
        
def report_button_cls_metrics(preds, targets, thres=0.5):
   
    # import pdb 
    # pdb.set_trace()

    button_idx_to_name = {
        0: "Y",
        1: "X", 
        2: "B",
        3: "A",
        4: "L", 
        5: "R",
        6: "Z"
    }
    preds = preds > thres 
   
    acc, recall, fscore, support = precision_recall_fscore_support(targets, preds)
    
    for button_idx in range(preds.shape[1]):
        if np.sum(targets[:, button_idx]) == 0:
            continue
        print(f'{button_idx_to_name[button_idx]} acc: {acc[button_idx]} recall: {recall[button_idx]} fscore: {fscore[button_idx]} support: {support[button_idx]}')
    
   
def train(model, trn_dl, val_dl, epoch, print_out_freq, device, pos_weigts):
    model.to(device)

    optim = Adam(model.parameters(), lr=0.0001)
    mse_crit, bce_crit = MSELoss(reduction='mean'), BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(pos_weigts))
    button_press_thres = 0.5
    
    for i in range(epoch):
        iter_num = 0 
        epoch_loss = 0
        button_targets = []
        button_preds = []
        model.train()
        for batch in tqdm(trn_dl):
            iter_num += 1
            optim.zero_grad()
            features, cts_targets, bin_cls_targets = batch
            # import pdb 
            # pdb.set_trace()
            cts_o, logits_o = model(features)
            mse_loss = mse_crit(cts_o, cts_targets)
            # import pdb 
            # pdb.set_trace()
            bce_loss = bce_crit(logits_o, bin_cls_targets)
            total_loss = mse_loss + bce_loss
            total_loss.backward()
            epoch_loss += total_loss.item()         
            optim.step()
            button_preds.append(sigmoid(logits_o))
            button_targets.append(bin_cls_targets)
             
            if iter_num % print_out_freq == 0:
                print(f'epoch: {i} trn_loss: {epoch_loss / iter_num}')
        button_preds = torch.cat(button_preds, dim=0).to('cpu').detach().numpy()
        button_targets = torch.cat(button_targets, dim=0).to('cpu').detach().numpy()
        print(f'end of {i}th epoch trn_loss: {epoch_loss / iter_num}')
        report_button_cls_metrics(button_preds, button_targets)         
        print(f'Eval epoch {i}')
        eval(model, val_dl, device, bce_crit)
        # import pdb 
        # pdb.set_trace()
                
        
            
            
            
                
        
            
            
            