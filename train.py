import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import relu,tanh,sigmoid,softmax
import pandas as pd
from dataset import SSBMDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import pdb
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def top_n_accuracy(preds, targets, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]

    correct = 0
    for i in range(targets.shape[0]):
        if targets[i] in best_n[i,:]:
            correct += 1 
    return correct / targets.shape[0]

def eval(model, val_dl, device, state):

    total_mse_loss = 0
    total_loss = 0
    button_preds = []
    button_labels = []
    num_batch = 0
    mse_crit = MSELoss(reduction='mean')
    ce_crit = CrossEntropyLoss(reduction='mean')
    model.eval()
    
    for batch in tqdm(val_dl, position=0, leave=True):
        num_batch += 1
        features, cts_targets, button_targets = batch
        features = features.to(device)
        cts_targets = cts_targets.to(device)
        button_targets = button_targets.to(device)
        with torch.no_grad():
            cts_o, logits_o = model(features)
            mse_loss = mse_crit(cts_o, cts_targets)
            total_mse_loss += mse_loss
            total_loss +=  ce_crit(logits_o, button_targets.long()) + mse_loss
        button_labels.append(button_targets)
        button_preds.append(softmax(logits_o))

    val_loss = total_loss.item()/num_batch
    mse = total_mse_loss.item()/num_batch

    state['val_loss'].append(val_loss)
    state["total_trigger_MSE"].append(mse)

    print(f'val_loss: {val_loss}v val_mse: {mse}')
    eval_button_preds(button_preds, button_labels, state)
   
    # report_button_cls_metrics(button_preds,button_targets)
        
def report_button_cls_metrics(preds, targets, thres=0.5):
   
    # import pdb 
    # pdb.set_trace()

    button_idx_to_name = {
        0: "X/Y",
        1: "B",
        2: "A",
        3: "L/R", 
        4: "Z"
    }
    preds = preds > thres 
   
    acc, recall, fscore, support = precision_recall_fscore_support(targets, preds)
    
    for button_idx in range(preds.shape[1]):
        if np.sum(targets[:, button_idx]) == 0:
            continue
        print(f'{button_idx_to_name[button_idx]} acc: {acc[button_idx]} recall: {recall[button_idx]} fscore: {fscore[button_idx]} support: {support[button_idx]}')
    
   
def eval_button_preds(button_preds, button_labels, state = None):
    button_preds = torch.cat(button_preds, dim=0).to('cpu').detach().numpy()
    button_labels = torch.cat(button_labels, dim=0).to('cpu').detach().numpy()

    f1_button_preds = np.array([ item for index, item in enumerate(button_preds) if button_labels[index] != 0])
    f1_button_labels = np.array([ item for item in button_labels if item != 0])
    
    top_3_acc, top_1_acc = top_n_accuracy(button_preds, button_labels, 3), top_n_accuracy(button_preds, button_labels, 1)
    if state:
        state["button_top_3_accuracy"].append(top_3_acc)
        state["button_top_1_accuracy"].append(top_1_acc)
    print(f'top_3_acc: {top_3_acc}  top_1_acc: {top_1_acc}')   
    print(f'f1_top_3_acc: {top_n_accuracy(f1_button_preds, f1_button_labels, 3)}  f1_top_1_acc: {top_n_accuracy(f1_button_preds, f1_button_labels, 1)}')
    # print(f'f2_top_3_acc: {top_n_accuracy(f2_button_preds, f2_button_labels, 3)}  f2_top_1_acc: {top_n_accuracy(f2_button_preds, f2_button_labels, 1)}')          
   

def train(model, trn_dl, val_dl, epoch, print_out_freq, device, pos_weigts):
    model.to(device)

    optim = Adam(model.parameters(), lr=0.0001)
    mse_crit, ce_crit = MSELoss(reduction='mean'), CrossEntropyLoss(reduction='mean')
    # button_press_thres = 0.5
    state = {
       "trn_loss": [],
       "val_loss": [],
       "total_trigger_MSE": [],
       "button_top_3_accuracy": [],
       "button_top_1_accuracy": []
    }
   
    
    for i in range(epoch):
        iter_num = 0 
        epoch_loss = 0
        button_labels = []
        button_preds = []
        model.train()
        for batch in tqdm(trn_dl, position=0, leave=True):
            iter_num += 1
            optim.zero_grad()
            features, cts_targets, button_targets = batch
            features = features.to(device)
            cts_targets = cts_targets.to(device)
            button_targets = button_targets.to(device)
            # import pdb 
            # pdb.set_trace()
            cts_o, logits_o = model(features)

            mse_loss = mse_crit(cts_o, cts_targets)
            # import pdb 
            # pdb.set_trace()
            ce_loss = ce_crit(logits_o, button_targets.reshape(-1).long())
            total_loss = mse_loss + ce_loss
            total_loss.backward()
            epoch_loss += total_loss.item()         
            optim.step()
            button_preds.append(softmax(logits_o))
            button_labels.append(button_targets)
             
            if iter_num % print_out_freq == 0:
                print(f'epoch: {i} trn_loss: {epoch_loss / iter_num}')

        final_trn_loss = epoch_loss / iter_num
        state['trn_loss'].append(final_trn_loss)
        print(f'end of {i}th epoch trn_loss: {final_trn_loss}')
        eval_button_preds(button_preds, button_labels, None)
        print(f'Eval epoch {i}')
        eval(model, val_dl, device, state)
        import pdb 
        pdb.set_trace()
    return state
                
        
            
            
            
                
        
            
            
            