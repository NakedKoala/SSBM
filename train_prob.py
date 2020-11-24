import torch
from torch.nn.functional import softmax, log_softmax, one_hot, cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import controller_indices as c_idx
import math

ACCURACY_EPS = 1e-1
# each input is a tuple of 8 tensors
def get_correct(choices, targets):
    # are the buttons the same?
    correct_btn = torch.eq(targets[0], choices[0])
    # are the buttons the same and the sticks in the same general area?
    correct_coarse = correct_btn.clone()
    # are the buttons the same and the sticks very close to each other?
    correct_fine = correct_btn.clone()

    # fine correctness for individual parts
    correct_stick = torch.zeros_like(correct_btn)
    correct_cstick = torch.zeros_like(correct_btn)
    correct_trigger = torch.zeros_like(correct_btn)

    # check that continuous outputs are close enough
    for i in range(correct_btn.shape[0]):
        # get stick/trigger values
        def ch(idx):
            return choices[idx][i].item()
        def t(idx):
            return targets[idx][i].item()
        ch_stick_x, ch_stick_y = c_idx.stick.to_stick(ch(1), ch(2), ch(3))
        t_stick_x, t_stick_y = c_idx.stick.to_stick(t(1), t(2), t(3))
        ch_cstick_x, ch_cstick_y = c_idx.stick.to_stick(ch(4), ch(5), ch(6))
        t_cstick_x, t_cstick_y = c_idx.stick.to_stick(t(4), t(5), t(6))
        ch_trigger = c_idx.trigger.to_trigger(ch(7))
        t_trigger = c_idx.trigger.to_trigger(t(7))

        # inputs coarsely correct?
        # buttons must be correct
        if not correct_btn[i].item or not (
            c_idx.stick.close_coarse(ch_stick_x, ch_stick_y, t_stick_x, t_stick_y) and
            c_idx.stick.close_coarse(ch_cstick_x, ch_cstick_y, t_cstick_x, t_cstick_y) and
            c_idx.trigger.close_coarse(ch_trigger, t_trigger)
        ):
            correct_coarse[i] = False

        correct_stick[i] = c_idx.stick.close_fine(ch_stick_x, ch_stick_y, t_stick_x, t_stick_y)
        correct_cstick[i] = c_idx.stick.close_fine(ch_cstick_x, ch_cstick_y, t_cstick_x, t_cstick_y)
        correct_trigger[i] = c_idx.trigger.close_fine(ch_trigger, t_trigger)

        # inputs finely correct?
        # buttons must be correct
        if not correct_btn[i].item or not (
            correct_stick[i] and correct_cstick[i] and correct_trigger[i]
        ):
            correct_fine[i] = False


    return tuple(map(
        torch.sum,
        (correct_btn, correct_coarse, correct_fine, correct_stick, correct_cstick, correct_trigger)
    ))

# this should be moved to common_parsing_logic later
def convert_cts_to_idx(cts_targets):
    stick_x = cts_targets[:,0]
    stick_y = cts_targets[:,1]
    cstick_x = cts_targets[:,2]
    cstick_y = cts_targets[:,3]
    trigger, _ = torch.max(cts_targets[:,4:6], dim=1)
    # int64 required for one_hot
    indices = np.ndarray((cts_targets.shape[0], 7), dtype=np.int64)
    for i in range(cts_targets.shape[0]):
        idx = indices[i]
        idx[0], idx[1], idx[2] = c_idx.stick.to_index(stick_x[i].item(), stick_y[i].item())
        idx[3], idx[4], idx[5] = c_idx.stick.to_index(cstick_x[i].item(), cstick_y[i].item())
        idx[6] = c_idx.trigger.to_index(trigger[i].item())

    return tuple(torch.from_numpy(indices[:,i]) for i in range(indices.shape[1]))

def convert_idx_to_one_hot(indices_t):
    return torch.cat((
        one_hot(indices_t[0], num_classes=c_idx.stick.COARSE_N),
        one_hot(indices_t[1], num_classes=c_idx.stick.PRECISE_N),
        one_hot(indices_t[2], num_classes=c_idx.stick.MAGN_N),
        one_hot(indices_t[3], num_classes=c_idx.stick.COARSE_N),
        one_hot(indices_t[4], num_classes=c_idx.stick.PRECISE_N),
        one_hot(indices_t[5], num_classes=c_idx.stick.MAGN_N),
        one_hot(indices_t[6], num_classes=c_idx.trigger.NUM_INDICES)
    ), 1)


def train_eval_common_compute(model, batch, eval_behavior, compute_acc, device):
    features, cts_targets, button_targets = batch

    cts_idx = convert_cts_to_idx(cts_targets)
    all_targets = (button_targets.long(),) + cts_idx
    forced_action = torch.cat(
        tuple(target.unsqueeze(1) for target in all_targets),
        axis=1
    ).to(device)

    features = features.to(device)
    # import pdb
    # pdb.set_trace()

    # generate accuracy
    if compute_acc:
        with torch.no_grad():
            _, choices, _ = model(features, behavior=eval_behavior)
            # was the model's choices correct?
            choices = choices.to(button_targets.device) # assume button/cts_targets on the same device
            c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger = get_correct(
                tuple(choices[:,i] for i in range(choices.shape[1])),
                (button_targets,) + cts_idx
            )
            c_btn = c_btn.item()
            c_coarse = c_coarse.item()
            c_fine = c_fine.item()
            c_stick = c_stick.item()
            c_cstick = c_cstick.item()
            c_trigger = c_trigger.item()
    else:
        c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger = (0,) * 6

    # generate loss
    action_logits, _, _ = model(features, forced_action=forced_action)
    loss = torch.zeros(1).to(device)
    for logits, target in zip(action_logits, all_targets):
        loss += cross_entropy(logits, target.to(device))

    return loss, c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger

def train_eval_common_loop(model, dataloader, eval_behavior, device, compute_acc=True, print_out_freq=None, optim=None):
    model.to(device)
    if optim:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct_btn = 0
    correct_coarse = 0
    correct_fine = 0
    correct_stick = 0
    correct_cstick = 0
    correct_trigger = 0
    num_batch = 0
    num_total = 0

    def print_stats():
        print(f'iter: {num_batch}, '
            f'loss: {total_loss/num_batch}, ', flush=True)
        if compute_acc:
            print(f'button acc: {correct_btn/num_total}, '
                f'coarse acc: {correct_coarse/num_total}, '
                f'fine acc: {correct_fine/num_total}, '
                f'stick acc: {correct_stick/num_total}, '
                f'cstick acc: {correct_cstick/num_total}, '
                f'trigger acc: {correct_trigger/num_total}, ', flush=True)

    for batch in tqdm(dataloader, position=0, leave=True):
        num_batch += 1
        num_total += batch[0].shape[0]
        if optim:
            optim.zero_grad()
            loss, c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger = \
                train_eval_common_compute(model, batch, eval_behavior, compute_acc, device)
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                loss, c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger = \
                    train_eval_common_compute(model, batch, eval_behavior, compute_acc, device)

        total_loss += loss.item()
        correct_btn += c_btn
        correct_coarse += c_coarse
        correct_fine += c_fine
        correct_stick += c_stick
        correct_cstick += c_cstick
        correct_trigger += c_trigger

        if print_out_freq and num_batch % print_out_freq == 0:
            print_stats()

    print_stats()
    return total_loss, correct_btn, correct_coarse, correct_fine, correct_stick, correct_cstick, correct_trigger

def eval(model, val_dl, eval_behavior, device):
    train_eval_common_loop(model, val_dl, eval_behavior, device)

def train(model, trn_dl, val_dl, epoch, eval_behavior, print_out_freq, compute_acc, device, initial_lr=0.01):
    def lr_schedule(epoch):
        if epoch < 3:
            return 1
        elif epoch < 6:
            return 0.1
        elif epoch < 9:
            return 0.01
        return 0.001

    optim = Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = LambdaLR(optim, lr_lambda=[lr_schedule])

    for i in range(epoch):
        print(f'***TRAIN EPOCH {i}***', flush=True)
        train_eval_common_loop(model, trn_dl, eval_behavior, device, optim=optim, print_out_freq=print_out_freq, compute_acc=compute_acc)

        print(f'***EVAL EPOCH {i}***')
        eval(model, val_dl, eval_behavior, device)
        scheduler.step()
