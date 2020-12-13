import torch
from torch.nn.functional import softmax, log_softmax, one_hot, cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import math

from .. import controller_indices as c_idx

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


def train_eval_common_compute(model, batch, held_input_loss_factor, eval_behavior, compute_acc, device):
    if len(batch) == 4:
        features, cts_targets, button_targets, held_input = batch
        inputs = features.to(device)
    else:
        features, cts_targets, button_targets, held_input, recent_actions = batch
        inputs = (features.to(device), recent_actions.to(device))

    button_targets = button_targets.long()

    loss_factor_t = (held_input_loss_factor-1.0) * held_input.to(device) + 1.0

    cts_idx = convert_cts_to_idx(cts_targets)
    all_targets = (button_targets,) + cts_idx
    forced_action = torch.cat(
        tuple(target.unsqueeze(1) for target in all_targets),
        axis=1
    ).to(device)

    # generate accuracy
    if compute_acc:
        with torch.no_grad():
            _, choices, _ = model(inputs, behavior=eval_behavior)

            # did the choices match the target categories?
            # use min, so any non-equal row will result in 0 row.
            category_match, _ = torch.min(
                (choices == forced_action), dim=1
            )
            c_match = category_match.sum().item()

            # was the model's choices correct logically?
            # i.e. the converted choice -> input is close to the actual target inputs
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
        c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger, c_match = (0,) * 7

    # generate loss
    action_logits, _, _ = model(inputs, forced_action=forced_action)
    loss = torch.zeros(1).to(device)
    for logits, target in zip(action_logits, all_targets):
        loss_unreduced = cross_entropy(logits, target.to(device), reduction='none')
        loss += (loss_unreduced * loss_factor_t).mean()

    return loss, c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger, c_match

_MOVING_AVG_FACTOR = 0.99
def train_eval_common_loop(
    model, dataloader, held_input_loss_factor, eval_behavior, device, compute_acc=True,
    print_out_freq=None, optim=None, stats_tracker=None, short_circuit=None
):
    model.to(device)
    if optim:
        model.train()
    else:
        model.eval()

    total_loss = None
    correct_btn = None
    correct_coarse = None
    correct_fine = None
    correct_stick = None
    correct_cstick = None
    correct_trigger = None
    correct_match = None
    num_batch = 0

    def print_stats():
        print(f'iter: {num_batch}, '
            f'loss: {total_loss}, ', flush=True)
        if compute_acc:
            print(f'button acc: {correct_btn}, '
                f'coarse acc: {correct_coarse}, '
                f'fine acc: {correct_fine}, '
                f'stick acc: {correct_stick}, '
                f'cstick acc: {correct_cstick}, '
                f'trigger acc: {correct_trigger}, '
                f'match acc: {correct_match} ', flush=True)

    def upd_moving_avg(old, new):
        if old is None:
            return new
        return old * _MOVING_AVG_FACTOR + new * (1.0 - _MOVING_AVG_FACTOR)

    for batch in tqdm(dataloader, position=0, leave=True):
        num_batch += 1

        batch_size = batch[0].shape[0]
        if optim:
            optim.zero_grad()
            loss, c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger, c_match = \
                train_eval_common_compute(model, batch, held_input_loss_factor, eval_behavior, compute_acc, device)
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                loss, c_btn, c_coarse, c_fine, c_stick, c_cstick, c_trigger, c_match = \
                    train_eval_common_compute(model, batch, held_input_loss_factor, eval_behavior, compute_acc, device)

        total_loss = upd_moving_avg(total_loss, loss.item())
        correct_btn = upd_moving_avg(correct_btn, c_btn/batch_size)
        correct_coarse = upd_moving_avg(correct_coarse, c_coarse/batch_size)
        correct_fine = upd_moving_avg(correct_fine, c_fine/batch_size)
        correct_stick = upd_moving_avg(correct_stick, c_stick/batch_size)
        correct_cstick = upd_moving_avg(correct_cstick, c_cstick/batch_size)
        correct_trigger = upd_moving_avg(correct_trigger, c_trigger/batch_size)
        correct_match = upd_moving_avg(correct_match, c_match/batch_size)

        if stats_tracker is not None:
            stats_tracker['total_loss'].append(loss.item())
            stats_tracker['correct_btn'].append(c_btn/batch_size)
            stats_tracker['correct_coarse'].append(c_coarse/batch_size)
            stats_tracker['correct_fine'].append(c_fine/batch_size)
            stats_tracker['correct_stick'].append(c_stick/batch_size)
            stats_tracker['correct_cstick'].append(c_cstick/batch_size)
            stats_tracker['correct_trigger'].append(c_trigger/batch_size)
            stats_tracker['correct_match'].append(c_match/batch_size)

        if print_out_freq and num_batch % print_out_freq == 0:
            print_stats()

        if short_circuit is not None and num_batch >= short_circuit:
            break

    print_stats()

def create_stats_tracker():
    stats_tracker = {}
    stats_tracker['total_loss'] = []
    stats_tracker['correct_btn'] = []
    stats_tracker['correct_coarse'] = []
    stats_tracker['correct_fine'] = []
    stats_tracker['correct_stick'] = []
    stats_tracker['correct_cstick'] = []
    stats_tracker['correct_trigger'] = []
    stats_tracker['correct_match'] = []
    return stats_tracker

def eval(model, val_dl, held_input_loss_factor, eval_behavior, device, stats_tracker=None):
    train_eval_common_loop(model, val_dl, held_input_loss_factor, eval_behavior, device, stats_tracker=stats_tracker)

def train(
    model, trn_dl, val_dl, epochs, held_input_loss_factor, eval_behavior, print_out_freq, compute_acc, device,
    initial_lr=0.01, track_stats=True, short_circuit=None, lr_schedule=None
):
    def def_lr_schedule(epoch):
        if epoch < 3:
            return 1
        elif epoch < 6:
            return 0.1
        elif epoch < 9:
            return 0.01
        return 0.001

    if lr_schedule is None:
        lr_schedule = def_lr_schedule

    optim = Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = LambdaLR(optim, lr_lambda=[lr_schedule])

    if track_stats:
        train_stats = create_stats_tracker()
        eval_stats = create_stats_tracker()
    else:
        train_stats = None
        eval_stats = None

    for i in range(epochs):
        print(f'***TRAIN EPOCH {i}***', flush=True)
        train_eval_common_loop(
            model, trn_dl, held_input_loss_factor, eval_behavior, device, optim=optim,
            print_out_freq=print_out_freq, compute_acc=compute_acc, stats_tracker=train_stats,
            short_circuit=short_circuit
        )

        print(f'***EVAL EPOCH {i}***', flush=True)

        if track_stats:
            eval_len = len(eval_stats['total_loss'])

        eval(model, val_dl, held_input_loss_factor, eval_behavior, device, stats_tracker=eval_stats)

        if track_stats:
            for stat, lst in eval_stats.items():
                epoch_mean = sum(lst[eval_len:])/(len(lst)-eval_len)
                new_lst = lst[:eval_len] + [epoch_mean]
                eval_stats[stat] = new_lst

        scheduler.step()

    return (train_stats, eval_stats)
