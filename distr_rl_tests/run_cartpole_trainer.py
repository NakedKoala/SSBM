from ssbm_bot.distr_rl.a3c import A3CTrainer
from ssbm_bot.distr_rl.trainer import train_loop
from ssbm_bot.distr_rl.cartpole_test_model import CartPoleModel

import torch

if __name__ == '__main__':
    model = CartPoleModel()
    optim = torch.optim.Adam(model.parameters(), lr=3e-2)
    train_loop(
        trainer=A3CTrainer(model),
        optimizer=optim,
        exp_port=50000,
        param_port=50001,
        exp_process_port=50002,
        window_size=1,
        save_path='./test.pt',
        send_param_every=10,
        output_loss_every=5,
        max_steps=None,
        save_every=None,
        spawn_proc=True,
    )
