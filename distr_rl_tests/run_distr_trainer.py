from ssbm_bot.model.lstm_model_prob import SSBM_LSTM_Prob
from ssbm_bot.distr_rl.a3c import A3CTrainer
from ssbm_bot.distr_rl.trainer import train_loop

import torch

if __name__ == '__main__':
    model = SSBM_LSTM_Prob(
        action_embedding_dim = 100, button_embedding_dim = 50, hidden_size = 256,
        num_layers = 1, bidirectional=False, dropout_p=0.2, attention=False
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_loop(
        trainer=A3CTrainer(model),
        optimizer=optim,
        exp_port=50000,
        param_port=50001,
        exp_process_port=50002,
        window_size=15,
        frame_delay=5,
        save_path='./test.pt',
        send_param_every=10,
        output_loss_every=5,
        max_steps=None,
        save_every=None,
        spawn_proc=True,
    )
