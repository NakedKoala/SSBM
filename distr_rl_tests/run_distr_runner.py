from ssbm_bot.model.lstm_model_prob import SSBM_LSTM_Prob
from ssbm_bot.distr_rl.a3c import A3CTrainer
from ssbm_bot.distr_rl.environment import SLPEnvironment
from ssbm_bot.distr_rl.runner import run_loop

import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError(
            ".slp file must be specified for runner"
        )
    model = SSBM_LSTM_Prob(
        action_embedding_dim = 100, button_embedding_dim = 50, hidden_size = 256,
        num_layers = 1, bidirectional=False, dropout_p=0.2, attention=False
    )
    environment = SLPEnvironment(
        frame_delay=1,
        slp_filename=sys.argv[1],
        device='cpu'
    )
    run_loop(
        trainer=A3CTrainer(model),
        environment=environment,
        trainer_ip=None,
        exp_port=50000,
        param_port=50001,
        adversary_port=50003,
        window_size=15,
        frame_delay=5,
        send_exp_every=60,
        check_model_upd_every=5,
        output_reward_every=600,
        max_episodes=None,
    )
