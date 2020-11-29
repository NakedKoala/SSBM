from ssbm_bot.distr_rl.a3c import A3CTrainer
from ssbm_bot.distr_rl.environment import CartPoleEnvironment
from ssbm_bot.distr_rl.runner import run_loop
from ssbm_bot.distr_rl.cartpole_test_model import CartPoleModel

import sys

if __name__ == '__main__':
    model = CartPoleModel()
    environment = CartPoleEnvironment()
    run_loop(
        trainer=A3CTrainer(model),
        environment=environment,
        trainer_ip=None,
        exp_port=50000,
        param_port=50001,
        window_size=1,
        frame_delay=0,
        send_exp_every=300,
        check_model_upd_every=5,
        # output_reward_every=1000,
        output_eps_every=20,
        max_episodes=None,
    )
