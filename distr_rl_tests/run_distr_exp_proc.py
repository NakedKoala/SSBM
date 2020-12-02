from ssbm_bot.distr_rl.trainer import process_exps_loop

if __name__ == '__main__':
    process_exps_loop(
        exp_port=50000,
        return_port=50002,
        window_size=15,
        frame_delay=5,
    )
