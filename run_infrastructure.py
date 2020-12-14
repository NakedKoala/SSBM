import torch

from ssbm_bot.infrastructure import MeleeAI
from ssbm_bot.model.lstm_model_prob import SSBM_LSTM_Prob

if __name__ == "__main__":

    out_hidden_sizes=[
        [256, 128], # buttons
        [512, 256, 128], # stick coarse - NOTE - actually has 129 outputs
        [128, 128], # stick fine
        [128, 128], # stick magn
        [256, 128], # cstick coarse - NOTE - actually has 129 outputs
        [16, 16], # cstick fine
        [128, 128], # cstick magn
        [256, 128], # trigger
    ]

    model = SSBM_LSTM_Prob(
        action_embedding_dim = 100, hidden_size = 256,
        num_layers = 1, bidirectional=False, dropout_p=0.2,
        out_hidden_sizes=out_hidden_sizes, recent_actions=True,
        attention=False, include_opp_input=False, latest_state_reminder=True,
        own_dropout_p=1.0, opp_dropout_p=0.5, no_own_input=True
    )
    model.load_state_dict(torch.load('./weights/no_own_input_reminder.pth', map_location=lambda storage, loc: storage))

    agent = MeleeAI(action_frequence=None, window_size=60, frame_delay=15, include_opp_input=False, multiAgent=False, model=model, iso_path=None)
    agent.start()
