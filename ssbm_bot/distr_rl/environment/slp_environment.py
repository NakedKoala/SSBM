from collections import deque
import slippi
import torch

from .environment import BaseEnvironment

from ...data.infra_adaptor import convert_frame_to_input_tensor


# reads environment from slippi file
# not tested yet...
# NOTE the environment outputs 1 state frame at a time.
# The caller is responsible for stacking multiple frames for model input.
# However, the environment IS responsible for frame delay, so the caller does not need to
# deal with frame delay on the states (but it needs to keep track of recent actions on its own).
class SLPEnvironment(BaseEnvironment):
    # NOTE convention is: frame_delay == 0 means agent acts ASAP;
    # equivalent to shifting by 1 when forming the dataset.
    def __init__(self, frame_delay, slp_filename, player_port, device):
        self.frame_delay = frame_delay
        self.cur_frame = 0
        self.slp = slippi.Game(slp_filename)
        self.stage = self.slp.start.stage

        self.player_port = player_port
        for i, port in enumerate(self.slp.start.players):
            if port is not None and i != player_port:
                self.adv_port = i
                break

        dummy_features = convert_frame_to_input_tensor(self.slp.frames[0], char_port=player_port, stage_id=self.stage, include_opp_input=False)
        # unwrap first dimension
        self.state_shape = dummy_features.shape[1:]

        self.reward_buffer = deque()
        self.recent_buffer = deque()
        self.adversary_buffer = deque()

        self.device = device

    # always return empty state
    def reset(self):
        self.cur_frame = 0
        self.reward_buffer.clear()
        self.recent_buffer.clear()
        self.adversary_buffer.clear()
        return torch.zeros(*self.state_shape, device=self.device), torch.zeros(*self.state_shape, device=self.device)

    # ignore action and pretend the agent input the action specified by the current frame.
    def step(self, action):
        if self.cur_frame < len(self.slp.frames):
            frame = self.slp.frames[self.cur_frame]

            # compute new state for now and save it
            new_state = convert_frame_to_input_tensor(frame, char_port=self.player_port, stage_id=self.stage, include_opp_input=False)[0]
            self.recent_buffer.append(new_state)

            new_state_adv = convert_frame_to_input_tensor(frame, char_port=self.adv_port, stage_id=self.stage, include_opp_input=False)[0]
            self.adversary_buffer.append(new_state_adv)

            # compute reward function for now and save it
            # test reward function: +1 for stock taken, -1 for stock loss
            # (note: we don't actually check who won for this test environment)
            reward = 0
            if self.cur_frame > 0:
                prev_frame = self.slp.frames[self.cur_frame-1]
                for i, port in enumerate(frame.ports):
                    if port:
                        # hardcode captain falcon as agent
                        # other character is not falcon
                        reward_change = -1 if port.leader.post.character == 2 else 1
                        if port.leader.post.stocks < prev_frame.ports[i].leader.post.stocks:
                            # print("reward change", reward_change)
                            reward += reward_change

            # is the game done?
            if self.cur_frame == len(self.slp.frames)-1:
                self.reward_buffer.append((reward, True))
            else:
                self.reward_buffer.append((reward, False))

        self.cur_frame += 1

        # not enough recent states - return 0
        if self.cur_frame < len(self.slp.frames) and len(self.recent_buffer) <= self.frame_delay:
            return torch.zeros(*self.state_shape, device=self.device), torch.zeros(*self.state_shape, device=self.device), 0, False

        # get current delayed state and reward
        if len(self.recent_buffer) > 0:
            delayed_state_t = self.recent_buffer.popleft()
        else:
            delayed_state_t = torch.zeros(*self.state_shape, device=self.device)

        if len(self.adversary_buffer) > 0:
            delayed_adv_state_t = self.adversary_buffer.popleft()
        else:
            delayed_adv_state_t = torch.zeros(*self.state_shape, device=self.device)

        if len(self.reward_buffer) > 0:
            delayed_reward, delayed_done = self.reward_buffer.popleft()
        else:
            delayed_reward, delayed_done = 0, True

        return delayed_state_t, delayed_adv_state_t, delayed_reward, delayed_done
