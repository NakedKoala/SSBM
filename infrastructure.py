#!/usr/bin/python3
import sys
import melee
import torch
import time

from ssbm_bot.model.mvp_model import SSBM_MVP
from ssbm_bot.model.lstm_model import SSBM_LSTM
from ssbm_bot.model.lstm_model_prob import SSBM_LSTM_Prob
from ssbm_bot.data.infra_adaptor import convert_frame_to_input_tensor, convert_output_tensor_to_command, FrameContext, convert_action_state_to_command
from collections import deque

DOLPHIN_EXE_PATH = '/Applications/Slippi Dolphin.app' # change to yours

# def record_prediction(dataframe_dict, cmd):
#     dataframe_dict['pre_joystick_x'].append(cmd["main_stick"][0])
#     dataframe_dict['pre_joystick_y'].append(cmd["main_stick"][1])
#     dataframe_dict['pre_cstick_x'].append(cmd["c_stick"][0])
#     dataframe_dict['pre_cstick_y'].append(cmd["c_stick"][1])
#     dataframe_dict['pre_triggers_x'].append(cmd["l_shoulder"])
#     dataframe_dict['pre_triggers_y'].append(cmd["r_shoulder"])
#     dataframe_dict['top1_idx'].append(cmd['top_idx'][0])
#     dataframe_dict['top2_idx'].append(cmd['top_idx'][1])
#     dataframe_dict['top3_idx'].append(cmd['top_idx'][2])


class MeleeAI:

    class MeleeFrame:
        class Object(object):
            pass

        def __init__(self, frameIndex):
            self.ports = []
            self.index = frameIndex


    def __init__(self, action_frequence, window_size, frame_delay):
        # self.model = SSBM_MVP(100, 50)
        # self.model.load_state_dict(torch.load('./weights/mvp_fit5_EP7_VL0349.pth',  map_location=lambda storage, loc: storage))
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
        self.model = SSBM_LSTM_Prob(
            action_embedding_dim = 100, hidden_size = 256,
            num_layers = 1, bidirectional=False, dropout_p=0.2,
            out_hidden_sizes=out_hidden_sizes, recent_actions=False,
            attention=False, include_opp_input=False, latest_state_reminder=False,
        )
        self.model.load_state_dict(torch.load('./weights/lstm_recent_action_no_opp_input_delay_0_2020_12_05.pth',  map_location=lambda storage, loc: storage))
        self.model.eval()
        # self.model.load_state_dict(torch.load('./weights/lstm_fd1_wz30_noshuffle_reminder.pth',  map_location=lambda storage, loc: storage))

        self.frame_delay = frame_delay
        self.window_size = window_size
        self.frame_ctx = FrameContext(window_size=window_size, frame_delay=frame_delay)
        self.state_buffer = deque()
        self.last_model_output = None

        self.action_frequence = action_frequence
        self.time = 0

        self.frames = []

        self.previousPosition = [(0, 0), (0, 0)]
        self.previousDamage = [0, 0]
        self.previousFacing = [True, False]
        self.previousAction = [melee.enums.Action.UNKNOWN_ANIMATION, melee.enums.Action.UNKNOWN_ANIMATION]
        self.frameCount = -1

        self.button_dict = {
            "START": 2 ** 12,
            "Y": 2 ** 11,
            "X": 2 ** 10,
            "B": 2 ** 9,
            "A": 2 ** 8,
            "L": 2 ** 6,
            "R": 2 ** 5,
            "Z": 2 ** 4,
            "D_UP": 2 ** 3,
            "D_DOWN": 2 ** 2,
            "D_LEFT": 2 ** 1,
            "D_RIGHT": 2 ** 0
        }

        self.console = melee.Console(path=DOLPHIN_EXE_PATH, blocking_input=True)
        self.console.render = True
        self.controller = melee.Controller(self.console, 2)
        self.console.run()

        print("Connecting to console...")
        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            sys.exit(-1)

        print("Connecting controller to console...")
        if not self.controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        print("Controller connected")
        self.dataframe_dict = {
            'pre_joystick_x': [],
            'pre_joystick_y': [],
            'pre_cstick_x': [],
            'pre_cstick_y': [],
            'pre_triggers_x': [],
            'pre_triggers_y': [],
            'top1_idx': [],
            'top2_idx': [],
            'top3_idx': []
            }

    def next_state(self):
        # print(time.time() - self.time)
        self.time = time.time()
        return self.console.step()  # get frame data

    def input_model_commands(self, frame, stage_id):
        self.state_buffer.append(frame)

        if self.frame_delay > 0:
            if len(self.state_buffer) <= self.frame_delay:
                stale_frame = torch.zeros(self.model.in_dim)
                action = None if self.frame_delay == 0 else torch.zeros(1, self.frame_delay, 7)
                model_input = (torch.zeros(1, self.window_size, self.model.in_dim), action)
            else:
                stale_frame = self.state_buffer.popleft()
                model_input = self.frame_ctx.push_frame(frame, char_port=1, stage_id=stage_id, include_opp_input=False, last_action=self.last_model_output)
        else:
            model_input, _ = self.frame_ctx.push_frame(frame, char_port=1, stage_id=stage_id, include_opp_input=False, last_action=None)


        if self.action_frequence == None or self.frameCount % self.action_frequence == 0:
            # action_frequence == None -> we want action every frame
            _, choices, _ = self.model(model_input, behavior=0)
            self.last_model_output = choices[0]
            commands = convert_action_state_to_command(choices[0])

        if self.action_frequence != None and self.frameCount % (self.action_frequence + 2) == 0:
            # a/b/x/z only holds 2 frame
            self.controller.release_button(melee.enums.Button.BUTTON_A)
            self.controller.release_button(melee.enums.Button.BUTTON_B)
            self.controller.release_button(melee.enums.Button.BUTTON_X)
            self.controller.release_button(melee.enums.Button.BUTTON_Z)


        if self.action_frequence != None and self.frameCount % self.action_frequence != 0:
            # If we don't want action every frame then we should return here
            return

        for button, pressed in commands["button"].items():
            if pressed == 1:
                self.controller.press_button(button)
            else:
                self.controller.release_button(button)

        self.controller.press_shoulder(melee.enums.Button.BUTTON_L, commands["l_shoulder"] if commands["l_shoulder"] > 0 else 0)
        self.controller.press_shoulder(melee.enums.Button.BUTTON_R, commands["r_shoulder"] if commands["r_shoulder"] > 0 else 0)

        self.controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, commands["main_stick"][0], commands["main_stick"][1])
        self.controller.tilt_analog_unit(melee.enums.Button.BUTTON_C, commands["c_stick"][0], commands["c_stick"][1])


    def parse_gamestate(self, gamestate):
        frame = self.MeleeFrame(gamestate.frame)

        i = 1
        for player in gamestate.player:

            frame.ports.append(frame.Object())
            frame.ports[i - 1].leader = frame.Object()
            frame.ports[i - 1].leader.pre = frame.Object()
            frame.ports[i - 1].leader.post = frame.Object()

            playerState = gamestate.player[player]
            controllerState = playerState.controller_state

            if self.frameCount == -1:
                self.previousPosition[i - 1] = (playerState.x, playerState.y)
                self.previousFacing[i - 1] = playerState.facing
                self.previousDamage[i - 1] = playerState.percent
                self.previousAction[i - 1] = playerState.action

            frame.ports[i - 1].leader.pre.position = frame.Object()
            frame.ports[i - 1].leader.pre.position.x = self.previousPosition[i - 1][0]
            frame.ports[i - 1].leader.pre.position.y = self.previousPosition[i - 1][1]

            frame.ports[i - 1].leader.pre.joystick = frame.Object()
            frame.ports[i - 1].leader.pre.joystick.x = (controllerState.main_stick[0] - 0.5) * 2
            frame.ports[i - 1].leader.pre.joystick.y = (controllerState.main_stick[1] - 0.5) * 2

            frame.ports[i - 1].leader.pre.cstick = frame.Object()
            frame.ports[i - 1].leader.pre.cstick.x = (controllerState.c_stick[0] - 0.5) * 2
            frame.ports[i - 1].leader.pre.cstick.y = (controllerState.c_stick[1] - 0.5) * 2

            frame.ports[i - 1].leader.pre.triggers = frame.Object()
            frame.ports[i - 1].leader.pre.triggers.physical = frame.Object()
            frame.ports[i - 1].leader.pre.triggers.physical.l = controllerState.l_shoulder
            frame.ports[i - 1].leader.pre.triggers.physical.r = controllerState.r_shoulder

            frame.ports[i - 1].leader.pre.buttons = frame.Object()
            frame.ports[i - 1].leader.pre.buttons.physical = frame.Object()
            frame.ports[i - 1].leader.pre.buttons.physical.value = 0

            for button in controllerState.button:
                if controllerState.button[button]:
                    frame.ports[i - 1].leader.pre.buttons.physical.value += self.button_dict[button.value]

            frame.ports[i - 1].leader.pre.direction = 1 if self.previousFacing[i - 1] else -1
            frame.ports[i - 1].leader.pre.damage = (self.previousDamage[i - 1],)
            frame.ports[i - 1].leader.pre.state = self.previousAction[i - 1].value

            frame.ports[i - 1].leader.post.character = playerState.character.value
            frame.ports[i - 1].leader.post.position = (playerState.x, playerState.y)

            frame.ports[i - 1].leader.post.position = frame.Object()
            frame.ports[i - 1].leader.post.position.x = playerState.x
            frame.ports[i - 1].leader.post.position.y = playerState.y

            frame.ports[i - 1].leader.post.direction = 1 if playerState.facing else -1
            frame.ports[i - 1].leader.post.damage = playerState.percent
            frame.ports[i - 1].leader.post.shield = playerState.shield_strength
            frame.ports[i - 1].leader.post.stocks = playerState.stock
            frame.ports[i - 1].leader.post.hit_stun = playerState.hitstun_frames_left
            frame.ports[i - 1].leader.post.airborne = not playerState.on_ground
            frame.ports[i - 1].leader.post.ground = playerState.on_ground
            frame.ports[i - 1].leader.post.jumps = playerState.jumps_left
            frame.ports[i - 1].leader.post.state_age = playerState.action_frame
            frame.ports[i - 1].leader.post.state = playerState.action.value

            self.previousPosition[i - 1] = (playerState.x, playerState.y)
            self.previousFacing[i - 1] = playerState.facing
            self.previousDamage[i - 1] = playerState.percent
            self.previousAction[i - 1] = playerState.action

            i += 1

        return frame

    _STAGE_CONVERSION = {
        melee.enums.Stage.BATTLEFIELD: 31,
        melee.enums.Stage.DREAMLAND: 28,
        melee.enums.Stage.FINAL_DESTINATION: 32,
        melee.enums.Stage.FOUNTAIN_OF_DREAMS: 2,
        melee.enums.Stage.POKEMON_STADIUM: 3,
        melee.enums.Stage.YOSHIS_STORY: 8,
    }

    def game_loop(self):
        while True:
            gamestate = self.next_state()
            if gamestate is None:  # loop happened before game state changed/posted new frame
                continue

            frame = self.parse_gamestate(gamestate)
            self.input_model_commands(frame, self._STAGE_CONVERSION[gamestate.stage])

            self.frameCount += 1

    def start(self):

        while True:
            gamestate = self.next_state()

            if gamestate is None:  # loop happened before game state changed/posted new frame
                continue

            if gamestate.menu_state == melee.enums.Menu.IN_GAME:
                self.game_loop()

            elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
                melee.menuhelper.MenuHelper.choose_character(melee.enums.Character.CPTFALCON, gamestate, self.controller, swag=True)

            elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
                melee.menuhelper.MenuHelper.choose_stage(melee.enums.Stage.FINAL_DESTINATION, gamestate, self.controller)

            else:
                self.controller.release_all()


if __name__ == "__main__":
    agent = MeleeAI(action_frequence=1, window_size=60, frame_delay=0)
    agent.start()
