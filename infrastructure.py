#!/usr/bin/python3
import sys
import melee

from mvp_model import SSBM_MVP
from infra_adaptor import convert_frame_to_input_tensor, convert_output_tensor_to_command

DOLPHIN_EXE_PATH = '/home/cs488/Desktop/slippi/AppRun' # TODO: modularize to arg

class MeleeAI:

    class MeleeFrame:
        class Object(object):
            pass

        def __init__(self, frameIndex):
            self.ports = []
            self.index = frameIndex


    def __init__(self):
        self.model = SSBM_MVP(100, 50)

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

        self.console = melee.Console(path=DOLPHIN_EXE_PATH)
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

    def next_state(self):
            #self.controller.release_all()  # releases buttons pressed last frame
            return self.console.step()  # get frame data

    def input_model_commands(self, frame):

        convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)

        feature_tensor = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)
        cts_targets, button_targets = self.model(feature_tensor)
        commands = convert_output_tensor_to_command(cts_targets, button_targets)

        for button, pressed in commands["button"].items():
            if pressed == 1:
                self.controller.press_button(button)
            else:
                self.controller.release_button(button)

        self.controller.press_shoulder(melee.enums.Button.BUTTON_L, commands["l_shoulder"])
        self.controller.press_shoulder(melee.enums.Button.BUTTON_R, commands["r_shoulder"])
        self.controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, commands["main_stick"][0], commands["main_stick"][1])
        self.controller.tilt_analog(melee.enums.Button.BUTTON_C, commands["c_stick"][0], commands["c_stick"][1])

    def parse_gamestate(self, gamestate):
        frame = self.MeleeFrame(self.frameCount)

        # pre_frame_dict = []
        # post_frame_dict = []

        for i in gamestate.player:

            frame.ports.append(frame.Object())
            frame.ports[i - 1].leader = frame.Object()
            frame.ports[i - 1].leader.pre = frame.Object()
            frame.ports[i - 1].leader.post = frame.Object()

            playerState = gamestate.player[i]
            controllerState = playerState.controller_state

            if self.frameCount == -1:
                self.previousFacing[i - 1] = playerState.facing
                self.previousDamage[i - 1] = playerState.percent
                self.previousAction[i - 1] = playerState.action

            frame.ports[i - 1].leader.pre.position = frame.Object()
            frame.ports[i - 1].leader.pre.position.x = playerState._prev_x
            frame.ports[i - 1].leader.pre.position.y = playerState._prev_y

            frame.ports[i - 1].leader.pre.joystick = frame.Object()
            frame.ports[i - 1].leader.pre.joystick.x = controllerState.main_stick[0]
            frame.ports[i - 1].leader.pre.joystick.y = controllerState.main_stick[1]

            frame.ports[i - 1].leader.pre.cstick = frame.Object()
            frame.ports[i - 1].leader.pre.cstick.x = controllerState.c_stick[0]
            frame.ports[i - 1].leader.pre.cstick.y = controllerState.c_stick[1]

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
            frame.ports[i - 1].leader.pre.damage = (self.previousDamage[i - 1], )
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

            self.previousFacing[i - 1] = playerState.facing
            self.previousDamage[i - 1] = playerState.percent
            self.previousAction[i - 1] = playerState.action

        return frame

    def game_loop(self):
        while True:
            gamestate = self.next_state()
            if gamestate is None:  # loop happened before game state changed/posted new frame
                continue

            frame = self.parse_gamestate(gamestate)
            self.input_model_commands(frame)

            self.frameCount += 1

    def start(self):
        while True:
            gamestate = self.next_state()
            if gamestate is None:  # loop happened before game state changed/posted new frame
                continue

            if gamestate.menu_state == melee.enums.Menu.IN_GAME:
                self.game_loop()

            elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
                melee.menuhelper.MenuHelper.choose_character(melee.enums.Character.CPTFALCON, gamestate, 2, self.controller, swag=True)
            elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
                melee.menuhelper.MenuHelper.choose_stage(melee.enums.Stage.FINAL_DESTINATION, gamestate, self.controller)

            else:
                self.controller.release_all()

if __name__ == "__main__":
    agent = MeleeAI()
    agent.start()
