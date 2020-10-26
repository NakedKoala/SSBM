#!/usr/bin/python3
import sys
import melee
from slippi.parse import parse
from slippi.parse import ParseEvent
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import random


DOLPHIN_EXE_PATH = '/home/cs488/Desktop/slippi/AppRun' # TODO: modularize to arg


class MeleeAI:
    def __init__(self):
        self.slippi_replay_file = ""
        self.looking_for_file = False

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


    def wait_for_slippi_file(self):
        slippi_path = "/home/cs488/Slippi"  # TODO: modularize to a cli arg
        observer = Observer()

        class SlippiFileDetectionHandler(FileSystemEventHandler):
            def __init__(self, agent):
                self.agent = agent

            def on_created(self, event):
                # TODO: regex to make sure its actually a slippi file
                self.agent.slippi_replay_file = event.src_path
                print(event.src_path)
                observer.stop()

        event_handler = SlippiFileDetectionHandler(self)
        observer.schedule(event_handler, path=slippi_path, recursive=False)
        observer.start()
        self.looking_for_file = True

        # list_of_files = glob.glob('/home/cs488/Slippi/*')  # * means all if need specific format then *.csv
        # latest_file = max(list_of_files, key=os.path.getctime)
        # self.slippi_replay_file = latest_file

    def next_state(self):
            #self.controller.release_all()  # releases buttons pressed last frame
            return self.console.step()  # get frame data

    def featurize_frame(self, frame):
        # TODO: what do we need to featurize?
        return frame

    def get_model_command(self, frame):
        # TODO: send frame to model
        # TODO: convert model output to command
        return frame

    def parse_frame(self, frame):
        print(type(frame))
        featurized_frame = self.featurize_frame(frame)
        command = self.get_model_command(featurized_frame)
        #print(type(command))
        # self.controller.press_button(command)

    def game_loop(self):
        while True:
            gamestate = self.next_state()
            if gamestate is None:  # loop happened before game state changed/posted new frame
                continue
            self.controller.release_all()

            action = random.randint(0, 4)
            if(action == 0):
                self.controller.press_button(melee.enums.Button.BUTTON_A)
            elif(action == 1):
                self.controller.press_button(melee.enums.Button.BUTTON_B)
            elif(action == 2):
                self.controller.press_button(melee.enums.Button.BUTTON_D_UP)
            else:
                self.controller.press_button(melee.enums.Button.BUTTON_D_DOWN)

        # f = subprocess.Popen(['tail', '-c+1', '-f', self.slippi_replay_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # handlers = {ParseEvent.FRAME: self.parse_frame}
        # while True:
        #     parse(f.stdout, handlers)

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
                if not self.looking_for_file:
                    self.wait_for_slippi_file()
                melee.menuhelper.MenuHelper.choose_stage(melee.enums.Stage.FINAL_DESTINATION, gamestate, self.controller)

            else:
                self.controller.release_all()

if __name__ == "__main__":
    agent = MeleeAI()
    agent.start()
