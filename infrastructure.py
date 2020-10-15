#!/usr/bin/python3
import sys
import melee
from slippi.parse import parse
from slippi.parse import ParseEvent
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DOLPHIN_EXE_PATH = '/home/cs488/Desktop/slippi/AppRun' # TODO: modularize to arg


class MeleeAI:
    def __init__(self):
        self.slippi_replay_file = ""

        self.console = melee.Console(path=DOLPHIN_EXE_PATH,
                                    slippi_address="127.0.0.1",
                                    slippi_port=51441,
                                    blocking_input=False,
                                    polling_mode=False,
                                    logger=None)
        self.console.render = True
        self.controller = melee.Controller(console=self.console, port=2, type=melee.ControllerType.STANDARD)
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

        class SlippiFileDetectionHandler(FileSystemEventHandler):
            def __init__(self, agent):
                self.agent = agent

            def on_created(self, event):
                # TODO: regex to make sure its actually a slippi file
                self.agent.slippi_replay_file = event.src_path

        slippi_path = "/home/cs488/Slippi"  # TODO: modularize to a cli arg
        event_handler = SlippiFileDetectionHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path=slippi_path, recursive=False)
        observer.start()
        while True:
            if self.slippi_replay_file != "":
                break
            time.sleep(1)

        observer.stop()
        observer.join()

    def next_state(self):
            self.controller.release_all()  # releases buttons pressed last frame
            return self.console.step()  # get frame data

    def featurize_frame(self, frame):
        # TODO: what do we need to featurize?
        return frame

    def get_model_command(self, frame):
        # TODO: send frame to model
        # TODO: convert model output to command
        return frame

    def parse_frame(self, frame):
        featurized_frame = self.featurize_frame(frame)
        command = self.get_model_command(featurized_frame)
        print(type(command))
        # self.controller.press_button(command)

    def game_loop(self):
        # this might need to be done sooner, since it could create the file before libmelee realizes we are in game
        self.wait_for_slippi_file()
        f = subprocess.Popen(['tail', '-c+1', '-f', self.slippi_replay_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        handlers = {ParseEvent.FRAME: self.parse_frame}
        while True:
            parse(f.stdout, handlers)

    def start(self):
        while True:
            gamestate = self.next_state()
            if gamestate is None:  # loop happened before game state changed/posted new frame
                continue

            if gamestate.menu_state == melee.enums.Menu.IN_GAME:
                self.game_loop()

            # TODO: get into game somehow but for now do nothing
            # else:
                # melee.menuhelper.MenuHelper.menu_helper_simple(gameState,
                #                                               controller,
                #                                               2,
                #                                               melee.enums.Character.FOX,
                #                                               melee.enums.Stage.FINAL_DESTINATION,
                #                                               "YUNA#917",
                #                                               autostart=True,
                #                                               swag=True)

if __name__ == "__main__":
    agent = MeleeAI()
    agent.start()
