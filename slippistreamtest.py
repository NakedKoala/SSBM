import time
import subprocess
import select
import sys
from slippi.parse import parse
from slippi.parse import ParseEvent

def test(frame):
    print(type(frame))

def poll_game_state(filename):
    f = subprocess.Popen(['tail','-c+1','-f',filename],stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    handlers = {ParseEvent.FRAME: test}
    while True:
        parse(f.stdout, handlers)


poll_game_state("/home/cs488/Desktop/Game_20201026T190916.slp")

#handlers = {ParseEvent.FRAME: test}
#while True:
    #parse(sys.stdin.buffer, handlers)

