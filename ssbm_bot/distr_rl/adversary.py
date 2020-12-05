# adversary agent for runner.py
# no running logic involved - simply listens for
# messages from the runner to either reset the model
# or to generate output.

from .a3c import A3CTrainer
from .communication import *
from .payloads import AdversaryParamPayload, AdversaryInputPayload

from ..data.infra_adaptor import FrameContext

from collections import deque
import sys
import torch

def adversary_loop(
    runner_port,
    window_size,
    frame_delay,
):
    runner_socket = PairSocket(None, runner_port)

    # first input is model
    model = runner_socket.recv()
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Adversary received non-torch.nn.Module as first message!")
    adversary = A3CTrainer(model)

    frame_ctx = FrameContext(window_size, frame_delay)
    last_action = None

    while True:
        payload = runner_socket.recv()
        if isinstance(payload, AdversaryParamPayload):
            adversary.model.load_state_dict(payload.state_dict)

            # assume new game started.
            frame_ctx = FrameContext(window_size, frame_delay)
            last_action = None
        elif isinstance(payload, AdversaryInputPayload):
            # payload is inputs - send output back.
            state, behavior = payload

            # get full state input
            state_t, action_t = frame_ctx.push_tensor(state, None if last_action is None else last_action[0])
            last_action = adversary.choose_action((state_t, action_t), behavior)

            runner_socket.send(last_action)
        else:
            sys.stderr.write("Error when receiving runner payload!\n")
