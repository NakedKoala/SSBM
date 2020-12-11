from .environment import BaseEnvironment
from infrastructure import MeleeAI

class LibmeleeEnvironment(BaseEnvironment):
    def __init__(self, frame_delay):
        self.agent = MeleeAI(action_frequence=None, window_size=60, frame_delay=frame_delay, include_opp_input=False, multiAgent=True, weights='../../../weights/mvp_fit5_EP7_VL0349.pth')
        self.agent.start()  # returns after game starts

        self.frame_delay = frame_delay
        self.buffer = []

        self.done = False

    # resets the environment and returns an initial state.
    def reset(self):
        self.agent.shutdown()
        # just make a new console, since menuing post game seems difficult.
        self.agent = MeleeAI(action_frequence=None, window_size=60, frame_delay=self.frame_delay, include_opp_input=False, multiAgent=True, weights='../../../weights/mvp_fit5_EP7_VL0349.pth')

    # executes action immediately and returns delayed state/reward/done.
    def step(self, action):
        self.agent.preform_action(action)

        if not self.done:
            frame, reward, done = self.agent.step()
            self.done = done

            self.buffer.append((frame.ports[0], frame.ports[1], reward, done))

        if len(self.buffer) > 0 and (len(self.buffer) > self.frame_delay or self.done):
            return self.buffer.pop(0)

        return 0, 0, 0 # TODO: what do i return when we dont have enough frames yet?
