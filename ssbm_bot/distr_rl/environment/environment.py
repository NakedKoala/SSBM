class BaseEnvironment(object):
    def __init__(self, frame_delay):
        raise NotImplementedError()

    # resets the environment and returns an initial state.
    def reset(self):
        raise NotImplementedError()

    # executes action immediately and returns delayed state/reward/done.
    def step(self, action):
        raise NotImplementedError()

