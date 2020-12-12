# converts indices to/from button output
import numpy as np

NUM_INDICES = 2 ** 5

# order is arbitrary, as long as input/output is consistent.
# buttons is list of numbers: non-zero means corresponding button is pressed.
def to_index(buttons):
    idx = 0
    for i, button in enumerate(reversed(buttons)):
        if button != 0:
            idx += 2 ** i
    return idx

def to_buttons(idx):
    buttons = [0] * 5
    for i in reversed(range(5)):
        if idx % 2 == 1:
            buttons[i] = 1
        idx //= 2
    return buttons

if __name__ == '__main__':
    assert(NUM_INDICES == 32)

    button_answers = [
        ([0, 0, 0, 0, 0], 0),
        ([0, 0, 0, 0, 1], 1),
        ([0, 0, 0, 1, 0], 2),
        ([0, 0, 1, 0, 0], 4),
        ([0, 1, 0, 0, 0], 8),
        ([1, 0, 0, 0, 0], 16),
        ([1, 1, 1, 1, 1], 31),
    ]

    for buttons, idx in button_answers:
        assert(to_index(buttons) == idx)
        assert(to_buttons(idx) == buttons)
