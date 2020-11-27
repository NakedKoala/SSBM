# converts indices to/from trigger output
NUM_INDICES = 128  # [0, 127] magnitude

# trigger amount from 0 to 1
def to_index(trigger):
    return round(trigger * (NUM_INDICES-1))

def to_trigger(idx):
    return 1.0 / (NUM_INDICES-1) * idx

ACCURACY_EPS = 1e-1
def close_fine(t1, t2):
    return abs(t1 - t2) <= ACCURACY_EPS

def close_coarse(t1, t2):
    # within same quarter line?
    sector1 = round(t1 / (1.0 / 4))
    sector2 = round(t2 / (1.0 / 4))
    return sector1 == sector2

if __name__ == '__main__':
    assert(NUM_INDICES == 128)

    assert(to_index(0) == 0)
    assert(to_index(1) == 127)
    assert(to_index(0.5) == 64)

    for index in range(0, 128):
        assert(to_index(to_trigger(index)) == index)

    assert(close_coarse(1.0, 1.0))
    assert(close_coarse(1.0, 0.9))
    assert(not close_coarse(1.0, 0.75))
    assert(close_coarse(0.7, 0.8))
    assert(close_coarse(0.6, 0.5))
    assert(close_coarse(0.1, 0.0))
    assert(not close_coarse(0.1, 0.2))
