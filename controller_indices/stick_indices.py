# converts indices to/from stick output
import math

# These 3 should multiply out to something significantly larger than 256*256
# so that every point in the 128 radius circle in 256*256 can be represented
# by these 3 numbers.
COARSE_N = 129 # 128 angles, + 1 for the center
PRECISE_N = 32 # [0, 16] magnitude
# PRECISE_N = 16 misses just a few possible control stick positions.
assert(PRECISE_N % 2 == 0)
MAGN_N = 128 # [1, 128] magnitude

# stick x, y from [-1, 1]
# returns coarse index, angle index, magnitude index
def to_index(x, y):
    magnitude = min(math.hypot(x, y), 1.0)
    magn_idx = round(magnitude * MAGN_N)
    if magn_idx == 0:
        return 0, 0, 0

    precise_width = 2 * math.pi / (COARSE_N-1) / PRECISE_N
    angle = math.atan2(y, x)
    if angle < 0:
        angle += math.pi * 2
    precise_sector = round(angle / precise_width)

    coarse_idx = math.floor((precise_sector + PRECISE_N/2) / PRECISE_N)
    precise_idx = precise_sector - coarse_idx * PRECISE_N + PRECISE_N//2
    if coarse_idx == COARSE_N - 1:
        coarse_idx = 0

    return 1+coarse_idx, precise_idx, magn_idx-1

# returns stick x, y
def to_stick(coarse_idx, precise_idx, magn_idx):
    if coarse_idx == 0:
        return 0, 0
    coarse_idx -= 1

    sector_width = 2 * math.pi / (COARSE_N - 1)
    coarse_angle = coarse_idx * sector_width
    angle = coarse_angle + (precise_idx - PRECISE_N/2) * (sector_width / PRECISE_N)

    magn = (magn_idx+1) / MAGN_N

    return magn * math.cos(angle), magn * math.sin(angle)

ACCURACY_EPS = 1e-1
def close_fine(x1, y1, x2, y2):
    return math.hypot(x1-x2, y1-y2) <= ACCURACY_EPS

DEAD_ZONE = 0.2
def close_coarse(x1, y1, x2, y2):
    # dead zone
    magn1 = math.hypot(x1, y1)
    magn2 = math.hypot(x2, y2)
    if magn1 <= DEAD_ZONE and magn2 <= DEAD_ZONE:
        return True
    elif magn1 <= DEAD_ZONE or magn2 <= DEAD_ZONE:
        return False

    # both pointing in the same 45 deg angle?
    ang1 = math.atan2(y1, x1)
    ang2 = math.atan2(y2, x2)

    if ang1 < 0:
        ang1 += 2 * math.pi
    if ang2 < 0:
        ang2 += 2 * math.pi

    sector1 = round(ang1 / (math.pi / 4))
    sector2 = round(ang2 / (math.pi / 4))

    if sector1 == 8:
        sector1 = 0
    if sector2 == 8:
        sector2 = 0

    return sector1 == sector2

if __name__ == '__main__':
    # test stick granularity is consistent (forces you to change tests)
    assert(COARSE_N == 129)
    assert(PRECISE_N == 32)
    assert(MAGN_N == 128)

    # center
    assert(to_index(0, 0) == (0, 0, 0))
    # x/y-axes
    assert(to_index(1, 0) == (1, 16, 127))
    assert(to_index(-1, 0) == (65, 16, 127))
    assert(to_index(0, 1) == (33, 16, 127))
    assert(to_index(0, -1) == (97, 16, 127))
    # beyond the unit circle
    assert(to_index(1, 1) == (17, 16, 127))

    # inverse check - every possible index can be recovered by applying to_index after to_stick
    for coarse in range(129):
        for precise in range(32):
            for magn in range(128):
                x, y = to_stick(coarse, precise, magn)
                if coarse == 0:
                    assert(to_index(x, y) == (0, 0, 0))
                else:
                    n_coarse, n_precise, n_magn = to_index(x, y)
                    assert((n_coarse, n_precise, n_magn) == (coarse, precise, magn)), \
                        "%d %d %d %d %d %d" % (coarse, precise, magn, n_coarse, n_precise, n_magn)

    # completeness check - check that every control stick position in [-128, 127] x [-128, 127] in the 127.5 radius circle
    # around (-0.5, -0.5) maps to a different index
    stick_idx_set = set()
    valid_stick_pos_cnt = 0
    for x in range(-128, 128):
        for y in range(-128, 128):
            norm_x = (x-0.5)/127.5
            norm_y = (y-0.5)/127.5
            if math.hypot(norm_x, norm_y) <= 1:
                valid_stick_pos_cnt += 1
                stick_idx_set.add(to_index(norm_x, norm_y))
    assert(len(stick_idx_set) == valid_stick_pos_cnt), "%d %d" % (len(stick_idx_set), valid_stick_pos_cnt)

    assert(close_coarse(0.8, 0.05, 0.8, -0.05))
    assert(close_coarse(
        math.cos(math.radians(22)), math.sin(math.radians(22)),
        1.0, 0
    ))
    assert(not close_coarse(
        math.cos(math.radians(23)), math.sin(math.radians(23)),
        1.0, 0
    ))
