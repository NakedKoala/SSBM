"""
    Various dataset utilities, including:
        - approximately filtering out non-tournament matches
"""

from pathlib import Path
from slippi import Game

# returns true if the slippi file `filename` is a tournament game:
#   - no teams
#   - 2 players (1 v. 1)
#   - at least 30 seconds long
#   - at least 50 total damage dealt
#   - not PAL
def is_tournament_game(filename, min_dur=30, min_dmg=100):
    game = Game(filename)
    num_players = 0
    for val in game.start.players:
        if val:
            num_players += 1
    duration = game.metadata.duration * (1.0 / 60)  # frames => seconds
    # print(game.start.slippi.version)
    if (
        not game.start.is_teams and
        num_players == 2 and
        duration >= min_dur and
        not game.start.is_pal
    ):
        # check for damage
        tot_dmg = 0.0
        last_dmg = [0] * len(game.start.players)
        for frame in game.frames:
            done = False
            for i, player in enumerate(frame.ports):
                if not player:
                    continue
                dmg = player.leader.post.damage
                tot_dmg += max(0, dmg - last_dmg[i])
                if tot_dmg >= min_dmg:
                    done = True
                    break
                last_dmg[i] = dmg
            if done:
                break
        return tot_dmg >= min_dmg

    return False


def filter_tournament_games(filepath, min_dur=30, min_dmg=100, mod=1, offset=0):
    path = Path(filepath)
    for i, child in enumerate(path.iterdir()):
        if i % mod != offset:
            continue
        try:
            if is_tournament_game(
                str(child.resolve()),
                min_dur=min_dur,
                min_dmg=min_dmg
            ):
                yield str(child.relative_to(path))
        except:
            pass

