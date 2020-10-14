import argparse
import sys

import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', type=str, default='.')
    parser.add_argument('--min_dur', type=float, default=30)
    parser.add_argument('--min_dmg', type=float, default=100)
    parser.add_argument('--log_every', type=int, default=0)
    args = parser.parse_args()

    game_list = dataset.filter_tournament_games(
        args.dir, min_dur=args.min_dur, min_dmg=args.min_dmg
    )
    for i, game in enumerate(game_list):
        if args.log_every > 0:
            if (i+1) % args.log_every == 0:
                print(i+1, file=sys.stderr)
        print(game)
