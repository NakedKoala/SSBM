import argparse

import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', type=str, default='.')
    args = parser.parse_args()

    game_list = dataset.filter_tournament_games(args.dir)
    print('\n'.join(game_list))
