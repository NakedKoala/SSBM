import argparse
import multiprocessing
import sys

import dataset

def proc_fn(filepath, min_dur, min_dmg, mod, offset, outname, log_every):
    game_list = dataset.filter_tournament_games(
        filepath, min_dur=min_dur, min_dmg=min_dmg,
        mod=mod, offset=offset
    )
    with open(outname, 'w') as f:
        for i, game in enumerate(game_list):
            if log_every > 0:
                if (i+1) % log_every == 0:
                    print(i+1, file=sys.stderr)
            f.write(game + '\n')
            f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', type=str, default='.')
    parser.add_argument('--min_dur', type=float, default=30)
    parser.add_argument('--min_dmg', type=float, default=100)
    parser.add_argument('--log_every', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('output', nargs='?', type=str)
    args = parser.parse_args()

    if args.output is None:
        if args.num_workers == 1:
            args.output = '/dev/stdout'
        else:
            raise RuntimeError('`output` required if `num_workers` >= 2')

    if args.num_workers == 1:
        proc_fn(args.dir, args.min_dur, args.min_dmg, 1, 0, args.output, args.log_every)
    else:
        procs = [None] * args.num_workers
        for i in range(args.num_workers):
            procs[i] = multiprocessing.Process(
                target=proc_fn,
                args=(
                    args.dir,
                    args.min_dur,
                    args.min_dmg,
                    args.num_workers,
                    i,
                    str(i) + '_' + args.output,
                    args.log_every
                ),
                daemon=True
            )
            procs[i].start()

        for i in range(args.num_workers):
            procs[i].join()
