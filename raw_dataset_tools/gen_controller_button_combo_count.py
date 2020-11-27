import argparse
import multiprocessing
import pickle
import sys

import dataset

def proc_fn(filepath, mod, offset, outname, log_every):
    data = dataset.ProcessControllerState()
    game_list = dataset.process_controller_states(
        filepath, data.update, mod=mod, offset=offset
    )
    for i, filename in enumerate(game_list):
        if log_every > 0:
            if (i+1) % log_every == 0:
                print(i+1, file=sys.stderr)
    with open(outname, 'wb') as f:
        pickle.dump(data.data_by_characters, f)
    # with open(outname, 'w') as f:
    #     f.write(str(data.data_by_characters))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', type=str, default='.')
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
        proc_fn(args.dir, 1, 0, args.output, args.log_every)
    else:
        procs = [None] * args.num_workers
        for i in range(args.num_workers):
            procs[i] = multiprocessing.Process(
                target=proc_fn,
                args=(
                    args.dir,
                    args.num_workers,
                    i,
                    str(i) + '_' + args.output,
                    args.log_every
                ),
                daemon=False
            )
            procs[i].start()

        for i in range(args.num_workers):
            procs[i].join()
