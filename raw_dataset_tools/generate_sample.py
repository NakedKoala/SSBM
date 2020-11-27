import argparse
from pathlib import Path
import random
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('list_file', type=str)
    parser.add_argument('count', type=int)

    args = parser.parse_args()

    with open(args.list_file, 'r') as f:
        lines = f.read().strip().split('\n')
        for line in random.sample(lines, args.count):
            data = Path(args.in_dir) / line
            dest = Path(args.out_dir) / line
            shutil.copy(str(data), str(dest))
