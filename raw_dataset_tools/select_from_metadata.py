import argparse
import sys

def str_to_version(ver_str):
    return list(map(int, ver_str.split('.')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_in', type=str, nargs='?')
    parser.add_argument('output', type=str, nargs='?')
    parser.add_argument('--min_dur', type=float, default=30)
    parser.add_argument('--min_version', type=str)
    parser.add_argument('--char1', type=str, nargs='*')
    parser.add_argument('--char2', type=str, nargs='*')
    parser.add_argument('--stage', type=str, nargs='*')
    args = parser.parse_args()

    infile = sys.stdin
    if args.metadata_in is not None:
        infile = open(args.metadata_in, 'r')

    outfile = sys.stdout
    if args.output is not None:
        outfile = open(args.output, 'w')

    min_version = str_to_version(args.min_version) if args.min_version else None

    try:
        lines = infile.read().strip().split('\n')
        for line in lines:
            entries = line.split('|||')
            name = entries[0]
            entries = entries[1].split(',')
            metadata = {}
            for entry in entries:
                tokens = entry.split(':')
                metadata[tokens[0]] = tokens[1]

            # version
            if min_version:
                if str_to_version(metadata['version']) < min_version:
                    continue

            # stage
            if args.stage and metadata['stage'] not in args.stage:
                continue

            # duration
            if args.min_dur and float(metadata['duration']) < args.min_dur:
                continue

            # characters
            chars = metadata['characters'].split(';')
            if args.char1 and args.char2:
                if not(
                    (chars[0] in args.char1 and chars[1] in args.char2) or
                    (chars[0] in args.char2 and chars[1] in args.char1)
                ):
                    continue
            elif args.char1 or args.char2:
                required = args.char1 or args.char2
                if not(
                    chars[0] in required or chars[1] in required
                ):
                    continue

            outfile.write(name + '\n')
    finally:
        infile.close()
        outfile.close()
