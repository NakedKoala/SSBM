import argparse
import warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('num_files', type=int)

    args = parser.parse_args()

    outputs = set()
    for i in range(args.num_files):
        filename = str(i) + '_' + args.output
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                if line in outputs:
                    warnings.warn('duplicate entry found: ' + line)
                outputs.add(line)
    for line in sorted(outputs):
        print(line)
