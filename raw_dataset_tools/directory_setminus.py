from pathlib import Path
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise RuntimeError(
            "requires 2 arguments: <directory to perform setminus on> <directory containing the set to subtract>"
        )
    print("starting")
    target_dir = Path(sys.argv[1])
    minus_dir = Path(sys.argv[2])
    count = 0
    for child in minus_dir.iterdir():
        other = target_dir / child.name
        if other.exists():
            other.unlink()
            count += 1
    print("done")
    print("removed", count, "files.")
