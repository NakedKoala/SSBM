from ssbm_bot.data.slp_parser import SLPParser
from pathlib import Path
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise RuntimeError(
            "requires 2 arguments: <source slp directory> <target csv directory>"
        )
    src_path = Path(sys.argv[1])
    dest_path = Path(sys.argv[2])
    SLPParser(src_dir=str(src_path / 'train'), dest_dir=str(dest_path / 'train'))()
    SLPParser(src_dir=str(src_path / 'eval'), dest_dir=str(dest_path / 'eval'))()
