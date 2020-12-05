from ssbm_bot.data.slp_parser import SLPParser
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

SLPParser(src_dir='../dataset/falcon_v_fox_marth_fd_2.0.1_dataset/train', dest_dir='../dataset/falcon_v_fox_marth_fd_2.0.1_dataset_csv/train')()
SLPParser(src_dir='../dataset/falcon_v_fox_marth_fd_2.0.1_dataset/eval', dest_dir='../dataset/falcon_v_fox_marth_fd_2.0.1_dataset_csv/eval')()
