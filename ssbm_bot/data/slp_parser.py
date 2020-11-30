from slippi import Game
import pandas as pd
import os
import pdb
from tqdm import tqdm
import traceback
import concurrent.futures

from .common_parsing_logic import *

'''
Remove set:
{
"random_seed"
"raw_analog_x"
"l_cancel"
"combo_count"
"flags"
"last_hit_by"
"last_attack_landed"
}
7
'''
class SLPParser:
    pre_frame_attributes = ["state", "position", "direction", "joystick", "cstick", "triggers", "buttons", "damage"]
    post_frame_attributes = ["character", "state", "state_age", "position", "direction", "damage", "shield", "stocks",\
                             "hit_stun", "airborne", "ground", "jumps"]
    split_coord_attributes = ["position", "joystick", "cstick", "triggers"]
    def __init__(self, src_dir, dest_dir):

        self.src_dir, self.dest_dir = src_dir, dest_dir


    def format_filename(self):

        timestamp = int(self.slp_object.metadata.date.timestamp())

        characters = []
        for player in self.slp_object.start.players:
            if player == None:
                continue
            characters.append(player.character.name)

        stage = self.slp_object.start.stage.name
        return f'{timestamp}_{"_".join(characters)}_{stage.replace("_", "")}'

    def proc_frames(self, frames, dest_fname):

        dataframe_dict = initialize_dataframe_dict(SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)

        for frame in frames:
            proc_frame(dataframe_dict, frame, SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)

        df = pd.DataFrame.from_dict(dataframe_dict)

        df = data_pre_proc_mvp(df)

        df.to_csv(os.path.join(self.dest_dir, dest_fname + ".csv"), index=False)

    def proc_file(self, src_fname,rename_only=False):
        try:
                self.slp_object = Game(os.path.join(self.src_dir, src_fname))

                dest_fname = self.format_filename()

                if rename_only == True:

                        os.rename(os.path.join(self.src_dir, src_fname), os.path.join(self.src_dir, dest_fname + '.slp'))

                else:
                    dataframe_dict = initialize_dataframe_dict(SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)

                    for frame in self.slp_object.frames:
                        proc_frame(dataframe_dict, frame, SLPParser.pre_frame_attributes, SLPParser.post_frame_attributes, SLPParser.split_coord_attributes)

                    df = pd.DataFrame.from_dict(dataframe_dict)

                    # assert(df.shape[1] == 33)

                    df = data_pre_proc_mvp(df)

                    df.to_csv(os.path.join(self.dest_dir, dest_fname + ".csv"), index=False)
        except:
            print(f'Failed to parse {src_fname}')
            traceback.print_exc()

    def __call__(self):
        slp_files = [fname for fname in os.listdir(self.src_dir) if ".slp" in fname]

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            o = list(tqdm(executor.map(self.proc_file, slp_files), total=len(slp_files)))
