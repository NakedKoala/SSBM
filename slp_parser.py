from slippi import Game
import pandas as pd
import os
import pdb 

class SLPParse:
    def __init__(self, src_dir, dest_dir):

        self.src_dir, self.dest_dir = src_dir, dest_dir
        self.pre_frame_attributes = ["state", "position", "direction", "joystick", "cstick", "triggers", "buttons", "random_seed", "raw_analog_x", "damage"]
        self.post_frame_attributes = ["character", "state", "state_age", "position", "direction", "damage", "shield", "stocks", "last_attack_landed", \
                                      "last_hit_by", "combo_count", "flags", "hit_stun", "airborne", "ground", "jumps", "l_cancel"]
    def format_filename(self):

        timestamp = int(self.slp_object.metadata.date.timestamp())

        characters = []
        for player in self.slp_object.metadata.players:
            if player == None:
                continue 
            characters.append(list(player.characters.keys())[0].name)
        assert(len(characters) == 2)
        char1, char2 = characters
        stage = self.slp_object.start.stage.name
        return f'{timestamp}_{char1.replace("_","")}_{char2.replace("_", "")}_{stage.replace("_", "")}.csv'

    def proc_attr_value(self, state, att_name):

        att_value = getattr(state, att_name)
        if att_name == "flags":
           att_value = att_value.__repr__()
        if att_name == "buttons":
           att_value = att_value.logical.__repr__()
        if att_name == "triggers":
            att_value = round(att_value.physical.l, 3), round(att_value.physical.r, 3)
        return att_value

    def proc_frame(self, dataframe_dict, frame):

        frame_index = frame.index
        dataframe_dict['frame_index'].append(frame_index)
        dataframe_dict['frame_index'].append(frame_index)

        char1_state_pre, char1_state_post = (frame.ports[1].leader.pre, frame.ports[1].leader.post)
        char2_state_pre, char2_state_post = (frame.ports[2].leader.pre, frame.ports[2].leader.post)

        for att_name in self.pre_frame_attributes:
            dataframe_dict[f'pre_{att_name}'].append(self.proc_attr_value(char1_state_pre, att_name))
            dataframe_dict[f'pre_{att_name}'].append(self.proc_attr_value(char2_state_pre, att_name))

        for att_name in self.post_frame_attributes:
            dataframe_dict[f'post_{att_name}'].append(self.proc_attr_value(char1_state_post, att_name))
            dataframe_dict[f'post_{att_name}'].append(self.proc_attr_value(char2_state_post, att_name))

    def __call__(self, slp_files):

        for src_fname in slp_files:
            self.slp_object = Game(os.path.join(self.src_dir, src_fname))

            dest_fname = self.format_filename()

            dataframe_dict = {"frame_index": []}

            for item in self.pre_frame_attributes:
                dataframe_dict[f'pre_{item}'] = []
            
            for item in self.post_frame_attributes:
                dataframe_dict[f'post_{item}'] = []

            for frame in self.slp_object.frames:
                self.proc_frame(dataframe_dict, frame)

            df = pd.DataFrame.from_dict(dataframe_dict)
            df.to_csv(os.path.join(self.dest_dir, dest_fname), index=False)
            
