from slippi import Game
import pandas as pd
import os
import pdb 

class SLPParser:
    def __init__(self, src_dir, dest_dir):

        self.src_dir, self.dest_dir = src_dir, dest_dir
        self.pre_frame_attributes = ["state", "position", "direction", "joystick", "cstick", "triggers", "buttons", "random_seed", "raw_analog_x", "damage"]
        self.post_frame_attributes = ["character", "state", "state_age", "position", "direction", "damage", "shield", "stocks", "last_attack_landed", \
                                      "last_hit_by", "combo_count", "flags", "hit_stun", "airborne", "ground", "jumps", "l_cancel"]
        self.split_coord_attributes = ["position", "joystick", "cstick", "triggers"]
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
        #    pdb.set_trace()
           att_value = att_value.physical.value 
        if att_name == "triggers":
            att_value = round(att_value.physical.l, 3), round(att_value.physical.r, 3)
        return att_value

    def extract_coord(self, state, att_name):
        att_value = getattr(state, att_name)
        if att_name == "triggers":
            return att_value.physical.l, att_value.physical.r
        else:
            return att_value.x, att_value.y
    def proc_frame(self, dataframe_dict, frame):

        frame_index = frame.index
        dataframe_dict['frame_index'].append(frame_index)
        dataframe_dict['frame_index'].append(frame_index)

        char1_state_pre, char1_state_post = (frame.ports[1].leader.pre, frame.ports[1].leader.post)
        char2_state_pre, char2_state_post = (frame.ports[2].leader.pre, frame.ports[2].leader.post)

        for att_name in self.pre_frame_attributes:
            if att_name in self.split_coord_attributes:
                
                char1_coord = self.extract_coord(char1_state_pre, att_name)
                char2_coord = self.extract_coord(char2_state_pre, att_name)
                dataframe_dict[f'pre_{att_name}_x'].append(char1_coord[0])
                dataframe_dict[f'pre_{att_name}_x'].append(char2_coord[0])
                dataframe_dict[f'pre_{att_name}_y'].append(char1_coord[1])
                dataframe_dict[f'pre_{att_name}_y'].append(char2_coord[1])
            else:
                dataframe_dict[f'pre_{att_name}'].append(self.proc_attr_value(char1_state_pre, att_name))
                dataframe_dict[f'pre_{att_name}'].append(self.proc_attr_value(char2_state_pre, att_name))

        for att_name in self.post_frame_attributes:
            if att_name in self.split_coord_attributes:
                char1_coord = self.extract_coord(char1_state_post, att_name)
                char2_coord = self.extract_coord(char2_state_post, att_name)
                dataframe_dict[f'post_{att_name}_x'].append(char1_coord[0])
                dataframe_dict[f'post_{att_name}_x'].append(char2_coord[0])
                dataframe_dict[f'post_{att_name}_y'].append(char1_coord[1])
                dataframe_dict[f'post_{att_name}_y'].append(char2_coord[1])
            else:
                dataframe_dict[f'post_{att_name}'].append(self.proc_attr_value(char1_state_post, att_name))
                dataframe_dict[f'post_{att_name}'].append(self.proc_attr_value(char2_state_post, att_name))
    
    def data_pre_proc_mvp(self, df):
       
       df['post_ground'].fillna(value = 0.0, inplace=True)
       df['post_last_hit_by'].fillna(value = 0.0, inplace=True)
       df['post_last_attack_landed'].fillna(value = 0.0, inplace=True)
       df['pre_damage'] = df['pre_damage'].apply(lambda x: x[0])
       df['post_airborne'] = df['post_airborne'].astype(int)
       df['pre_state'] = df['pre_state'].astype(int)
       df['post_state'] = df['post_state'].astype(int)
       pdb.set_trace()
      
      
      
       categorical_cols = ['pre_direction', 'post_direction', 'post_ground', 'post_jumps', 'post_stocks']
       one_hot_features = [ pd.get_dummies(df[col], prefix=col) for col in categorical_cols]

       df = pd.concat([df] + one_hot_features, axis=1)
     
       for col in categorical_cols:
           df.drop(col, axis=1, inplace=True)
       return df
       

       
    def compute_df_cols(self):
        pre_frame_cols = [ f'pre_{col}' for col in self.pre_frame_attributes if col not in self.split_coord_attributes]
        post_frame_cols = [ f'post_{col}' for col in self.post_frame_attributes if col not in self.split_coord_attributes]
        split_coord_cols = []
        for col in self.pre_frame_attributes:
            if col in self.split_coord_attributes:
                split_coord_cols.append(f'pre_{col}_x')
                split_coord_cols.append(f'pre_{col}_y')
        for col in self.post_frame_attributes:
            if col in self.split_coord_attributes:
                split_coord_cols.append(f'post_{col}_x')
                split_coord_cols.append(f'post_{col}_y')
        return ["frame_index"] + pre_frame_cols + post_frame_cols + split_coord_cols
        
    
    def __call__(self, slp_files):

        for src_fname in slp_files:
            self.slp_object = Game(os.path.join(self.src_dir, src_fname))

            dest_fname = self.format_filename()
            
            dataframe_dict = {col: []  for col in self.compute_df_cols()}

            for frame in self.slp_object.frames:
                self.proc_frame(dataframe_dict, frame)

            df = pd.DataFrame.from_dict(dataframe_dict)
            df = self.data_pre_proc_mvp(df)
            # pdb.set_trace()
            df.to_csv(os.path.join(self.dest_dir, dest_fname), index=False)
            
