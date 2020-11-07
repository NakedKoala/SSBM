

import pandas as pd 
import numpy as np 
import torch
from sklearn.preprocessing import StandardScaler

def scale(tensor):
    mean = torch.tensor([1.41423043e+00,  7.85902318e+00,  1.41610300e+00,  7.80339904e+00,
        5.54843011e+01,  1.20621781e+01,  8.08904896e-01,  5.88113544e+01,
        1.83485191e+01,  5.59438437e+00,  6.52009154e-01,  5.29972526e-01,
        5.54842529e+01,  5.03226905e-01,  4.96555996e-01,  5.03235357e-01,
        4.96547543e-01,  2.71878764e+00,  1.67740513e+00,  3.57006903e+00,
        1.67848391e+00,  3.52045915e+00,  4.57292101e+01,  1.09459958e+01,
        9.26322530e-01,  5.92744284e+01,  1.58369980e+01,  6.77617745e+00,
        4.64083937e-01,  4.64926234e-01,  4.57292101e+01,  5.07767453e-01,
        4.92015448e-01,  5.07770270e-01,  4.92012630e-01,  2.71254100e+00,
        4.33544745e-03, -5.96091991e-02, -1.71527698e-03, -1.15610866e-02,
        1.12961590e-01,  8.38983897e-02, -9.18879334e-04, -3.93047363e-02,
       -2.32658232e-03,  2.02196862e-03,  8.83636491e-02,  7.81077922e-02])
    std = torch.tensor([ 63.15214909, 31.85448842, 63.20340685, 31.88907913, 45.4699356 ,
       12.17418325,  0.47170615,  3.79583704, 17.29841457, 17.53603757,
        1.09023949,  0.49910085, 45.46992937,  0.49998957,  0.49998814,
        0.49998951,  0.4999881 ,  1.07013528, 67.62576071, 25.21796792,
       67.71633154, 25.23324746, 38.06941933, 11.66731482,  0.85936804,
        2.81122129, 12.55309953, 23.5396452 ,  0.96226853,  0.49876832,
       38.06941933,  0.49993966,  0.49993625,  0.49993964,  0.4999362 ,
        1.07326573,  0.65573621,  0.37714628,  0.16054365,  0.15059148,
        0.28983791,  0.25826303,  0.60554965,  0.44761662,  0.11764879,
        0.13110503,  0.2549518 ,  0.24309364])
    return (tensor - mean) / std
      
       

def proc_button_press(buttons_values_np, button_press_indicator_dim):

        def convert_button_value_to_indicator_vector(value):
           
            tmp_val = value
            bits = []

            while value > 0:
                bits.append(value % 2)
                value //= 2 
            bits = list(reversed(bits)) 


            if len(bits) > 12 :
                bits = bits[1:]

            padded_len = button_press_indicator_dim - len(bits)

            if padded_len > 0:
                bits = [0] * padded_len + bits
           
            assert(len(bits) == 12 )
          
           
            selected_bits = bits[0:4] + bits[5:8]
            assert( len(selected_bits) == 7)
            #  Keep valid keys (most significant bit --> least significant bit)  [start, y, x, b, a, none, l, r, z, d_up, d_down, d_right, d_left]
            return np.array(selected_bits)
           
     
        f = np.vectorize(convert_button_value_to_indicator_vector, otypes=[np.ndarray])
        # try:
        bin_cls_targets_np = np.stack(f(buttons_values_np), axis=0) 
        return bin_cls_targets_np

def proc_df(df, char_id, opponent_id, frame_delay, button_press_indicator_dim):
        
       
        feat_cols = ['pre_state',  'post_state','pre_position_x', 'pre_position_y',  'post_position_x', 'post_position_y', \
                'post_damage', 'post_state_age', 'post_combo_count', 'post_shield', 'post_last_attack_landed', \
                'post_hit_stun', 'post_last_hit_by', 'post_airborne', 'pre_damage',  'post_direction_1', 'post_direction_-1', \
                'pre_direction_1', 'pre_direction_-1', 'post_stocks']
        target_cols = ['pre_joystick_x', 'pre_joystick_y',  'pre_cstick_x', 'pre_cstick_y', \
                       'pre_triggers_x', 'pre_triggers_y', 'pre_buttons']
        
        df_char = df[df['post_character'] == char_id]
        df_opp = df[df['post_character'] == opponent_id]
       
        char_features_df, char_cmd_df, char_targets_df = df_char[feat_cols].shift(frame_delay).fillna(0), \
                                                         df_char[target_cols].shift(frame_delay).fillna(0), \
                                                         df_char[target_cols]

        opp_features_df, opp_cmd_df = df_opp[feat_cols].shift(frame_delay).fillna(0),\
                                      df_opp[target_cols].shift(frame_delay).fillna(0)

        # import pdb 
        # pdb.set_trace()
        features_np = np.concatenate([char_features_df.to_numpy()[:,0:2], opp_features_df.to_numpy()[:,0:2],\
                                      char_features_df.to_numpy()[:,2:], opp_features_df.to_numpy()[:,2:]], axis=1)
        # embedding_feature  reg_features  opp_target  

        
        char_target_button_values_np = char_targets_df['pre_buttons'].to_numpy().reshape(-1)
        char_cmd_button_values_np = char_cmd_df['pre_buttons'].to_numpy().reshape(-1)
        opp_cmd_button_values_np = opp_cmd_df['pre_buttons'].to_numpy().reshape(-1)
        char_targets_df.drop('pre_buttons', axis=1, inplace=True)
        char_cmd_df.drop('pre_buttons', axis=1, inplace=True)
        opp_cmd_df.drop('pre_buttons', axis=1, inplace=True)


        # import pdb 
        # pdb.set_trace()
        char_target_bin_cls_targets_np = proc_button_press(char_target_button_values_np, button_press_indicator_dim)
        char_cmd_bin_cls_targets_np = proc_button_press(char_cmd_button_values_np, button_press_indicator_dim)
        opp_cmd_bin_cls_targets_np = proc_button_press(opp_cmd_button_values_np, button_press_indicator_dim)

        features_np = np.concatenate([features_np, char_cmd_df.to_numpy(), opp_cmd_df.to_numpy()], axis=1)
       
        # scaler = StandardScaler()
        # features_np[:,4:] = scaler.fit_transform(features_np[:,4:])
        # 4 -> 52 
        # import pdb 
        # pdb.set_trace()

        features_np = np.concatenate([features_np, char_cmd_bin_cls_targets_np, opp_cmd_bin_cls_targets_np], axis=1)

        cts_targets_np = char_targets_df.to_numpy()


        features_tensor, cts_targets_tensor, char_bin_class_targets_tensor = torch.from_numpy(features_np), torch.from_numpy(cts_targets_np), torch.from_numpy(char_target_bin_cls_targets_np)
        
        assert(features_tensor.shape[1] == 20 * 2 + 13 * 2)
        assert(cts_targets_tensor.shape[1] == 6)
        assert(char_bin_class_targets_tensor.shape[1] == 7)

        return features_tensor.float(), cts_targets_tensor.float(),  char_bin_class_targets_tensor.float()
        

def compute_df_cols(pre_frame_attributes, post_frame_attributes, split_coord_attributes):

    pre_frame_cols = [ f'pre_{col}' for col in pre_frame_attributes if col not in split_coord_attributes]
    post_frame_cols = [ f'post_{col}' for col in post_frame_attributes if col not in split_coord_attributes]
    split_coord_cols = []
    for col in pre_frame_attributes:
        if col in split_coord_attributes:
            split_coord_cols.append(f'pre_{col}_x')
            split_coord_cols.append(f'pre_{col}_y')
    for col in post_frame_attributes:
        if col in split_coord_attributes:
            split_coord_cols.append(f'post_{col}_x')
            split_coord_cols.append(f'post_{col}_y')
    return ["frame_index"] + pre_frame_cols + post_frame_cols + split_coord_cols

def initialize_dataframe_dict(pre_frame_attributes, post_frame_attributes, split_coord_attributes):
    return {col: []  for col in compute_df_cols(pre_frame_attributes, post_frame_attributes, split_coord_attributes)}

def data_pre_proc_mvp(df):
       
    df['post_ground'].fillna(value = 0.0, inplace=True)
    df['post_last_hit_by'].fillna(value = 0.0, inplace=True)
    df['post_last_attack_landed'].fillna(value = 0.0, inplace=True)
    df['pre_damage'] = df['pre_damage'].apply(lambda x: x[0])
    ## TODO: all dev data set have airborne = None. Find out why. Version ?
    df['post_airborne'] = df['post_airborne'].astype(int)

    df['pre_state'] = df['pre_state'].astype(int)
    df['post_state'] = df['post_state'].astype(int)

    
    categorical_cols = ['pre_direction', 'post_direction']
    # import pdb 
    # pdb.set_trace()
    one_hot_features = [ pd.get_dummies([1, -1], prefix=col) for col in categorical_cols]
    # import pdb 
    # pdb.set_trace()

    df = pd.concat([df] + one_hot_features, axis=1)
    
    for col in categorical_cols:
        df.drop(col, axis=1, inplace=True)
    return df

def proc_attr_value(state, att_name):
       

    att_value = getattr(state, att_name)
    if att_name == "flags":
        att_value = att_value.__repr__()
    if att_name == "buttons":
        att_value = att_value.physical.value 
    if att_name == "triggers":
        att_value = round(att_value.physical.l, 3), round(att_value.physical.r, 3)
    return att_value


def extract_coord(state, att_name):
        att_value = getattr(state, att_name)
        if att_name == "triggers":
            return att_value.physical.l, att_value.physical.r
        else:
            return att_value.x, att_value.y
  
def proc_frame(dataframe_dict, frame, pre_frame_attributes, post_frame_attributes, split_coord_attributes):

    frame_index = frame.index
    dataframe_dict['frame_index'].append(frame_index)
    dataframe_dict['frame_index'].append(frame_index)
    valid_ports = [ item for item  in frame.ports if item != None]
    assert(len(valid_ports) == 2)
    
    char1_state_pre, char1_state_post = (valid_ports[0].leader.pre, valid_ports[0].leader.post)
    char2_state_pre, char2_state_post = (valid_ports[1].leader.pre, valid_ports[1].leader.post)
    
    for att_name in pre_frame_attributes:
        if att_name in split_coord_attributes:
            
            char1_coord = extract_coord(char1_state_pre, att_name)
            char2_coord = extract_coord(char2_state_pre, att_name)
            dataframe_dict[f'pre_{att_name}_x'].append(char1_coord[0])
            dataframe_dict[f'pre_{att_name}_x'].append(char2_coord[0])
            dataframe_dict[f'pre_{att_name}_y'].append(char1_coord[1])
            dataframe_dict[f'pre_{att_name}_y'].append(char2_coord[1])
        else:
            dataframe_dict[f'pre_{att_name}'].append(proc_attr_value(char1_state_pre, att_name))
            dataframe_dict[f'pre_{att_name}'].append(proc_attr_value(char2_state_pre, att_name))

    for att_name in post_frame_attributes:
        if att_name in split_coord_attributes:
            char1_coord = extract_coord(char1_state_post, att_name)
            char2_coord = extract_coord(char2_state_post, att_name)
            dataframe_dict[f'post_{att_name}_x'].append(char1_coord[0])
            dataframe_dict[f'post_{att_name}_x'].append(char2_coord[0])
            dataframe_dict[f'post_{att_name}_y'].append(char1_coord[1])
            dataframe_dict[f'post_{att_name}_y'].append(char2_coord[1])
        else:
            dataframe_dict[f'post_{att_name}'].append(proc_attr_value(char1_state_post, att_name))
            dataframe_dict[f'post_{att_name}'].append(proc_attr_value(char2_state_post, att_name))
