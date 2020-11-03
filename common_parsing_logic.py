

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

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

        # FIXME need to handle case where character ID can change
        # e.g. zelda/sheik transform
        # The above should be the only case.
        # Also handle case where both players use the same character.
        # Using port is probably a better option in the long run
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

        # TODO model is currently fed in opponent controller inputs.
        # This might be regarded as cheating from human POV, so we should
        # consider removing opponent controller input from features in the future.
        # Ideally, the model will still be able to tell what the opponent is doing
        # by looking at position + action state + state age
        features_np = np.concatenate([features_np, char_cmd_df.to_numpy(), opp_cmd_df.to_numpy()], axis=1)
        scaler = StandardScaler()
        features_np[:,4:] = scaler.fit_transform(features_np[:,4:])
        features_np = np.concatenate([features_np, char_cmd_bin_cls_targets_np, opp_cmd_bin_cls_targets_np], axis=1)

        cts_targets_np = char_targets_df.to_numpy()


        features_tensor, cts_targets_tensor, char_bin_class_targets_tensor = torch.from_numpy(features_np), torch.from_numpy(cts_targets_np), torch.from_numpy(char_target_bin_cls_targets_np)

        assert(features_tensor.shape[1] == 20 * 2 + 13 * 2)
        assert(cts_targets_tensor.shape[1] == 6)
        assert(char_bin_class_targets_tensor.shape[1] == 7)

        return features_tensor.float(), cts_targets_tensor.float(),  char_bin_class_targets_tensor.float()

def _fix_align_queue(align_queue, window_size, tensor_shape):
    for _ in range(window_size - len(align_queue)):
        align_queue.insert(0, torch.zeros(*tensor_shape))

# align_queue list list - not as efficient as queue, but we need to copy to tuple/list for torch.stack anyway
def align(align_queue, window_size, new_frame):
    if len(align_queue) < window_size:
        _fix_align_queue(align_queue, window_size, new_frame.shape)
    align_queue.pop(0)
    align_queue.append(new_frame)
    return torch.stack(align_queue)

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
    one_hot_features = [ pd.get_dummies(df[col], prefix=col) for col in categorical_cols]

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

