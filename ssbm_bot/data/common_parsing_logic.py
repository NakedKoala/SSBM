

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def scale(tensor):

    mean = torch.tensor([ 1.41423043e+00,  7.85902318e+00,  1.41610300e+00,  7.80339904e+00,
        5.54843011e+01,  1.20621781e+01,  5.88113544e+01,  5.59438437e+00,
        5.29972526e-01,  5.54842529e+01,  5.03226905e-01,  4.96555996e-01,
        5.03235357e-01,  4.96547543e-01,  2.71878764e+00,  1.67740513e+00,
        3.57006903e+00,  1.67848391e+00,  3.52045915e+00,  4.57292101e+01,
        1.09459958e+01,  5.92744284e+01,  6.77617745e+00,  4.64926234e-01,
        4.57292101e+01,  5.07767453e-01,  4.92015448e-01,  5.07770270e-01,
        4.92012630e-01,  2.71254100e+00,  4.33544745e-03, -5.96091991e-02,
       -1.71527698e-03, -1.15610866e-02,  1.12961590e-01,  8.38983897e-02,
       -9.18879334e-04, -3.93047363e-02, -2.32658232e-03,  2.02196862e-03,
        8.83636491e-02,  7.81077922e-02])
    std = torch.tensor([63.15214909, 31.85448842, 63.20340685, 31.88907913, 45.4699356 ,
       12.17418325,  3.79583704, 17.53603757,  0.49910085, 45.46992937,
        0.49998957,  0.49998814,  0.49998951,  0.4999881 ,  1.07013528,
       67.62576071, 25.21796792, 67.71633154, 25.23324746, 38.06941933,
       11.66731482,  2.81122129, 23.5396452 ,  0.49876832, 38.06941933,
        0.49993966,  0.49993625,  0.49993964,  0.4999362 ,  1.07326573,
        0.65573621,  0.37714628,  0.16054365,  0.15059148,  0.28983791,
        0.25826303,  0.60554965,  0.44761662,  0.11764879,  0.13110503,
        0.2549518 ,  0.24309364])
    return (tensor - mean) / std

def bitmap_to_number(bits):
    power = 0
    ans = 0
    while len(bits) > 0:
        ans += bits.pop() * (2 ** power)
        power += 1
    return ans

def proc_button_press(buttons_values_np, button_press_indicator_dim, bitmap=False):

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

            # Combine X/Y and L/R
            # "Y" "X" "B" "A" "L" "R" "Z"
            bits = [max(selected_bits[0], selected_bits[1])] + selected_bits[2:4] + [max(selected_bits[4], selected_bits[5])] + [selected_bits[6]]

            if bitmap == True:
                return np.array(bits)
            else:
                result =  bitmap_to_number(bits)
                assert(result >= 0 and result <= 31)
                return result



        f = np.vectorize(convert_button_value_to_indicator_vector, otypes=[np.ndarray])
        # try:
        button_targets_np = np.stack(f(buttons_values_np), axis=0)

        return button_targets_np

def proc_df(df, char_id, opponent_id, frame_delay, button_press_indicator_dim, dist=True):


        feat_cols = ['pre_state',  'post_state','pre_position_x', 'pre_position_y',  'post_position_x', 'post_position_y', \
                'post_damage', 'post_state_age', 'post_shield', \
                'post_hit_stun', 'post_airborne', 'pre_damage',  'post_direction_1', 'post_direction_-1', \
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

        char_target_button_values_np = char_targets_df['pre_buttons'].to_numpy().reshape(-1)
        char_cmd_button_values_np = char_cmd_df['pre_buttons'].to_numpy().reshape(-1)
        opp_cmd_button_values_np = opp_cmd_df['pre_buttons'].to_numpy().reshape(-1)
        char_targets_df.drop('pre_buttons', axis=1, inplace=True)
        char_cmd_df.drop('pre_buttons', axis=1, inplace=True)
        opp_cmd_df.drop('pre_buttons', axis=1, inplace=True)

        if dist == True:
            char_target_button_targets_np = proc_button_press(char_target_button_values_np, button_press_indicator_dim)
        else:
            char_target_button_targets_np = proc_button_press(char_target_button_values_np, button_press_indicator_dim, bitmap=True)
        char_cmd_button_targets_np = proc_button_press(char_cmd_button_values_np, button_press_indicator_dim)
        opp_cmd_button_targets_np = proc_button_press(opp_cmd_button_values_np, button_press_indicator_dim)


        features_np = np.concatenate([char_features_df.to_numpy()[:,0:2], char_cmd_button_targets_np.reshape(-1, 1),  \
                                      opp_features_df.to_numpy()[:,0:2],  opp_cmd_button_targets_np.reshape(-1, 1), \
                                      char_features_df.to_numpy()[:,2:], opp_features_df.to_numpy()[:,2:]], axis=1)

        # TODO model is currently fed in opponent controller inputs.
        # This might be regarded as cheating from human POV, so we should
        # consider removing opponent controller input from features in the future.
        # Ideally, the model will still be able to tell what the opponent is doing
        # by looking at position + action state + state age
        features_np = np.concatenate([features_np, char_cmd_df.to_numpy(), opp_cmd_df.to_numpy()], axis=1)

        # scaler = StandardScaler()
        # features_np[:,4:] = scaler.fit_transform(features_np[:,4:])
        # 4 -> 52
        # import pdb
        # pdb.set_trace()

        # features_np = np.concatenate([features_np, char_cmd_button_targets_np, opp_cmd_button_targets_np], axis=1)

        cts_targets_np = char_targets_df.to_numpy()


        features_tensor, cts_targets_tensor, char_button_targets_tensor = torch.from_numpy(features_np), torch.from_numpy(cts_targets_np), torch.from_numpy(char_target_button_targets_np)
        features_tensor[:,6:] = scale(features_tensor[:,6:])
        # import pdb
        # pdb.set_trace()
        assert(features_tensor.shape[1] == 17 * 2 + 7 * 2)
        assert(cts_targets_tensor.shape[1] == 6)
        # assert(char_button_targets_tensor.shape[1] == 5)

        # import pdb
        # pdb.set_trace()
        return features_tensor.float(), cts_targets_tensor.float(),  char_button_targets_tensor.float()

def _fix_align_queue(align_queue, window_size, tensor_shape):
    for _ in range(window_size - len(align_queue)):
        align_queue.appendleft(torch.zeros(*tensor_shape))

# align_queue is deque
def align(align_queue, window_size, new_frame):
    if len(align_queue) < window_size:
        _fix_align_queue(align_queue, window_size, new_frame.shape)
    align_queue.popleft()
    align_queue.append(new_frame)
    return torch.stack(tuple(align_queue))

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
    # df['post_last_hit_by'].fillna(value = 0.0, inplace=True)
    # df['post_last_attack_landed'].fillna(value = 0.0, inplace=True)
    df['pre_damage'] = df['pre_damage'].apply(lambda x: x[0])
    ## TODO: all dev data set have airborne = None. Find out why. Version ?
    df['post_airborne'] = df['post_airborne'].astype(int)

    df['pre_state'] = df['pre_state'].astype(int)
    df['post_state'] = df['post_state'].astype(int)

    categorical_cols = ['pre_direction', 'post_direction']
    # import pdb
    # pdb.set_trace()
    # add 1, -1 dummy rows to ensure 1, -1 are always generated by pd.dummies
    df = df.append({col:int(-1) for col in categorical_cols}, ignore_index=True)
    df = df.append({col:1 for col in categorical_cols}, ignore_index=True)
    # convert categories to int - dummy rows were added as float
    df = df.astype({col: 'int32' for col in categorical_cols})
    # columns argument will overwrite the target columns with one-hot encoded columns
    df = pd.get_dummies(df, columns=categorical_cols)
    # remove dummy rows
    df.drop(df.tail(2).index, axis=0, inplace=True)

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

