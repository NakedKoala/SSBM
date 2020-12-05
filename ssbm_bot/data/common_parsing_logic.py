

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def scale(tensor, include_opp_input=True):
    # NOTE data from falcon vs fox on FD, v2.0.1+
    if include_opp_input:
        mean = torch.tensor([ 5.02553956e-01,  4.95817799e-01,  1.40581029e+00,  7.89299335e+00,
            1.40768720e+00,  7.83883126e+00,  5.53661720e+01,  1.20425827e+01,
            5.87270675e+01,  5.59028995e+00,  5.28925141e-01,  5.53661238e+01,
            5.02546097e-01,  4.95825659e-01,  2.71677399e+00,  1.29873838e+00,
            5.07101919e-01,  4.91269836e-01,  1.66814213e+00,  3.59902029e+00,
            1.66925132e+00,  3.55073426e+00,  4.56217565e+01,  1.09246301e+01,
            5.91899782e+01,  6.75767245e+00,  4.63838514e-01,  4.56217565e+01,
            5.07098508e-01,  4.91273247e-01,  2.71052200e+00,  1.39730901e+00,
            4.32836837e-03, -5.96757321e-02, -1.71501006e-03, -1.15619207e-02,
            1.12902121e-01,  8.38341973e-02, -9.35819824e-04, -3.93679828e-02,
           -2.32215766e-03,  2.02003898e-03,  8.82905794e-02,  7.80616650e-02])
        std = torch.tensor([ 0.49999347,  0.49998253, 63.02228144, 31.72558865, 63.07312014,
           31.75579231, 45.46381422, 12.17110586,  4.39057318, 17.53156162,
            0.49916263, 45.4638078 ,  0.49999351,  0.49998258,  1.07355414,
            0.74220231,  0.49994954,  0.49992377, 67.43632909, 25.079946  ,
           67.52402291, 25.08930499, 38.05912047, 11.65851841,  3.58490302,
           23.51914017,  0.49869064, 38.05912047,  0.49994962,  0.49992385,
            1.07662804,  0.71687031,  0.65542925,  0.37686592,  0.16051784,
            0.15054085,  0.28979694,  0.2581864 ,  0.60523536,  0.44734949,
            0.11757759,  0.13107539,  0.25487635,  0.24304935])
    else:
        mean = torch.tensor([ 5.0110638e-01,  4.9725160e-01,  1.2718267e+00,  7.9390016e+00,
            1.2731744e+00,  7.8842220e+00,  5.5050415e+01,  1.2016634e+01,
            5.8729122e+01,  5.5891562e+00,  5.2796239e-01,  5.5050369e+01,
            5.0110865e-01,  4.9724931e-01,  2.7175202e+00,  1.2996517e+00,
            5.0666440e-01,  4.9169359e-01,  1.5811079e+00,  3.6081700e+00,
            1.5833863e+00,  3.5589495e+00,  4.5626381e+01,  1.0828938e+01,
            5.9171535e+01,  6.7977214e+00,  4.6230331e-01,  4.5626369e+01,
            5.0665838e-01,  4.9169958e-01,  2.7044199e+00,  1.4005343e+00,
            4.2670579e-03, -6.0793433e-02, -2.1057569e-03, -1.1726107e-02,
            1.1069417e-01,  8.5245676e-02])
        std = torch.tensor([ 0.4999988 ,  0.4999925 , 62.96348   , 31.91961   , 63.013542  ,
           31.9502    , 45.35477   , 12.135419  ,  4.404445  , 17.571968  ,
            0.49921754, 45.354767  ,  0.4999988 ,  0.49999246,  1.0734138 ,
            0.7424888 ,  0.49995562,  0.49993104, 67.272995  , 25.13912   ,
           67.36112   , 25.148598  , 38.240543  , 11.368521  ,  3.627853  ,
           23.622425  ,  0.49857697, 38.240543  ,  0.4999557 ,  0.49993113,
            1.0750854 ,  0.7149457 ,  0.65529776,  0.37615   ,  0.16024348,
            0.15091088,  0.2874641 ,  0.26076567])

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



def proc_df(df, char_port, frame_delay, button_press_indicator_dim, include_opp_input=True, dist=True):

        feat_cols = ['pre_state',  'post_state', 'post_character', 'pre_direction_1', 'pre_direction_-1', 'pre_position_x', 'pre_position_y',  'post_position_x', 'post_position_y',\
                'post_damage', 'post_state_age', 'post_shield', \
                'post_hit_stun', 'post_airborne', 'pre_damage',  'post_direction_1', 'post_direction_-1', \
                'post_stocks', 'post_jumps']
        target_cols = ['pre_joystick_x', 'pre_joystick_y',  'pre_cstick_x', 'pre_cstick_y', \
                       'pre_triggers_x', 'pre_triggers_y', 'pre_buttons']

        # NOTE assumes there are only 2 players playing
        df_char = df[df['port'] == char_port]
        df_opp = df[df['port'] != char_port]

        # NOTE frame delay convention: 0 frame delay means 'predict the next frame,' which means a shift of 1 is needed.
        char_features_df, char_cmd_df, char_targets_df = df_char[feat_cols].shift(frame_delay).fillna(0), \
                                                         df_char[target_cols].shift(frame_delay).fillna(0), \
                                                         df_char[target_cols].shift(-1).fillna(0),

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
            recent_buttons_np = char_target_button_targets_np
        else:
            recent_buttons_np = proc_button_press(char_target_button_values_np, button_press_indicator_dim)
            char_target_button_targets_np = proc_button_press(char_target_button_values_np, button_press_indicator_dim, bitmap=True)
        char_cmd_button_targets_np = proc_button_press(char_cmd_button_values_np, button_press_indicator_dim)
        opp_cmd_button_targets_np = proc_button_press(opp_cmd_button_values_np, button_press_indicator_dim)

        num_examples = len(char_features_df)
        char_features_np = char_features_df.to_numpy()
        opp_features_np = opp_features_df.to_numpy()
        features_list = [df_char['stage'].to_numpy().reshape(-1, 1)]
        features_list.extend([char_features_np[:, 0:3], char_cmd_button_targets_np.reshape(-1, 1), opp_features_np[:, 0:3]])
        if include_opp_input:
            features_list.append(opp_cmd_button_targets_np.reshape(-1, 1))

        features_list.extend([
            char_features_np[:,3:], opp_features_np[:,3:], \
            char_cmd_df.to_numpy()
        ])
        if include_opp_input:
            features_list.append(opp_cmd_df.to_numpy())

        features_np = np.concatenate(features_list, axis=1)
        # scaler = StandardScaler()
        # features_np[:,9:] = scaler.fit_transform(features_np[:,9:])
        # # 4 -> 52
        # import pdb
        # pdb.set_trace()

        # features_np = np.concatenate([features_np, char_cmd_button_targets_np, opp_cmd_button_targets_np], axis=1)

        cts_targets_np = char_targets_df.to_numpy()
        recent_actions_np = np.concatenate([
            recent_buttons_np.reshape(-1, 1), cts_targets_np
        ], axis=1)


        features_tensor, cts_targets_tensor, char_button_targets_tensor = torch.from_numpy(features_np), torch.from_numpy(cts_targets_np), torch.from_numpy(char_target_button_targets_np)

        # normalize continuous features
        if include_opp_input:
            features_tensor[:,9:] = scale(features_tensor[:,9:])
        else:
            features_tensor[:,8:] = scale(features_tensor[:,8:], include_opp_input=False)

        recent_actions_tensor = torch.from_numpy(recent_actions_np)
        # import pdb
        # pdb.set_trace()
        assert(cts_targets_tensor.shape[1] == 6)
        assert(recent_actions_tensor.shape[1] == 6 + 1)
        # assert(char_button_targets_tensor.shape[1] == 5)
        if include_opp_input:
            assert(features_tensor.shape[1] == 19 * 2 + 7 * 2 + 1)
        else:
            assert(features_tensor.shape[1] == (19 * 2) + 7 * 1 + 1)

        # import pdb
        # pdb.set_trace()

        return features_tensor.float(), cts_targets_tensor.float(),  char_button_targets_tensor.float(), \
            recent_actions_tensor.float()

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
    return ["frame_index", "port", "stage"] + pre_frame_cols + post_frame_cols + split_coord_cols

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

def proc_frame(dataframe_dict, frame, stage, pre_frame_attributes, post_frame_attributes, split_coord_attributes):

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

    # push port and stage information
    valid_port_idx = [ idx for idx, item in enumerate(frame.ports) if item != None]
    assert(len(valid_port_idx) == 2)
    dataframe_dict['port'].append(valid_port_idx[0])
    dataframe_dict['port'].append(valid_port_idx[1])
    dataframe_dict['stage'].append(stage)
    dataframe_dict['stage'].append(stage)
