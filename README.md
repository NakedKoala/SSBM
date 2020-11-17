# f2020-cs486-g84
Super Smash Bros. Melee agent trained using tournament data for the Fall 2020 CS 486 project.

## Project description:

This project trains an AI agent to play SSBM with Slippi replay data. The model is based on LSTM.


## Features:

### Data preprocessing:

`common_parsing_logic.py`, `slp_parser.py` and `dataset.py` contain data preprocessing utilities in . These utilities parse SLP replay files to CSV and convert CSV to pytorch tensor features, required library, commands to run your code and an example output).

### Model

- `mvp_model.py` contains the basedline ANN model. The model output is deterministic action.

- `lstm_model.py` contains the LSTM model.  The model output is deterministic action.

- `mvp_model_prob.py` contains theANN model + probability action head. The model output is a distribution of action.

- `lstm_model_prob.py` contains the LSTM model + probability action head. The model output is a distribution of action.

### Training script

`train.py` and `train_prob.py` contains training loop for training models with/without probability action head.

### Infrastructure Adaptor

`infra_adaptor.py` contains adaptor that converts game state to input tensor and output tensor to game action.


### Required Library

Use `environment.yml` to create a Conda environment. It contains all necessary dependency


## Usage

Convert .slp replay files to CSV

```

# Sample usage for the parser
if __name__ == '__main__':
    parser = SLPParser(src_dir="./dev_data_slp", dest_dir="./dev_data_csv")
    parser()

```


Training a LSTM model with 15 frame window size for 5 epochs

```

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trn_ds = SSBMDataset(src_dir="./trn_data", char_id=2, opponent_id=1, window_size=15, device=device)
trn_dl = DataLoader(trn_ds, batch_size=256, shuffle=True, num_workers=0)

val,_ds = SSBMDataset(src_dir="./val_data", char_id=2, opponent_id=1, window_size=15, device=device)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=True, num_workers=0)

model = SSBM_LSTM(100, 50, hidden_size=4, num_layers=1, bidirectional=False)
train(model, trn_dl, val_dl, 5,  5000, device, [1] * 5)

```


Inference. Looping through frames of a single replay and produce action prediction


```
model = SSBM_MVP(100, 50)
model.load_state_dict(torch.load('./weights/mvp_fit5_EP7_VL0349.pth',  map_location=lambda storage, loc: storage))
model.eval()
slp_object = Game("./(YOTB) Fox vs Falcon (MN) [FD] Game_20200222T152806.slp")

cmd_lst = []
for frame in tqdm(slp_object.frames):

    feature_tensor = convert_frame_to_input_tensor(frame, char_id=2, opponent_id=1)
    cts_targets, button_targets = model(feature_tensor)
    cmd_lst.append(convert_output_tensor_to_command(cts_targets, button_targets))

```


## Example output of predicted action

```
{   'button': {   <Button.BUTTON_A: 'A'>: 0,
                      <Button.BUTTON_B: 'B'>: 0,
                      <Button.BUTTON_X: 'X'>: 1,
                      <Button.BUTTON_Y: 'Y'>: 0,
                      <Button.BUTTON_Z: 'Z'>: 0,
                      <Button.BUTTON_L: 'L'>: 0,
                      <Button.BUTTON_R: 'R'>: 0},
        'c_stick': (-0.0011587474728003144, -0.0007997926441021264),
        'l_shoulder': 0.02555307187139988,
        'main_stick': (-0.5101867318153381, -0.2363722324371338),
        'r_shoulder': 0.011153359897434711
}

```