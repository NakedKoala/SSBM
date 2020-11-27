#!/usr/bin/env bash

cd ~/
gsutil cp gs://f2020-cs486-g84-data/dev_data_csv.zip .
unzip -q dev_data_csv.zip

git clone https://git.uwaterloo.ca/w2999wen/f2020-cs486-g84.git SSBM
mv dev_data_csv SSBM

cd SSBM

pip install py-slippi melee
git checkout -b wwen-action-head-train origin/wwen-action-head-train

cd ~/
