#!/usr/bin/env bash
export IMAGE_FAMILY="pytorch-1-6-cu110-ubuntu-1804"
export ZONE="us-east1-c"
export INSTANCE_NAME="train-exp-gpu"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True" \
  --custom-cpu=2 --custom-memory=8 --custom-vm-type=n1 \
  --boot-disk-size=200GB \
  --metadata-from-file="startup-script=./gcloud-startup-script.sh"
