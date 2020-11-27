#!/usr/bin/env bash
export IMAGE_FAMILY="pytorch-1-6-cpu-ubuntu-1804"
export ZONE="us-east1-c"
export INSTANCE_NAME="train-exp-cpu-$1"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --custom-cpu=2 --custom-memory=8 --custom-vm-type=n2 \
  --boot-disk-size=64GB
