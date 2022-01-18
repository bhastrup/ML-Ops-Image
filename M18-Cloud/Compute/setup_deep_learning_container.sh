#!/bin/bash

## Save environment variables
export IMAGE_FAMILY="pytorch-latest-cpu"
export ZONE="europe-west1-b"
export INSTANCE_NAME="deep-cybernetic-butler"

## Setup deep learning container
gcloud compute instances create $INSTANCE_NAME --zone=$ZONE --image-family=$IMAGE_FAMILY --image-project=deeplearning-platform-release

