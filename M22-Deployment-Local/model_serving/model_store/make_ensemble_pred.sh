#!/bin/bash

## First both models

## Create model
torch-model-archiver --model-name ensemble_model --version 1.0 --serialized-file ./deployable_ensemble_model.pt --export-path . --extra-files ./index_to_name.json --handler image_classifier
