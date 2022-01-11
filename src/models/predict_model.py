import argparse
import sys
import os

import glob
import numpy as np
import pandas as pd
import torch
from torch import optim, nn

from src.models.model import CNN_MNIST, CNN_MNIST_STRIDE, CNN_MNIST_DILATION


def predict(model, images):

    predictions = model(images)

    return None


if __name__ == "__main__":
    """
    Train on data from processed and save into models
    or load from models and evaluate
    """

    parser = argparse.ArgumentParser(
        description="Script for either training or evaluating",
        usage="python3 src/models/train_model.py <command> <LOAD_DATA_DIR> <MODEL_SAVE_DIR>",
    )
    parser.add_argument("--MODEL_LOAD", help="Path to model checkpoint")
    parser.add_argument("--model_name", help="Name of neural network model")
    parser.add_argument("--LOAD_DATA_DIR", help="Path to processed data")

    args = parser.parse_args()
    config = vars(args)

    script_dir = os.path.dirname(__file__)
    MODEL_LOAD = os.path.join(script_dir, "../../", config["MODEL_LOAD"])
    LOAD_DATA_DIR = os.path.join(script_dir, "../../", config["LOAD_DATA_DIR"])

    model_name = config["model_name"]
    if model_name == "CNN_MNIST":
        model = CNN_MNIST().float()
    elif model_name == "CNN_MNIST_STRIDE":
        model = CNN_MNIST_STRIDE().float()
    elif model_name == "CNN_MNIST_DILATION":
        model = CNN_MNIST_DILATION().float()

    state_dict = torch.load(MODEL_LOAD)
    model.load_state_dict(state_dict)
    images = torch.tensor(np.load(LOAD_DATA_DIR), dtype=torch.float)

    predict(model=model, images=images)
