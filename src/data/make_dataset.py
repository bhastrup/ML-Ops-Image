# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms


def main():
    """
        Load from raw, flatten image and save to processed 
    """
    parser = argparse.ArgumentParser(
        description="Script for either training or evaluating",
        usage="python3 src/data/make_dataset.py <LOAD_DIR> <SAVE_DIR>"
    )
    parser.add_argument("--LOAD_DIR", help="Path to raw data")
    parser.add_argument("--SAVE_DIR", help="Path to processed")
    parser.add_argument("--DATASET_NAME", help="Name of saved file once processed")

    args = parser.parse_args()
    config = vars(args)

    # SOLUTION:
    script_dir = os.path.dirname(__file__)
    LOAD_DIR = os.path.join(script_dir, "../../", config["LOAD_DIR"])
    SAVE_DIR = os.path.join(script_dir, "../../", config["SAVE_DIR"])
    os.makedirs(SAVE_DIR, exist_ok=True)
    DATASET_NAME = config["DATASET_NAME"]
    save_name = f'{SAVE_DIR}/{DATASET_NAME}.npz'

    # Load data
    mnist_data_paths = sorted(
        glob.glob(os.path.join(LOAD_DIR, '*'))
    )
    print(mnist_data_paths)

    all_images = np.array([], dtype=float).reshape(0, 28, 28)
    all_labels = np.array([], dtype=float)

    for f in mnist_data_paths:
        data = np.load(f)
        all_images = np.vstack([all_images, data["images"]])
        all_labels = np.append(all_labels, data["labels"])
        print(f'finished processing {f}')

    print(all_images.shape)
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))])

    all_images = torch.transpose(torch.transpose(transform(all_images), 0, 1), 1, 2).numpy()
    np.savez(save_name, images=all_images, labels=all_labels, allow_pickle=data["allow_pickle"])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    main()
