# SimpleShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Creates the labels.txt file then tests the dataset.py function."""
# ---------------------------------------------------------------------------

import os
import argparse

import tensorflow as tf

from src import dataset
from src.training.training_utils import read_files


def main(args):
    """Create the labels file and test the dataset."""
    # Write to file
    print("Writing the label classes to file.")
    with open(args.label_dir, "w+") as f:
        for x in os.listdir(args.dataset_dir):
            f.write(x)
            f.write("\n")
        f.close()
    
    # Debug mode for tensorflow
    tf.data.experimental.enable_debug_mode()
    
    # Reads the labels file
    label_list = read_files(args.labels_dir)
    # Create the dataset
    ds, _, _ = dataset.Dataset(
        dataset_dir=args.dataset_dir,
        label_list=label_list,
        batch_size=args.batch_size)()
    
    # try:
    for (x, y) in ds:
        print(f"Image is (Batch Size, Image Shape, 3): {x.shape}")
        print(f"Label is (Batch Size, Num Classes): {y.shape}")
        break
        
        print("Dataset pipeline test is complete.")
    # except:
    #     print("Dataset pipeline test failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-dir",
        type=str,
        default="data/images/labels.txt")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/images")
    args = parser.parse_args()

    main(args)
