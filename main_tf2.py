# SimpleShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Runs the training module."""
# ---------------------------------------------------------------------------

import os
import argparse

import tensorflow as tf

from src import dataset
from src.training import training_utils, training


def main(args):
    label_list = training_utils.read_files(
        os.path.join(
            args.dataset_dir,
            args.labels_file))
    training_dataset, validation_dataset, total_steps = dataset.Dataset(
        dataset_dir=args.dataset_dir,
        label_list=label_list,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        augment=args.augment,
        image_dims=args.image_dims)()
    
    efficientnet = model.get_model(
        model_name=args.model_name,
        num_classes=len(label_list),
        input_shape=args.image_dims)
    
    loss_func = tf.keras.losses.CategoricalCrossentropy( 
        reduction=tf.losses.Reduction.NONE)
    validation_loss_func = tf.keras.metrics.CategoricalAccuracy(
        name="val_acc_ema")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    if args.precision == "mixed_float16":
        print("Using mixed precision training.")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    trained_model = training.train(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=efficientnet,
        loss=loss_func,
        validation_loss=validation_loss_func,
        optimizer=optimizer,
        training_dir=args.training_dir,
        from_checkpoint=args.from_checkpoint,
        total_steps=total_steps,
        precision=args.precision,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        learning_rate_warmup=args.learning_rate_warmup,
        learning_rate_numwait=args.learning_rate_numwait,
        validation_frequency=args.validation_frequency,
        checkpoint_frequency=args.checkpoint_frequency)

    tf.keras.models.save_model(
        trained_model,
        os.path.join(args.training_dir, "model-exported"))
    
    print("Training is complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data\images",
        help="The directory containing the dataset.")
    parser.add_argument(
        "--labels-file",
        type=str,
        default="labels.txt",
        help="The file containing the labels.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b4",
        help="The model name.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="The batch size.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to augment the dataset.")
    parser.add_argument(
        "--image-dims",
        type=tuple,
        default=(256, 256),
        help="The image dimensions.")
    parser.add_argument(
        "--training-dir",
        type=str,
        default="training_dir/efficientnet",
        help="The directory to save the training logs.")
    parser.add_argument(
        "--from-checkpoint",
        type=bool,
        default=False,
        help="Whether to load the model from a checkpoint.")
    parser.add_argument(
        "--precision",
        type=str,
        default="mixed_float16",
        help="The precision to use.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="The number of epochs to train for.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4)
    parser.add_argument(
        "--learning-rate-warmup",
        type=int,
        default=0)
    parser.add_argument(
        "--learning-rate-numwait",
        type=int,
        default=0)
    parser.add_argument(
        "--validation-frequency",
        type=int,
        default=10,
        help="The frequency of validation steps.")
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=1,
        help="The frequency of checkpoint steps.")
    args = parser.parse_args()

    tf.keras.mixed_precision.set_global_policy(args.precision)
    tf.keras.backend.clear_session()

    main(args)