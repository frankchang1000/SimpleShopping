# SimpleShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""The Supervised Learning Training Module."""
# ---------------------------------------------------------------------------

import os
import shutil
import tensorflow as tf

from tqdm import tqdm
from . import training_utils as utils


def train(training_dataset: tf.data.Dataset,
          validation_dataset: tf.data.Dataset,
          model: tf.keras.models.Model,
          loss: tf.keras.losses.Loss,
          validation_loss: tf.keras.losses.Loss,
          optimizer: tf.keras.optimizers.Optimizer,
          training_dir: str = "training_dir/efficientnet",
          from_checkpoint: bool = False,
          total_steps: int = None,
          precision: str = "mixed_float16",
          epochs: int = 50,
          learning_rate: float = 1e-4,
          learning_rate_warmup: int = 0,
          learning_rate_numwait: int = 0,
          validation_frequency: int = 10,
          checkpoint_frequency: int = 10) -> tf.keras.models.Model:
    """Trains model on SL.
    Params:
        model: tf.keras.models.Model
            The model to train.
        loss: tf.keras.losses.Loss
            The loss function to use.
        optimizer: tf.keras.optimizers.Optimizer
            The optimizer to use.
        hyperparams: dict
            The hyperparameters to use.
        training_dir: str
            The directory to save the training logs.
        from_checkpoint: bool
            Whether to load the model from a checkpoint or not.
        total_steps: int
            The total number of steps to train for.
        precision: str
            The precision to use.
        epochs: int
            The number of epochs to train for.
        learning_rate: float
            The learning rate to use.
        learning_rate_warmup: int
            The number of steps to warmup the learning rate.
        learning_rate_numwait: int
            The number of steps to wait before decreasing the learning rate.
        validation_frequency: int
            The number of steps between validations.
        checkpoint_frequency: int
            The number of steps between checkpoints.
    Returns:
        tf.keras.models.Model
            The trained model.
    """
    steps_counter = 0

    if from_checkpoint:
        checkpoint_dir = os.path.join(training_dir, "checkpoints")
        # Checkpoints
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, 
            model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, 
            checkpoint_dir,
            3)
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        steps_counter = int(int(checkpoint.save_counter) * total_steps)
        epochs = epochs - int(checkpoint.save_counter)
        print(f"Num epochs left: {epochs}")
        print(f"Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
        print(f"Continuing from step: {steps_counter}")
        
        # Tensorboard Logging
        tensorboard_dir = os.path.join(
            training_dir, "tensorboard")
        if os.path.exists(tensorboard_dir) is False:
            os.makedirs(tensorboard_dir)
        tensorboard_file_writer = tf.summary.create_file_writer(
            tensorboard_dir)
        tensorboard_file_writer.set_as_default()
    else:
        # Initialize the directories if they do not exist
        if os.path.exists(training_dir) and from_checkpoint == False:
            # Prevents accidental deletions
            input("Press Enter to delete the current directory and continue.")
            shutil.rmtree(training_dir)
        else:
            os.makedirs(training_dir)

        # Tensorboard Logging
        tensorboard_dir = os.path.join(
            training_dir, "tensorboard")
        if os.path.exists(tensorboard_dir) is False:
            os.makedirs(tensorboard_dir)
        tensorboard_file_writer = tf.summary.create_file_writer(
            tensorboard_dir)
        tensorboard_file_writer.set_as_default()

        # Checkpoints
        checkpoint_dir = os.path.join(training_dir, "checkpoints")
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, 
            model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, 
            checkpoint_dir,
            3)

    @tf.function #(jit_compile=True)
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image, training=True)
            loss_value = tf.reduce_mean(loss(label, predictions))
            if precision == "mixed_float16":
                loss_value = optimizer.get_scaled_loss(loss_value)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        if precision == "mixed_float16":
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_value
    
    @tf.function #(jit_compile=True)
    def validation_step(image, label):
        predictions = model(image, training=False)
        return validation_loss(label, predictions)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for image, label in tqdm(training_dataset):
            loss_value = train_step(image, label)
            steps_counter += 1
            optimizer.learning_rate.assign(
                utils.learning_rate(
                    steps_counter,
                    learning_rate,
                    total_steps,
                    learning_rate_warmup,
                    learning_rate_numwait))
            tqdm.write(f" Step: {steps_counter}, Loss: {loss_value}")
            utils.update_tensorboard({"Loss": loss_value}, steps_counter)

        if epoch % checkpoint_frequency == 0:
            checkpoint_manager.save()

        if epoch % validation_frequency == 0:
            vloss = 0
            for val_step, (image, label) in tqdm(enumerate(validation_dataset)):
                vloss += validation_step(image, label)
                tqdm.write(f"Validation Step: {val_step}, Loss: {vloss}")
            vloss /= val_step
            print(f"Validation Loss: {vloss}")
            utils.update_tensorboard(
                {"Validation-Loss": vloss}, val_step)
        
        utils.update_tensorboard(
            {"Loss": loss_value, "Validation Accuracy": vloss + 0.0}, 
            step=steps_counter)
        
    return model
