# SimpleShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Creates the dataset using tf.data training pipeline."""
# ---------------------------------------------------------------------------

import os
import random

import numpy as np
import tensorflow as tf
import albumentations as A

from tqdm import tqdm
from typing import Tuple


class Dataset:
    def __init__(self,
                 image_dims: Tuple[int, int] = (512, 512),
                 dataset_dir: str = None,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 augment: bool = True,
                 label_list: list = None,
                 train_val_split: float = 0.8):
        """Creates the dataset.
        Params:
            image_dims: Tuple[int, int]
                The image dimension (w, h).
            dataset_dir: str
                The directory of the dataset.
            batch_size: int
                The batch size.
            shuffle: bool
                Whether to shuffle the dataset.
            augment: bool
                Whether to augment the dataset.
            label_list: List
                The list of labels.
            train_val_split: float
                The train/val split. 0.2 of train is validation.
        """
        self.image_dims = image_dims
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = len(label_list)
        self.label_list = label_list
        self.train_val_split = train_val_split

    def parse_image(self, image_path: str) -> tf.Tensor:
        """Images are parsed from the dataset."""
        image_path = bytes.decode(image_path[0], "utf-8")
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_dims)
        return image
    
    def one_hot_encode(self, label: tf.Tensor) -> tf.Tensor:
        """One-hot encodes the labels."""
        label = tf.one_hot(label, depth=self.num_classes)
        label = tf.cast(label, tf.int32)
        label = tf.squeeze(label, axis=0)
        return label
    
    def augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """Augments the images.
        Params:
            image: The image to augment.
        Returns:
            The augmented image.
        """
        image = np.array(image, np.uint8)
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.4),
            A.GridDropout(p=0.2),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.1)])
        image = augmentations(image=image)["image"]
        return tf.constant(image, dtype=tf.float32)

    def parse_classification(self, 
                             images: np.array, 
                             labels: np.array) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parses the classification dataset."""
        image = self.parse_image(images)
        label = self.one_hot_encode(labels)
        if self.augment:
            image = self.augment_image(image)
        return image, label

    def get_dataset(self, 
                    images: list, 
                    labels: list) -> tf.data.Dataset:
        """Creates the dataset.
        Params:
            images: List
                The list of images.
            labels: List
                The list of labels
        Returns:
            The dataset.
        """
        ds = tf.data.Dataset.from_tensor_slices(images)
        ds_labels = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((ds, ds_labels))
        ds = ds.map(
            lambda x, y: tf.numpy_function(
                self.parse_classification,
                [x, y],
                [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.batch_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
    
    def __call__(self):
        """Returns the dataset."""
        labels_list = []
        images_list = []

        # Parse the images and labels from the dataset.
        print("Parsing the images for the dataset...")
        for label in tqdm(os.listdir(self.dataset_dir)):
            labels_dir = os.path.join(self.dataset_dir, label)
            # Ensure that only directories would be parsed.
            if os.path.isdir(labels_dir):
                pass
            else:
                continue

            for image in os.listdir(labels_dir):
                image_path = os.path.join(labels_dir, image)
                labels_list.append(self.label_list.index(label)) # The index in the list
                images_list.append(image_path)
        
        # Shuffle the dataset.
        dataset = list(zip(images_list, labels_list))
        random.shuffle(dataset)
        images_dataset, labels_dataset = list(zip(*dataset))

        # Expand the dims of the images.
        # This is required for the from_tensor_slices function.
        images_dataset = tf.expand_dims(images_dataset, axis=-1)
        labels_dataset = tf.expand_dims(labels_dataset, axis=-1)

        # Split the dataset randomly
        train_images_dataset = images_dataset[:int(len(images_dataset) * self.train_val_split)]
        train_labels_dataset = labels_dataset[:int(len(labels_dataset) * self.train_val_split)]
        val_images_dataset = images_dataset[int(len(images_dataset) * self.train_val_split):]
        val_labels_dataset = labels_dataset[int(len(labels_dataset) * self.train_val_split):]

        # Create the dataset
        training = self.get_dataset(
            train_images_dataset, train_labels_dataset)
        if self.train_val_split != 1:
            validation = self.get_dataset(
                val_images_dataset, val_labels_dataset)

        return training, validation, len(train_images_dataset)/self.batch_size