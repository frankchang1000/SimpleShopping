# SimplisticShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Runs inference on a dataset to calculate accuracy OR runs inference on 
   a single image."""
# ---------------------------------------------------------------------------

import argparse
import tensorflow as tf

from typing import Tuple

from .training import training_utils


@tf.function
def parse_image(image_file: str,
                image_dims: Tuple[int, int]) -> tf.Tensor:
    """Read and preprocess images.
    Params:
        image_file: str
            The path to the image file.
        image_dims: Tuple[int, int]
            The image dimensions (w, h).
    Returns:
        image: tf.Tensor
            The preprocessed image (1, *image_dims, 3).
        original_dims: Tuple[int, int]
            The original image dimensions (w, h).
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image)
    image = tf.image.resize(image, image_dims)
    image = tf.image.convert_image_dtype(
        images=image, 
        dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    original_shape = (image.shape[0], image.shape[0])
    return image, original_shape


@tf.function
def run_inference(image: tf.Tensor,
                  model: tf.keras.models.Model) -> Tuple[str, float]:
    """Run inference on an image.
    Params:
        image_file: str
            The path to the image file.
        model: tf.keras.Model
            The model to use for inference.
        image_dims: Tuple[int, int]
            The image dimensions (w, h).
    Returns:
        class_name: str
            The class name.
        class_probability: float
            The class probability.
    """
    outputs = model(image, training=False)
    outputs = tf.squeeze(outputs, axis=0)
    label_index = tf.math.argmax(outputs, axis=0)
    probability = outputs[label_index]
    return label_index, probability


@tf.function
def main(image_path: str,
         model: tf.keras.models.Model,
         image_dims: Tuple[int, int],
         labels: list) -> Tuple[str, float]:
    """Run inference on an image.
    Params:
        image_file: str
            The path to the image file.
        model: tf.keras.Model
            The model to use for inference.
        image_dims: Tuple[int, int]
            The image dimensions (w, h).
        labels: list
            The list of labels.
    Returns:
        class_name: str
            The class name.
        class_index: int
            The class index.
        class_probability: float
            The class probability.
    """
    image, original_shape = parse_image(image_path, image_dims)
    class_index, class_probability = run_inference(image, model, image_dims)
    class_name = labels[class_index]
    return class_name, class_index, class_probability


def run_tests(dataset_dir: str,
              model: tf.keras.models.Model,
              image_dims: Tuple[int, int],
              labels: list) -> float:
    """Run inference on a list of images.
    Params:
        dataset_dir: str
            The path to the dataset.
        model: tf.keras.Model
            The model to use for inference.
        image_dims: Tuple[int, int]
            The image dimensions (w, h).
        labels: list
            The list of labels.
    Returns:
        prediction_list: list
            The list of predictions.
        labels_list: list
            The list of labels.
        accuracy: float
            The accuracy of the model.
    """
    import os
    from src import dataset
    from src import training_utils

    labels_list = training_utils.read_files(
        os.path.join(dataset_dir, "labels.txt"))

    _, val_ds, _ = dataset.Dataset(
        image_dims=image_dims,
        dataset_dir=dataset_dir,
        batch_size=4,
        augment=False,
        label_list=labels_list,
        train_val_split=0.1)()
    
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    for image, label in val_ds:
        prediction = model(image, training=False)
        accuracy_metric.update_state(y_true=label, y_pred=prediction)
    
    return accuracy_metric.result().numpy()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True)
    dir_group = parser.add_argument_group(
        "test-directory")
    dir_group.add_argument(
        "--dataset_dir", 
        type=str, 
        default=False) 
    sample_group = parser.add_argument_group(
        "single-image")
    sample_group.add_argument(
        "--image_path",
        type=str,
        required=False)
    parser.add_argument(
        "--image_dims", 
        type=tuple,
        default=(256, 256))
    parser.add_argument(
        "--labels_path", 
        type=str, 
        default="data/images/labels.txt")
    args = parser.parse_args()
    
    if args.dataset_dir == False:
        pass
    elif args.image_path == False:
        pass
    elif args.dataset_dir == True & args.image_path == False:
        model = tf.keras.models.load_model(args.model_path)
        labels = training_utils.read_files(args.labels_path)

        validation_acc = run_tests(
            dataset_dir=args.dataset_dir,
            model=model,
            image_dims=args.image_dims,
            labels=labels)
        
        print(f"Validation accuracy: {validation_acc}")
    elif args.dataset_dir == False & args.image_path == True:
        model = tf.keras.models.load_model(args.model_path)
        labels = training_utils.read_files(args.labels_path)

        class_name, class_index, class_probability = main(
            image_path=args.image_path,
            model=model,
            image_dims=args.image_dims,
            labels=labels)
        
        print(f"Class name: {class_name}")
        print(f"Class index: {class_index}")
        print(f"Class probability: {class_probability}")
    else:
        raise ValueError(
            "Invalid arguments. Can not test single image and dataset at the same time.")
