# SimpleShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Creates the modified EfficiendNet-BX architecture."""
# ---------------------------------------------------------------------------

import tensorflow as tf

from typing import Tuple


def get_model(model_name: str = "efficientnet_b4",
              input_shape: Tuple[int, int] = (256, 256),
              num_classes: int = 20,
              weights: str = "imagenet",
              include_top: bool = False,
              fine_tune: bool = False,
              **kwargs) -> tf.keras.models.Model:
    """Returns a model based on the EfficientNet-BX architecture.
    Params:
        model_name: str
            The name of the model.
        input_shape : Tuple[int, int]
            The input shape of the model.
        num_classes : int
            The number of classes in the dataset.
        weights : str
            The weights to use for the model.
        include_top : bool
            Whether to include the top of the model or not.
        pooling : str
            The pooling layer to use.
        kwargs : dict
            Additional keyword arguments.
    Returns:
        tf.keras.models.Model
            The model.
    """
    models = {
        "efficientnet_b0": tf.keras.applications.EfficientNetB0,
        "efficientnet_b1": tf.keras.applications.EfficientNetB1,
        "efficientnet_b2": tf.keras.applications.EfficientNetB2,
        "efficientnet_b3": tf.keras.applications.EfficientNetB3,
        "efficientnet_b4": tf.keras.applications.EfficientNetB4,
        "efficientnet_b5": tf.keras.applications.EfficientNetB5,
        "efficientnet_b6": tf.keras.applications.EfficientNetB6,
        "efficientnet_b7": tf.keras.applications.EfficientNetB7}
    backbone = models[model_name](weights=weights,
                               include_top=include_top,
                               **kwargs)

    inputs = tf.keras.layers.Input(shape=(*input_shape, 3))
    if fine_tune == True:
        backbone.trainable = False
    else:
        backbone.trainable = True
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="avg_pool")(x)
    x = tf.keras.layers.Dropout(
        0.5, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions")(x)
    return tf.keras.models.Model(
        inputs=inputs, 
        outputs=outputs, 
        name=model_name + "_supergrocer")