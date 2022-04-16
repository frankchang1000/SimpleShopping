# SimpleShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Util training functions for Supervised and Semi-Supervised Learning."""
# ---------------------------------------------------------------------------

import os
import tensorflow as tf

from . import model
from . import training_utils as utils

from typing import List

def train_mpl(
    l_dataset: tf.data.Dataset,
    u_dataset: tf.data.Dataset,
    models: List[tf.keras.models.Model], 
    optimizers: List[tf.keras.optimizers.Optimizer],
    hyperparams: dict,
    training_dir: str,
    resume_training: bool = False):
    """Trains the model with MPL.
    Params:
        l_dataset: tf.data.Dataset
            The labeled dataset.
        u_dataset: tf.data.Dataset
            The unlabeled datasets.
        models: List[tf.keras.models.Model]
            The models to train.
        optimizers: List[tf.keras.optimizers.Optimizer]
            The optimizers to use for training.
        hyperparams: dict
            The hyperparameters to use for training.
        resume_training: bool
            Whether to resume training or not.
    Returns:
        tf.keras.models.Model
            The trained model.  
    """
    # Define training dirs
    teacher_checkpoint_dir = os.path.join(training_dir, "teacher_checkpoints")
    student_checkpoint_dir = os.path.join(training_dir, "student_checkpoints")
    ema_checkpoint_dir = os.path.join(training_dir, "ema_checkpoints")
    # Create the training models
    teacher_model, student_model, ema_model = models[0], models[1], models[2]
    # Define the optimizers
    teacher_optimizer, student_optimizer = optimizers[0], optimizers[1]
    # Checkpoints
    teacher_checkpoint = tf.train.Checkpoint(
        optimizer=teacher_optimizer, 
        model=teacher_model)
    teacher_checkpoint_manager = tf.train.CheckpointManager(
        teacher_checkpoint, 
        teacher_checkpoint_dir, 
        3)
    student_checkpoint = tf.train.Checkpoint(
        optimizer=student_optimizer, 
        model=student_model)
    student_checkpoint_manager = tf.train.CheckpointManager(
        student_checkpoint, 
        student_checkpoint_dir, 
        3)
    ema_checkpoint = tf.train.Checkpoint(
        model=ema_model)
    ema_checkpoint_manager = tf.train.CheckpointManager(
        ema_checkpoint, 
        ema_checkpoint_dir, 
        3)    
    # Training steps
    best_val_loss = 0
    step = tf.Variable(0, dtype=tf.int64)
    continue_step = 0
    total_steps = hyperparams["total_steps"]
    # Restore models if conitnue to train
    if hyperparams[resume_training]:
        print("Loading models from checkpoints.")
        teacher_checkpoint.restore(
            teacher_checkpoint_manager.latest_checkpoint
        ).expect_partial()
        student_checkpoint.restore(
            student_checkpoint_manager.latest_checkpoint
        ).expect_partial()
        ema_checkpoint.restore(
            ema_checkpoint_manager.latest_checkpoint
        ).expect_partial()
        continue_step = student_checkpoint.save_counter * hyperparams["num_steps"]
    # Define the training metrics
    training_mean_metrics = {
        "mpl_uda/u-ratio": tf.keras.metrics.Mean(),
        "mpl_uda/l-ratio": tf.keras.metrics.Mean(),
        "mpl/dot-product": tf.keras.metrics.Mean(),
        "mpl/moving-dot-product": tf.keras.metrics.Mean(),
        "mpl_cross_entropy/teacher-on-l": tf.keras.metrics.Mean(),
        "mpl_cross_entropy/teacher-on-u": tf.keras.metrics.Mean(),
        "mpl_cross_entropy/student-on-u": tf.keras.metrics.Mean(),
        "mpl_cross_entropy/student-on-l": tf.keras.metrics.Mean()}
    val_student_loss_metric = tf.keras.metrics.Mean()
    val_student_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name="val_acc_student")
    val_ema_loss_metric = tf.keras.metrics.Mean()
    val_ema_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name="val_acc_ema")
    # Define the training losses
    mpl_loss = tf.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
    student_unlabeled_loss = tf.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
        label_smoothing=hyperparams["mpl_label_smoothing"])
    student_labeled_loss = tf.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    uda_cross_entropy = utils.uda_cross_entropy_fn(())

    @tf.function
    def train_step(
        step_tensor, 
        labeled_images, label_indices, org_images, aug_images, uda_weight):
        """Train one step on the data."""
        all_images = tf.concat([labeled_images, org_images, aug_images], 0)

        with tf.GradientTape() as ttape:
            all_logits = teacher_model(all_images, training=True)
            logits, labels, masks, cross_entropy = uda_cross_entropy(
                all_logits, 
                label_indices, 
                step_tensor)

        with tf.GradientTape() as stape:
            u_aug_and_l_images = tf.concat([aug_images, labeled_images], 0)
            logits["s_on_u_aug_and_l"] = student_model(
                u_aug_and_l_images, training=True)

            logits["s_on_u"], logits["s_on_l_old"] = tf.split(
                logits["s_on_u_aug_and_l"],
                [aug_images.shape[0], labeled_images.shape[0]],
                0)

            # for backprop
            cross_entropy["s_on_u"] = student_unlabeled_loss(
                y_true=tf.nn.softmax(logits["u_aug"], -1),
                y_pred=logits["s_on_u"])
            cross_entropy["s_on_u"] = tf.reduce_sum(
                cross_entropy["s_on_u"]) / float(hyperparams["unlabeled_batch_size"])

            # for Taylor
            cross_entropy["s_on_l_old"] = student_labeled_loss(
                y_true=labels["l"], y_pred=logits["s_on_l_old"])
            cross_entropy["s_on_l_old"] = tf.reduce_sum(
                cross_entropy["s_on_l_old"]) / float(hyperparams["mpl_batch_size"])

            if hyperparams["precision"]:
                cross_entropy["s_on_u"] = student_optimizer.get_scaled_loss(
                    cross_entropy["s_on_u"])
                # We do not run on s_on_l_old, because it will be scaled later
                # cross_entropy["s_on_l_old"] = student_optimizer.get_scaled_loss(
                #    cross_entropy["s_on_l_old"])

        student_grad_unlabeled = stape.gradient(
            cross_entropy["s_on_u"], student_model.trainable_variables)
        student_grad_unlabeled, _ = tf.clip_by_global_norm(
            student_grad_unlabeled, hyperparams["mpl_optimizer_grad_bound"])

        if hyperparams["precision"]:
            student_grad_unlabeled = student_optimizer.get_unscaled_gradients(
                student_grad_unlabeled)

        student_optimizer.apply_gradients(
            zip(student_grad_unlabeled, student_model.trainable_variables))

        logits["s_on_l_new"] = student_model(labeled_images)
        cross_entropy["s_on_l_new"] = student_labeled_loss(
            y_true=labels["l"], y_pred=logits["s_on_l_new"])
        cross_entropy["s_on_l_new"] = tf.reduce_sum(
            cross_entropy["s_on_l_new"]) / float(hyperparams["mpl_batch_size"])

        dot_product = cross_entropy["s_on_l_new"] - cross_entropy["s_on_l_old"]
        # limit = 3.0**(0.5)
        moving_dot_product = tf.compat.v1.get_variable(
            "moving_dot_product",
            trainable=False,
            shape=dot_product.shape)

        moving_dot_product = (0.01 * (moving_dot_product - dot_product)) - moving_dot_product
        dot_product = dot_product - moving_dot_product
        dot_product = tf.stop_gradient(dot_product)

        with ttape:
            cross_entropy["mpl"] = mpl_loss(
                y_true=tf.stop_gradient(tf.nn.softmax(logits["u_aug"], -1)),
                y_pred=logits["u_aug"],
            )
            cross_entropy["mpl"] = tf.reduce_sum(cross_entropy["mpl"]) / float(
                hyperparams["unlabeled_batch_size"]
            )

            # teacher train op
            teacher_loss = (
                cross_entropy["u"] * uda_weight
                + cross_entropy["l"]
                + cross_entropy["mpl"] * dot_product
            )
            # Scale the the teacher loss.
            if hyperparams["precision"]:
                teacher_loss = teacher_optimizer.get_scaled_loss(teacher_loss)

        teacher_grad = ttape.gradient(
            teacher_loss, teacher_model.trainable_variables
        )
        teacher_grad, _ = tf.clip_by_global_norm(
            teacher_grad, hyperparams["mpl_optimizer_grad_bound"]
        )

        # Unscale the gradients
        if hyperparams["precision"]:
            teacher_grad = teacher_optimizer.get_unscaled_gradients(teacher_grad)

        teacher_optimizer.apply_gradients(
            zip(teacher_grad, teacher_model.trainable_variables)
        )

        tf.print(
            "Step: ", step_tensor,
            "mpl-uda/u-ratio", tf.reduce_mean(masks["u"]),
            "mpl-uda/l-ratio", tf.reduce_mean(masks["l"]),
            "mpl/dot-product", dot_product,
            "mpl/moving-dot-product", moving_dot_product,
            "mpl-cross-entropy/teacher-on-l", cross_entropy["l"],
            "mpl-cross-entropy/teacher-on-u", cross_entropy["u"],
            "mpl-cross-entropy/student-on-u", cross_entropy["s_on_u"],
            "mpl-cross-entropy/student-on-l", cross_entropy["s_on_l_new"],
            )

        return {
            "mpl-uda/u-ratio": tf.reduce_mean(masks["u"]),
            "mpl-uda/l-ratio": tf.reduce_mean(masks["l"]),
            "mpl/dot-product": dot_product,
            "mpl/moving-dot-product": moving_dot_product,
            "mpl-cross-entropy/teacher-on-l": cross_entropy["l"],
            "mpl-cross-entropy/teacher-on-u": cross_entropy["u"],
            "mpl-cross-entropy/student-on-u": cross_entropy["s_on_u"],
            "mpl-cross-entropy/student-on-l": cross_entropy["s_on_l_new"],
        }

    @tf.function
    def val_step(images, labels):
        ema_predictions = tf.nn.softmax(ema_model(images, training=False), -1)
        student_predictions = tf.nn.softmax(
            student_model(images, training=False), -1
        )

        def update_metrics(predictions, val_loss_metric, val_acc_metric):
            loss = student_labeled_loss(
                y_true=labels,
                y_pred=predictions,
            )
            val_loss_metric(tf.reduce_sum(loss) / float(images.shape[0]))
            val_acc_metric.update_state(labels, predictions)

        update_metrics(
            ema_predictions, val_ema_loss_metric, val_ema_acc_metric)
        update_metrics(
            student_predictions, val_student_loss_metric, val_student_acc_metric)

    def update_val_metrics(val_loss_metric, val_acc_metric):
        val_loss = val_loss_metric.result()
        val_acc = val_acc_metric.result()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()
        return val_loss, val_acc        

    for epoch in range(hyperparams["training_epochs"]):
        print("<---------- Epoch: " + str(epoch) + " ---------->")
        for (l_image, l_label) in (l_dataset):
            u_image, u_augim = next(iter(u_dataset))
            step = step + 1
            student_optimizer.learning_rate.assign(
                utils.learning_rate(
                    step,
                    hyperparams["student_learning_rate"],
                    total_steps,
                    hyperparams["student_learning_rate_warmup"],
                    hyperparams["student_learning_rate_numwait"]))            
            teacher_optimizer.learning_rate.assign(
                utils.learning_rate(
                    step,
                    hyperparams["teacher_learning_rate"],
                    total_steps,
                    hyperparams["teacher_learning_rate_warmup"],
                    hyperparams["teacher_learning_rate_numwait"]))

            uda_weight = hyperparams["uda_weight"] * tf.math.minimum(
                1.0, float(step) / float(hyperparams["uda_steps"]))
            
            losses = train_step(
                step, l_image, l_label, u_image, u_augim, uda_weight)
            averaged_values = {"mpl-uda/weight": uda_weight}

            utils.update_ema_weights(hyperparams, ema_model, student_model, step)
            # Update tensorboard with training data
            utils.update_tensorboard(
                losses = losses, 
                step = step, 
                teacher_optimizer = teacher_optimizer, 
                student_optimizer = student_optimizer)
            
            # Save each checkpoint, only the best
            teacher_checkpoint_manager.save()
            student_checkpoint_manager.save()
            ema_checkpoint_manager.save()

        if (epoch % hyperparams["eval_every_epoch"] == 0 or epoch == hyperparams["mpl_epochs"] - 1):
            try:
                for images, label_indices in l_dataset:
                    val_step(images, label_indices)

                val_ema_loss, val_ema_acc = update_val_metrics(
                    val_ema_loss_metric, val_ema_acc_metric)
                val_student_loss, val_student_acc = update_val_metrics(
                    val_student_loss_metric, val_student_acc_metric)

                tf.summary.scalar(
                    "mpl-val/ema-loss", 
                    data=val_ema_loss, step=step + continue_step)
                tf.summary.scalar(
                    "mpl-val/ema-acc", data=val_ema_acc, step=step + continue_step)
                tf.summary.scalar(
                    "mpl-val/student-loss", data=val_student_loss, step=step + continue_step)
                tf.summary.scalar(
                    "mpl-val/student-acc", data=val_student_acc, step=step + continue_step)
            except: 
                print("Validation could not be run.")
            # if val_ema_loss <= best_val_loss:
            #    best_val_loss = val_ema_loss
            tf.keras.models.save_model(
                student_model, student_exported_dir)
            tf.keras.models.save_model(
                ema_model, ema_exported_dir)