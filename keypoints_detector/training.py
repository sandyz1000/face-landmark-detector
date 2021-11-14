import matplotlib.pyplot as plt
import time
import sys
import six
import numpy as np
import typing as t
from pathlib import Path
from keypoints_detector.data.generator import image_keypoints_generator
# from keypoints_detector.data.tfds import get_train_dataset
from keypoints_detector.networks.basic_models import build_model, LANDMARKS_MODELS
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from tensorflow.keras.losses import categorical_crossentropy
import collections
import logging
import glob
import os
import json
logger = logging.getLogger(__name__)


class DatasetCfg(collections.namedtuple('DatasetCfg',
                                        ('img_dirpath', 'keypts_dirpath'))):
    pass


def masked_categorical_crossentropy(gt, pr):
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [
        ff.replace(".index", "") for ff in all_checkpoint_files
    ]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint


def find_weight(y_tra):
    """
    :param y_tra: np.array of shape (N_image, height, width, N_landmark)
    :type y_tra: numpy.ndarray
    :return: weights ->
        np.array of shape (N_image, height, width, N_landmark)
        weights[i_image, :, :, i_landmark] = 1 
                        if the (x,y) coordinate of the landmark for this image is recorded.
        else  weights[i_image, :, :, i_landmark] = 0
    :rtype: [type]
    """
    weight = np.zeros_like(y_tra)
    count0, count1 = 0, 0
    for irow in range(y_tra.shape[0]):
        for ifeat in range(y_tra.shape[-1]):
            if np.all(y_tra[irow, :, :, ifeat] == 0):
                value = 0
                count0 += 1
            else:
                value = 1
                count1 += 1
            weight[irow, :, :, ifeat] = value
    logger.info("N landmarks={:5.0f}, N missing landmarks={:5.0f}, weight.shape={}".
                format(count0, count1, weight.shape))
    return weight


class Train:

    def __init__(
        self,
        net="default",
        checkpoints_path: str = "./weights",
        log_dir: str = "logs",
        train_dataset: DatasetCfg = None,
        valid_dataset: DatasetCfg = None,
        n_classes: int = 68,
        channels=3,
        input_height: int = None,
        input_width: int = None,
        ignore_zero_class=False,
        augmentation_name: str = 'non_geometric',
    ):
        self.log_dir = log_dir
        self.checkpoints_path = checkpoints_path
        self.train_dataset = train_dataset
        self.val_dataset = valid_dataset
        self.augmentation_name = augmentation_name
        self.ignore_zero_class = ignore_zero_class
        self.optimizer_name = 'adam'
        assert net in LANDMARKS_MODELS, "Invalid networks options"
        
        if isinstance(net, six.string_types):
            # create the model from the name
            assert (n_classes is not None), "Please provide the n_classes"
            if (input_height is not None) and (input_width is not None):
                self.model = LANDMARKS_MODELS[net](
                    n_classes, input_height=input_height, input_width=input_width)
            else:
                self.model = LANDMARKS_MODELS[net](n_classes)

        self.n_classes = self.model.n_classes
        self.input_height = self.model.input_height
        self.input_width = self.model.input_width
        self.output_height = self.model.output_height
        self.output_width = self.model.output_width

        self.model_name = net

    def init_train(
        self,
        epochs: int = 30,
        batch_size: int = 32,
        initial_epoch: int = 5,
        gen_use_multiprocessing: bool = False,
        load_weights=None,
        validate=False,
        auto_resume_checkpoint=True,
    ):
        train_gen = image_keypoints_generator(
            self.train_dataset.img_dirpath,
            self.train_dataset.keypts_dirpath,
            self.n_classes,
            batch_size=batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            output_height=self.output_height,
            output_width=self.output_width,
            do_augment=True,
            augmentation_name=self.augmentation_name
        )
        valid_gen = image_keypoints_generator(
            self.val_dataset.img_dirpath,
            self.val_dataset.keypts_dirpath,
            self.n_classes,
            batch_size=batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            output_height=self.output_height,
            output_width=self.output_width,
            do_augment=True,
            augmentation_name=self.augmentation_name
        )

        if self.ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        self.model.compile(loss=loss_k,
                           optimizer=self.optimizer_name,
                           metrics=['accuracy'])

        if self.checkpoints_path is not None:
            config_file = self.checkpoints_path + "_config.json"
            dir_name = os.path.dirname(config_file)

            if (not os.path.exists(dir_name)) and len(dir_name) > 0:
                os.makedirs(dir_name)

            with open(config_file, "w") as f:
                json.dump({
                    "model_class": self.model_name,
                    "n_classes": self.n_classes,
                    "output_height": self.output_height,
                    "output_width": self.output_width
                }, f)

        if load_weights is not None and len(load_weights) > 0:
            print("Loading weights from ", load_weights)
            self.model.load_weights(load_weights)

        initial_epoch = 0

        if auto_resume_checkpoint and (self.checkpoints_path is not None):
            latest_checkpoint = find_latest_checkpoint(self.checkpoints_path)
            if latest_checkpoint is not None:
                print("Loading the weights from latest checkpoint ", latest_checkpoint)
                self.model.load_weights(latest_checkpoint)

                initial_epoch = int(latest_checkpoint.split('.')[-1])

        callbacks = []

        if self.checkpoints_path is not None:
            callbacks.append(ModelCheckpoint(
                filepath=self.checkpoints_path + ".{epoch:05d}",
                save_weights_only=True, verbose=True
            ))

        if not validate:
            self.model.fit(
                train_gen, steps_per_epoch=train_gen.steps_per_epoch,
                epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch
            )
        else:
            self.model.fit(
                train_gen,
                steps_per_epoch=train_gen.steps_per_epoch,
                validation_data=valid_gen,
                validation_steps=valid_gen.steps_per_epoch,
                epochs=epochs, callbacks=callbacks,
                use_multiprocessing=gen_use_multiprocessing, initial_epoch=initial_epoch
            )


class TrainCustomIter:
    """
    NOTE: Deprecated
    Train class where with custom keras iterator
    Custom iteration with invoking fit method with epoch 1, use mainly for experimetation.
    """

    def __init__(
        self,
        checkpoints_path="./weights",
        train_dataset: DatasetCfg = None,
        val_dataset: DatasetCfg = None,
        nClasses: int = 15,
        img_dim: t.Tuple[int] = (96, 96),
    ):
        self.n_classes = nClasses
        self.img_dim = img_dim
        self.checkpoints_path = checkpoints_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(
        self,
        epochs: int = 30,
        batch_size: int = 32, initial_epoch: int = 5, log_dir: int = "logs"
    ):
        model = build_model(self.n_classes, input_height=self.img_dim[0], input_width=self.img_dim[1])
        history = {"loss": [], "val_loss": []}
        for iepoch in range(epochs):
            start = time.time()
            # TODO: Include image generator here
            x_batch, y_batch, w_batch = (None, None, None)
            xval_batch, yval_batch, wbatch_val = (None, None, None)

            hist = model.fit(x_batch, y_batch,
                             sample_weight=w_batch,
                             validation_data=(xval_batch, yval_batch, wbatch_val),
                             batch_size=batch_size,
                             epochs=1,
                             initial_epoch=initial_epoch,
                             verbose=0)
            history["loss"].append(hist.history["loss"][0])
            history["val_loss"].append(hist.history["val_loss"][0])
            end = time.time()
            print("Epoch {:03}: loss {:6.4f} val_loss {:6.4f} {:4.1f}sec".format(
                iepoch + 1, history["loss"][-1], history["val_loss"][-1], end - start))
        model.save(self.checkpoints_path)
        self.training_plt(history)

    def training_plt(self, history) -> None:
        for label in ["val_loss", "loss"]:
            plt.plot(history[label], label=label)
        plt.legend()
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser("Training script for predicting Landmark using FCN network")
    parser.add_argument('--epochs', default=30, type=int,
                        help="Epochs size")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="Batch size")
    parser.add_argument('--n_classes', default=15, type=int,
                        help="No of classes used for training")
    parser.add_argument('--checkpoints_path', default="./", type=str,
                        help="Model path")
    parser.add_argument('--data_dir', default="./", type=str,
                        help="Training data location")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    train_dataset = DatasetCfg(img_dirpath=data_dir.join("train"), keypts_dirpath=data_dir.join("train"))
    val_dataset = DatasetCfg(img_dirpath=data_dir.join("validate"), keypts_dirpath=data_dir.join("validate"))
    trainer = Train(args.checkpoints_path, train_dataset, val_dataset, n_classes=args.n_classes, )
    trainer.init_train(epochs=args.epochs, batch_size=args.batch_size)
