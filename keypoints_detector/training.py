import matplotlib.pyplot as plt
import time
from .data.generator import image_keypoints_generator
from .networks.basic_models import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import collections


class DatasetCfg(collections.namedtuple('DatasetCfg',
                                        ('img_dirpath', 'keypts_dirpath'))):
    pass


class Train:
    def __init__(self, checkpoints_path="./",
                 train_dataset: DatasetCfg = None, val_dataset: DatasetCfg = None,
                 nClasses=15, output_shape=(96, 96), augmentation_name='all'):
        self.n_classes = nClasses
        self.output_shape = output_shape
        self.checkpoints_path = checkpoints_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.augmentation_name = augmentation_name

    def init_train(self, epochs=30, batch_size=32, initial_epoch=5, gen_use_multiprocessing=False, log_dir=""):
        model = build_model(self.n_classes, input_shape=self.input_shape)
        train_gen, steps_per_epoch = image_keypoints_generator(
            self.train_dataset.img_dirpath,
            self.train_dataset.keypts_dirpath,
            batch_size,
            self.output_height,
            self.output_width,
            do_augment=True,
            augmentation_name=self.augmentation_name)
        valid_gen, val_steps_per_epoch = image_keypoints_generator(
            self.val_dataset.img_dirpath,
            self.val_dataset.keypts_dirpath,
            batch_size,
            self.output_height,
            self.output_width,
            do_augment=True,
            augmentation_name=self.augmentation_name)

        callbacks = [
            EarlyStopping(min_delta=0.001, patience=5),
            TensorBoard(log_dir=log_dir)
        ]
        if self.checkpoints_path is not None:
            default_callback = ModelCheckpoint(
                filepath=self.checkpoints_path + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            )

            callbacks += [
                default_callback
            ]

        model.fit(train_gen,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=valid_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=epochs, callbacks=callbacks,
                  use_multiprocessing=gen_use_multiprocessing, initial_epoch=initial_epoch)


class TrainCustomIter:
    """ Train class where with custom keras iterator
    Custom iteration with invoking fit method with epoch 1, use mainly for experimetation.
    """
    def __init__(self, checkpoints_path="./",
                 train_dataset: DatasetCfg = None, val_dataset: DatasetCfg = None,
                 nClasses=15, output_shape=(96, 96), ):
        self.n_classes = nClasses
        self.output_shape = output_shape
        self.checkpoints_path = checkpoints_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self, epochs=30, batch_size=32, initial_epoch=5, log_dir=""):
        model = None
        history = {"loss": [], "val_loss": []}
        for iepoch in range(epochs):
            start = time.time()

            x_batch, y_batch, w_batch = (None, None, None)
            xval_batch, yval_batch, wbatch_val = (None, None, None)

            hist = model.fit(x_batch, y_batch,
                             sample_weight=w_batch,
                             validation_data=(xval_batch, yval_batch, wbatch_val),
                             batch_size=batch_size,
                             epochs=1,
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


def _main():
    import argparse
    parser = argparse.ArgumentParser("Training script for predicting Landmark using FCN network")
    parser.add_argument('--epochs', default=30, type=int,
                        help="Epochs size")
    parser.add_argument('--n_classes', default=15, type=int,
                        help="No of classes used for training")
    parser.add_argument('--checkpoints_path', default="./", type=str,
                        help="Model path")
    parser.add_argument('--data_dir', default="./", type=str,
                        help="Training data location")

    args = parser.parse_args()
    


if __name__ == "__main__":
    _main()
