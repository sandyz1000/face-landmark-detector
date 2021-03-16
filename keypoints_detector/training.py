import matplotlib.pyplot as plt
import time
import sys
from .data.generator import get_train_dataset
from .networks.basic_models import build_model
from typing import NamedTuple


class DatasetCfg(NamedTuple):
    img_dirpath: str
    annot_dirpath: str


class Train:
    def __init__(self, model_path="./",
                 train_dataset: DatasetCfg = None, val_dataset: DatasetCfg = None,
                 nClasses=15, const=10, input_shape=(96, 96)):
        self.n_classes = nClasses
        self.input_shape = input_shape
        self.model_path = model_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def init_train(self, epochs=30, batch_size=32,):
        fcnmodel = build_model(self.n_classes, input_shape=self.input_shape)
        train_gen = get_train_dataset(self.train_dataset.img_dirpath, self.train_dataset.annot_dirpath,
                                      self.input_shape, batch_size=batch_size, is_train=True)
        valid_gen = get_train_dataset(self.val_dataset.img_dirpath, self.val_dataset.annot_dirpath,
                                      self.input_shape, batch_size=batch_size, is_train=False)

        history = {"loss": [], "val_loss": []}
        for iepoch in range(epochs):
            start = time.time()

            x_batch, y_batch, w_batch = (None, None, None)
            xval_batch, yval_batch, wbatch_val = (None, None, None)

            hist = fcnmodel.fit(x_batch, y_batch,
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
        fcnmodel.save(model_path)
        self.training_plt(history)

    def training_plt(self, history) -> None:
        for label in ["val_loss", "loss"]:
            plt.plot(history[label], label=label)
        plt.legend()
        plt.show()

    def prediction(fcnmodel, X_train, out_shape=(96, 96), nClasses=15):
        y_pred = fcnmodel.predict(X_train)
        y_pred = y_pred.reshape(-1, out_shape[0], out_shape[1], nClasses)
        return y_pred

    def ksg_init_train(n_classess, img_shape=(160, 160), **kwargs):
        try:
            import keras_segmentation
            from keras_segmentation.models.unet import vgg_unet
        except ImportError:
            print(">>>> =====Install keras-segmentation module =====>>>")
            sys.exit(1)

        model = vgg_unet(n_classes=n_classess, input_height=img_shape[0], input_width=img_shape[1])

        model.train(
            train_images="dataset1/images_prepped_train/",
            train_annotations="dataset1/annotations_prepped_train/",
            checkpoints_path="/tmp/vgg_unet_1",
            epochs=5
        )

    def ksg_evaluation(model, img_dir, annot_dir):
        return model.evaluate_segmentation(inp_images_dir=img_dir, annotations_dir=annot_dir)


def _main():
    import argparse
    parser = argparse.ArgumentParser("Training script for predicting Landmark using FCN network")
    parser.add_argument('--epochs', default=30, type=int,
                        help="Epochs size")
    parser.add_argument('--n_classes', default=15, type=int,
                        help="No of classes used for training")
    parser.add_argument('--model_path', default="./", type=str,
                        help="Model path")
    parser.add_argument('--data_dir', default="./", type=str,
                        help="Training data location")

    args = vars(parser.parse_args())
    init_train(**args.__dict__)


if __name__ == "__main__":
    _main()
