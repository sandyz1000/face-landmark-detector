import os
import sys
from keypoints_detector import prediction, training
import click
from time import time
from functools import wraps


def timing(f):
    @wraps(f)
    def _wrap_func(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        # print(f"{f.__name__} took: {te - ts} sec")
        return result
    return _wrap_func


@click.command()
@click.option('--data_dir', default="./", type=str, help="Training data location")
@click.option("--checkpoints_path", type=str, default="./weights", help="Keypoints model path")
@click.option('--net', default='default', type=str, help="Default network", choices=training.LANDMARKS_MODELS.keys())
@click.option('--epochs', default=30, type=int, help="Epochs size")
@click.option('--n_classes', default=68, type=int, help="No of classes used for training")
@click.option('--batch_size', default=32, type=int, help="Default batch size used in wach step")
@click.option('--augmentation_name', default='all', type=str,
              help="Augmentation to used while training, Supported" +
              "type: all, custom, geometric, non_geometric")
@timing
def train(data_dir, checkpoints_path, net, epochs=30, n_classes=68, batch_size=32,
          augmentation_name='all', log_dir="logs"):
    train_data_dir, valid_data_dir = [os.path.join(data_dir, x) for x in ('train', 'valid')]
    train_dataset = training.DatasetCfg(img_dirpath=train_data_dir, keypts_dirpath=train_data_dir)
    valid_dataset = training.DatasetCfg(img_dirpath=valid_data_dir, keypts_dirpath=valid_data_dir)
    instance = training.Train(
        checkpoints_path=checkpoints_path, train_dataset=train_dataset,
        valid_dataset=valid_dataset, n_classes=n_classes, augmentation_name=augmentation_name
    )
    instance.init_train(net=net, epochs=epochs, batch_size=batch_size, log_dir=log_dir)


@click.command()
@click.option("--checkpoints_path", type=str, default=None, help="Keypoints model path")
@click.option('--inp', required=True, type=str, help="Training data location")
@click.option('--net', default='default', type=str, help="Default network", choices=training.LANDMARKS_MODELS.keys())
@timing
def predict(checkpoints_path, inp, net, ):
    return prediction.keypts_predict(inp=inp, checkpoints_path=checkpoints_path)


@click.group()
def main():
    return 0


main.add_command(train, "train")
main.add_command(predict, "predict")

if __name__ == "__main__":
    sys.exit(main())
