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
        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        print(f"{f.__name__} took: {te - ts} sec")
        return result
    return _wrap_func


@click.command
@click.option("--checkpoints_path", type=str, default=None, help="Keypoints model path")
@click.option('--data_dir', default="./", type=str, help="Training data location")
@click.option('--epochs', default=30, type=int, help="Epochs size")
@click.option('--n_classes', default=68, type=int, help="No of classes used for training")
@click.option('--augmentation_name', default='all', type=str,
              help="Augmentation to used while training, Supported" +
              "type: all, custom, geometric, non_geometric")
@timing
def train(checkpoints_path, data_dir, epochs=30, n_classes=68, augmentation_name='all'):
    pass


@click.command
@click.option("--checkpoints_path", type=str, default=None, help="Keypoints model path")
@click.option('--data_dir', default="./", type=str, help="Training data location")
@timing
def predict(checkpoints_path, data_dir):
    pass


@click.group()
def main():
    return 0


main.add_command(train, "train")
main.add_command(predict, "predict")

if __name__ == "__main__":
    sys.exit(main())
