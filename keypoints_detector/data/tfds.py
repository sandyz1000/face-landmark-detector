import tensorflow as tf
import cv2
import numpy as np
from .generator import image_keypoints_generator


def get_train_dataset(
    images_path, segs_path, batch_size,
    n_classes, output_height, output_width,
    do_augment=False,
    augmentation_name="aug_all",
    preprocessing=None,
    grayscale=False
) -> tf.data.Dataset:
    """
    Function to convert python iterator to tensorflow dataset
    """
    image_gen = image_keypoints_generator(images_path, segs_path, batch_size,
                                          n_classes, output_height, output_width,
                                          do_augment=do_augment,
                                          augmentation_name=augmentation_name,
                                          preprocessing=preprocessing,
                                          grayscale=grayscale,
                                          ignore_keypts=False)

    steps_per_epoch = np.floor(len(images_path) / batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    img_shape = (output_height, output_width, 3)
    train_ds = tf.data.Dataset.from_generator(
        image_gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None, *img_shape], [None, *img_shape])
    ).repeat()

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, steps_per_epoch
