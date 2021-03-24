import cv2
import random
import itertools
import numpy as np
import re
import os
import six
import typing
import tensorflow as tf
from .image_aug import augment_keypoints
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from .config import IMAGE_ORDERING
ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
ACCEPTABLE_KEYPOINTS_FORMATS = [".pts"]


class DataLoaderError(Exception):
    pass


def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first', read_image_type=1):
    """ Load image array from input
    """
    if isinstance(image_input, np.ndarray):
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)
        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_image_list_from_path(images_path: typing.List):
    image_files = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            # file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append(os.path.join(images_path, dir_entry))
    return image_files


def get_pairs_from_paths(images_path: typing.List[str], keypts_path: typing.List[str],
                         ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the keypoints from the keypts_path directory while checking integrity of data.
    """

    image_files = []
    keypoints_files = {}

    _splt_file_ext = (lambda dir_entry: os.path.splitext(dir_entry))
    image_files = [(*_splt_file_ext(dir_entry), os.path.join(images_path, dir_entry))
                   for dir_entry in os.listdir(images_path)
                   if os.path.isfile(os.path.join(images_path, dir_entry)) and
                   os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS]

    for dir_entry in os.listdir(keypts_path):
        if os.path.isfile(os.path.join(keypts_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_KEYPOINTS_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(keypts_path, dir_entry)
            if file_name in keypoints_files:
                raise DataLoaderError("Segmentation file with filename {0} already exists and is ambiguous to"
                                      " resolve with path {1}. Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            keypoints_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in keypoints_files:
            return_value.append((image_full_path, keypoints_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            raise DataLoaderError("No corresponding segmentation found for image {0}.".format(image_full_path))

    return return_value


def keypoints_to_heatmap(kpsoi: KeypointsOnImage):
    """
    We also actually get one distance map per keypoint of same height and width as the image. 
    The maps are not normalized and hence can exceed the value range [0.0, 1.0]. 
    Let's normalize them now based on the maximum possible euclidean distance:
    """
    distance_maps = kpsoi.to_distance_maps()
    height, width = kpsoi.shape[0:2]
    max_distance = np.linalg.norm(np.float32([height, width]))
    distance_maps_normalized = distance_maps / max_distance
    # print("min:", distance_maps.min(), "max:", distance_maps_normalized.max())

    heatmaps = HeatmapsOnImage(distance_maps_normalized, shape=kpsoi.shape)
    return heatmaps


def read_keypoints(keypts_path):
    """Read keypoints from path, given the format
    - First line is version
    - Second line n_points
    - Split each line by space
    """
    with open(keypts_path, 'r') as fp:
        keypoints = []
        for line in fp.readline():
            _text = line.strip()
            if re.match(r'{|}', _text):
                continue
            if re.match('version', _text):
                version = _text
            elif re.match('n_points', _text):
                n_points = _text
            else:
                kps = _text.split()
                keypoints.append(Keypoint(*kps))
        return keypoints, n_points, version


def image_keypoints_generator(images_path, segs_path, batch_size,
                              n_classes, output_height, output_width,
                              do_augment=False,
                              augmentation_name="aug_all",
                              preprocessing=None,
                              read_image_type=cv2.IMREAD_COLOR,
                              ignore_keypts=False):

    if not ignore_keypts:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        random.shuffle(img_seg_pairs)
        zipped = itertools.cycle(img_seg_pairs)
    else:
        img_list = get_image_list_from_path(images_path)
        random.shuffle(img_list)
        img_list_gen = itertools.cycle(img_list)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            if ignore_keypts:
                im = next(img_list_gen)
                keypoints = None
            else:
                im, keypoints_path = next(zipped)
                keypoints, n_points, version = read_keypoints(keypoints_path)
                assert n_points == n_classes, "No of Keypoint not equivalent to model configurations"

            im = cv2.imread(im, read_image_type)

            if do_augment:
                assert not ignore_keypts, "Not supported yet"
                im, keypoints = augment_keypoints(im, keypoints, augmentation_name)

            if preprocessing is not None:
                im = preprocessing(im)

            X.append(get_image_array(im, output_width, output_height, ordering=IMAGE_ORDERING))

            if not ignore_keypts:
                Y.append(keypoints_to_heatmap(keypoints))

        if ignore_keypts:
            yield np.array(X)
        else:
            yield np.array(X), np.array(Y)


def get_train_dataset(images_path, segs_path, batch_size,
                      n_classes, input_height, input_width,
                      output_height, output_width,
                      do_augment=False,
                      augmentation_name="aug_all",
                      preprocessing=None,
                      read_image_type=cv2.IMREAD_COLOR):
    """
    Function to convert python iterator to tensorflow dataset
    """
    image_gen = image_keypoints_generator(images_path, segs_path, batch_size,
                                          n_classes, output_height, output_width,
                                          do_augment=do_augment,
                                          augmentation_name=augmentation_name,
                                          preprocessing=preprocessing,
                                          read_image_type=read_image_type,
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
