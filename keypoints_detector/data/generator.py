import cv2
import random
import itertools
import numpy as np
import re
import os
import six
import typing
import glob2
import tensorflow as tf
from functools import partial
from PIL import Image
from .image_aug import augment_keypoints
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from .config import IMAGE_ORDERING
from skimage import transform

LANDMARK_ORDER = {"orig": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
                  "new": [1, 0, 4, 5, 2, 3, 8, 9, 6, 7, 12, 11]}
ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
ACCEPTABLE_KEYPOINTS_FORMATS = [".pts"]


class DataLoaderError(Exception):
    pass


def get_image_array(
    image,
    width, height,
    imgNorm="sub_mean",
    ordering='channels_first',
    read_image_type=1,
):
    """ Load image array from input
    """
    if isinstance(image, np.ndarray):
        # It is already an array, use it as it is
        img = image
    elif isinstance(image, six.string_types):
        if not os.path.isfile(image):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image))
        img = cv2.imread(image, read_image_type)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image))))

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


def keypoints_to_heatmap(kpsoi: KeypointsOnImage) -> HeatmapsOnImage:
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
    heatmaps = HeatmapsOnImage((1.0 - distance_maps_normalized)**10, shape=kpsoi.shape)

    return heatmaps


def read_keypoints(keypts_path, is_imgaug_kps=False):
    """Read keypoints from path, given the format
    - First line is version
    - Second line n_points
    - Split each line by space
    """
    with open(keypts_path, 'r') as fp:
        keypoints = []
        for line in fp.readlines():
            _text = line.strip()
            if re.match(r'{|}', _text):
                continue
            if re.match('version', _text):
                version = re.findall(r'\d+', _text)[0]
            elif re.match('n_points', _text):
                n_points = int(re.findall(r'\d+', _text)[0])
            else:
                kps = [float(cord) for cord in _text.split()]
                keypoints.append(Keypoint(*kps) if is_imgaug_kps else kps)

        if not is_imgaug_kps:
            keypoints = np.array(keypoints)
        return keypoints, n_points, version


class transform_images:
    """ Apply transformation to an image
    """
    lm = {"orig": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
          "new": [1, 0, 4, 5, 2, 3, 8, 9, 6, 7, 12, 11]}

    def transform(
        self,
        im: np.ndarray,
        kpts: typing.List[float],
        max_rotation=0.01,
        max_shift=2,
        max_shear=0,
        max_scale=0.01,
        mode="edge",
    ):
        """Affine transformation for a single image
        """
        scale = (
            np.random.uniform(1 - max_scale, 1 + max_scale),
            np.random.uniform(1 - max_scale, 1 + max_scale)
        )
        rotation_tmp = np.random.uniform(-1 * max_rotation, max_rotation)
        translation = (
            np.random.uniform(-1 * max_shift, max_shift),
            np.random.uniform(-1 * max_shift, max_shift)
        )
        
        shear = np.random.uniform(-1 * max_shear, max_shear)
        tform = transform.AffineTransform(
            scale=scale,  # ,
            # Convert angles from degrees to radians.
            rotation=np.deg2rad(rotation_tmp),
            translation=translation,
            shear=np.deg2rad(shear)
        )
        im = transform.warp(im, tform, mode=mode)
        kpts = transform.warp(kpts, tform, mode=mode)

        return im, kpts

    def __call__(
        self,
        im: np.ndarray,
        kpts: typing.List[float],
        wt: np.ndarray = None
    ):
        """
        Invoke this method by default and apply transformation to image and keypoints
        :param im: Nd array of 3 channels. im.shape (height,width,n_channel)
        :type im: numpy.ndarray
        :param kpts: Nd array of k channels. kpts.shape (height,width,n_landmarks)
        :type kpts: numpy.ndarray
        """
        assert len(kpts) == len(self.lm['orig']), \
            ValueError("Length of keypoints is mismatched")

        im, kpts = self.transform(im, kpts)

        im, kpts, wt = self.horiz_flip(im, kpts, wt)
        return im, kpts, wt

    def swap_index(
        self,
        y_batch: typing.List[float],
        lo: np.ndarray, ln: np.ndarray
    ):
        """
        lm = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
            "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}
        lo, ln = np.array(lm["orig"]), np.array(lm["new"])
        """
        y_orig = y_batch[:, :, lo]
        y_batch[:, :, lo] = y_batch[:, :, ln]
        y_batch[:, :, ln] = y_orig
        return y_batch

    def horiz_flip(
        self,
        im: np.ndarray,
        kpts: typing.List[float], wt: np.ndarray = None
    ):
        """
        flip the image with 50% chance

        lm is a dictionary containing "orig" and "new" key
        This must indicate the potitions of heatmaps that need to be flipped
        landmark_order = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
                        "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}


        w is optional and if it is in the code, the position needs to be specified
        with loc_w_batch

        im.shape (height,width,n_channel)
        kpts.shape (height,width,n_landmarks)
        wt.shape (height,width,n_landmarks)
        """

        lo, ln = np.array(self.lm["orig"]), np.array(self.lm["new"])

        # Handle horizontal flip in x & y axis over here
        # im = tf.transpose(im, perm=[1, 0, 2])
        im = im[::-1, ...]
        # im = tf.transpose(im, perm=[0, 1, 2])
        kpts = self.swap_index(kpts, lo, ln)

        # when horizontal flip happens to image, we need to heatmap (y) and weights y and w
        # do this if loc_w_batch is within data length
        if wt:
            wt = self.swap_index(wt, lo, ln)
        return im, kpts, wt


def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  # (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  # (height,1)
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def generate_hm(height, width, keypoints, s=3):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing keypoints
        maxlenght : Lenght of the Bounding Box
    """
    hm = np.zeros((height, width, len(keypoints)), dtype=np.float32)
    for i in range(len(keypoints)):
        if not np.array_equal(keypoints[i], [-1, -1]):
            hm[:, :, i] = gaussian_k(keypoints[i][0], keypoints[i][1], s, height, width)
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm


def plot_img_hm_pair(img: np.ndarray, y_train: np.ndarray, outdir='output'):
    """Helper to visualize image heatmap pair before training

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
    """
    import matplotlib.pyplot as plt
    from tempfile import NamedTemporaryFile

    # fig = plt.figure(figsize=(20, 6))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("input")

    # Show heatmap below in grid
    row, col = 0, 0
    for j in range(y_train.shape[-1]):
        col = j % 4
        if not col:
            row += 1

        ax2 = plt.subplot2grid((2, 2), (row, 0), colspan=col)
        ax2.imshow(y_train[:, :, j], cmap="gray")
        ax2.set_title(f"index-{str(j)}")

    with NamedTemporaryFile(mode='w', prefix='im-hm', dir=outdir) as f:
        plt.savefig(f.name)

    return os.path.join(outdir, f.name)


class custom_image_keypts_generator:
    def __init__(
        self,
        images_path: str,
        keypts_path: str,
        n_classes: str,
        batch_size: int = 64,
        output_dim: typing.Tuple = (256, 256),
        do_augment: bool = False,
        read_image_type: int = cv2.IMREAD_COLOR,
        ignore_keypts: bool = False,
        repeat=2
    ):
        """ Apply custom transformation on image and keypoints
        """
        self.ignore_keypts = ignore_keypts
        self.batch_size = batch_size
        self.read_image_type = read_image_type
        self.do_augment = do_augment
        self.repeat = repeat
        self.xform_imgs = transform_images()

        if not self.ignore_keypts:
            img_list = get_pairs_from_paths(images_path, keypts_path)
            random.shuffle(img_list)
            self.img_list_gen = itertools.cycle(img_list)
        else:
            img_list = get_image_list_from_path(images_path)
            random.shuffle(img_list)
            self.img_list_gen = itertools.cycle(img_list)

        self.steps_per_epoch = np.floor(len(img_list) / batch_size)
        self.output_dim = output_dim
        self.n_classes = n_classes
        self._current_step = 0

    def __iter__(self) -> typing.Iterator:
        return self

    def __next__(self):
        X = []
        Y = []
        if self._current_step > self.steps_per_epoch * self.repeat:
            raise StopIteration("Index out of order")

        for _ in range(self.batch_size):
            if self.ignore_keypts:
                im = next(self.img_list_gen)
                keypoints = None
            else:
                im, keypoints_path = next(self.img_list_gen)
                keypoints, n_points, version = read_keypoints(keypoints_path)
                assert n_points == self.n_classes, "No of Keypoint not equivalent to model configurations"

            im = cv2.imread(im, self.read_image_type)

            # im = get_image_array(im, *self.output_dim, ordering=IMAGE_ORDERING)

            # TODO: Apply random weights to the input image
            im = self.xform_imgs(im, keypoints)
            X.append(im)

            if not self.ignore_keypts:
                heatmaps = generate_hm(*self.output_dim, keypoints, s=3)
                # Heatmaps of shape (H * W * num_keypoints), if you want to draw heatmaps in to image use
                # heatmaps.draw_on_image(im)
                Y.append(heatmaps)

        self._current_step += 1

        if self.ignore_keypts:
            return np.array(X)
        else:
            return np.array(X), np.array(Y)


def letterbox_image(image: np.ndarray, size: typing.Tuple[int, int]) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding
    """
    img_width, img_height, _ = image.shape
    w, h = size
    scale = min(w / img_width, h / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)

    image = Image.fromarray(image).resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return np.array(new_image)


class image_keypoints_generator:

    def __init__(
        self,
        images_path: str,
        keypts_path: str,
        n_classes: str,
        batch_size: int = 8,
        output_dim: typing.Tuple = (416, 416),
        do_augment: bool = False,
        shuffle: bool = True,
        augmentation_name: str = "all",
        preprocessing: typing.Callable = None,
        read_image_type: int = cv2.IMREAD_COLOR,
        ignore_keypts: bool = False,
        repeat=2
    ):
        """ Apply transformation on image and keypoints with imgaug
        """
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.do_augment = do_augment
        self.augmentation_name = augmentation_name
        self.ignore_keypts = ignore_keypts
        self.output_dim = output_dim
        self.preprocessing = preprocessing
        self.read_image_type = read_image_type
        self.repeat = repeat

        if not self.ignore_keypts:
            img_list = get_pairs_from_paths(images_path, keypts_path)
            if shuffle:
                random.shuffle(img_list)
            self.img_list_gen = itertools.chain(img_list)
        else:
            img_list = get_image_list_from_path(images_path)
            if shuffle:
                random.shuffle(img_list)
            self.img_list_gen = itertools.chain(img_list)

        self.steps_per_epoch = np.floor(len(img_list) / batch_size)
        self._current_step = 0

    def __iter__(self):
        return self

    def __next__(self):
        X = []
        Y = []
        if self._current_step > self.steps_per_epoch * self.repeat:
            raise StopIteration("Index out of order")

        for _ in range(self.batch_size):
            if self.ignore_keypts:
                im = next(self.img_list_gen)
                keypoints = None
            else:
                img_path, keypts_path = next(self.img_list_gen)
                keypoints, n_points, version = read_keypoints(keypts_path, is_imgaug_kps=True)
                assert n_points == self.n_classes, "No of Keypoint not equivalent to model configurations"

            im = cv2.imread(img_path, self.read_image_type)
            
            if self.do_augment:
                assert not self.ignore_keypts, "Not supported yet"
                im, kpsoi = augment_keypoints(im, keypoints, self.augmentation_name)
                # im = letterbox_image(im, size=self.output_dim)
                # kpsoi = kpsoi.on(im)
            
            if self.preprocessing is not None:
                im = self.preprocessing(im)
            X.append(im)

            if not self.ignore_keypts:
                hm_oimg = keypoints_to_heatmap(kpsoi)
                # Heatmaps of shape (H * W * num_keypoints), if you want to draw heatmaps in to image use
                # Image.fromarray(np.hstack(hm_oimg.draw_on_image(im))).show()
                Y.append(hm_oimg.to_uint8())

        self._current_step += 1

        if self.ignore_keypts:
            return np.array(X)
        else:
            return np.array(X), np.array(Y)


def get_train_dataset(
    images_path: str,
    segs_path: str,
    n_classes: str,
    generator_fn: iter,
    batch_size: int = 64,
    output_dim: typing.Tuple = (256, 256),
    read_image_type: int = cv2.IMREAD_COLOR,
    ignore_keypts: bool = False,
):
    """
    Function to convert Generator to tensorflow dataset

    Arguments:
        pairs_txt {[type]} -- [description]
        img_dir_path {[type]} -- [description]
        generator_fn {[type]} -- [description]

    Keyword Arguments:
        img_shape {[type]} -- [description] (default: {(160, 160, 3)})
        is_train {[type]} -- [description] (default: {True})
        batch_size {[type]} -- [description] (default: {64})

    Returns:
        [type] -- [description]
    """

    image_gen = partial(
        generator_fn,
        images_path,
        segs_path,
        n_classes,
        batch_size=batch_size,
        read_image_type=read_image_type,
        ignore_keypts=ignore_keypts,
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.from_generator(
        image_gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None, 3, *output_dim], [None, 1])
    )

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds
