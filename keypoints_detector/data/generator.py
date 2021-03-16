import cv2
import random
import itertools
import numpy as np


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False,
                                 augmentation_name="aug_all",
                                 custom_augmentation=None,
                                 other_inputs_paths=None, preprocessing=None,
                                 read_image_type=cv2.IMREAD_COLOR, ignore_segs=False):

    if not ignore_segs:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, other_inputs_paths=other_inputs_paths)
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
            if other_inputs_paths is None:

                if ignore_segs:
                    im = next(img_list_gen)
                    seg = None
                else:
                    im, seg = next(zipped)
                    seg = cv2.imread(seg, 1)

                im = cv2.imread(im, read_image_type)

                if do_augment:

                    assert not ignore_segs, "Not supported yet"

                    if custom_augmentation is None:
                        im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0], augmentation_name)
                    else:
                        im, seg[:, :, 0] = custom_augment_seg(im, seg[:, :, 0], custom_augmentation)

                if preprocessing is not None:
                    im = preprocessing(im)

                X.append(get_image_array(im, input_width, input_height, ordering=IMAGE_ORDERING))
            else:

                assert not ignore_segs, "Not supported yet"

                im, seg, others = next(zipped)

                im = cv2.imread(im, read_image_type)
                seg = cv2.imread(seg, 1)

                oth = []
                for f in others:
                    oth.append(cv2.imread(f, read_image_type))

                if do_augment:
                    if custom_augmentation is None:
                        ims, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                                        augmentation_name, other_imgs=oth)
                    else:
                        ims, seg[:, :, 0] = custom_augment_seg(im, seg[:, :, 0],
                                                               custom_augmentation, other_imgs=oth)
                else:
                    ims = [im]
                    ims.extend(oth)

                oth = []
                for i, image in enumerate(ims):
                    oth_im = get_image_array(image, input_width,
                                             input_height, ordering=IMAGE_ORDERING)

                    if preprocessing is not None:
                        if isinstance(preprocessing, Sequence):
                            oth_im = preprocessing[i](oth_im)
                        else:
                            oth_im = preprocessing(oth_im)

                    oth.append(oth_im)

                X.append(oth)

            if not ignore_segs:
                Y.append(get_segmentation_array(seg, n_classes, output_width, output_height))

        if ignore_segs:
            yield np.array(X)
        else:
            yield np.array(X), np.array(Y)
