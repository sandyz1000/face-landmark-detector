""" Convert kaggle dataset to general lfw format for keypoints detections
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = None
IMG_DIR = None

columns_lm = [
    "left_eye_center",
    "right_eye_center",
    "left_eye_inner_corner",
    "left_eye_outer_corner",
    "right_eye_inner_corner",
    "right_eye_outer_corner",
    "left_eyebrow_inner_end",
    "left_eyebrow_outer_end",
    "right_eyebrow_inner_end",
    "right_eyebrow_outer_end",
    "nose_tip",
    "mouth_left_corner",
    "mouth_right_corner",
    "mouth_center_top_lip",
    "mouth_center_bottom_lip",
]


def landmark_tuple_func(x, y):
    return (-1, -1) if np.isnan(x) or np.isnan(y) else (x, y)


def save_img_landmark(
    row, index, is_training=False, prefix="kaggle"
):
    img = row["Image"]
    pil_img = Image.fromarray(img)
    pil_img.save(os.path.join(IMG_DIR, f"{prefix}_{index}.png"))

    if is_training:
        landmarks = [
            landmark_tuple_func(row[colnm + "_x"], row[colnm + "_y"])
            for colnm in columns_lm
        ]
        with open(os.path.join(IMG_DIR, f"{prefix}_{index}.pts"), "w") as fp:
            fp.write("version: 1\n")
            fp.write("n_points: %d\n" % len(landmarks))
            fp.write("{\n")
            for landmark in landmarks:
                fp.write(" ".join([str(pt) for pt in landmark]) + "\n")
            fp.write("}")


def prepare_and_save(fname, img_shape=(96, 96), is_training=False, prefix="kaggle"):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are
           extracted. for example, cols could be:

          [left_eye_center_x, left_eye_center_y]

    return: None (Save the image and path to given location)

    """
    global IMG_DIR
    IMG_DIR = os.path.join(DATA_DIR, 'training' if is_training else 'test')
    os.makedirs(IMG_DIR, exist_ok=True)

    df = pd.read_csv(os.path.expanduser(fname))
    df["Image"] = (
        df["Image"].apply(lambda im: np.fromstring(im, sep=" ").reshape(img_shape[0], img_shape[1]).astype(np.uint8))
    )

    # row with at least one NA columns are removed!
    df = df.fillna(-1)

    for index, row in df.iterrows():
        save_img_landmark(
            row,
            index,
            is_training=is_training,
            prefix=prefix,
        )


def main():
    global DATA_DIR
    import argparse

    parser = argparse.ArgumentParser("Parse and prepare dataset from csv")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        required=True,
        help="csv path for training dataset",
    )
    # parser.add_argument('--sigma', default=5, type=int, help='default sigma value to generate heat-map')
    parser.add_argument(
        "--img_shape",
        default=(96, 96),
        type=tuple,
        help="image size to be accepted by network",
    )
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    FTRAIN = os.path.join(DATA_DIR, "training.csv")
    FTEST = os.path.join(DATA_DIR, "test.csv")

    prepare_and_save(FTRAIN, img_shape=args.img_shape, is_training=True)
    prepare_and_save(FTEST, img_shape=args.img_shape, is_training=False)


if __name__ == "__main__":
    main()
