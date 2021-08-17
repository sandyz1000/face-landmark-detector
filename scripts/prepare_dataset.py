""" Convert kaggle dataset to general lfw format for keypoints detections
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

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

landmark_tuple_func = (lambda x, y: (-1, -1) if np.isnan(x) or np.isnan(y) else (x, y))


def save_img_landmark(
    row, index, height, width, is_training_set=False, prefix="kaggle"
):

    landmarks = [
        landmark_tuple_func(row[colnm + "_x"], row[colnm + "_y"])
        for colnm in columns_lm
    ]
    img = row["Image"]
    pil_img = Image.fromarray(img).resize(width, height)
    pil_img.save(f"{prefix}_{index}.png")

    if is_training_set:
        with open(f"{prefix}_{index}.pts", "w") as fp:
            fp.write("version: 1")
            fp.write("n_points: %d" % len(landmarks))
            fp.write("{")
            for landmark in landmarks:
                fp.write(landmark.join(" "))
            fp.write("}")


def prepare_and_save(fname, img_shape=(96, 96), is_train=True, prefix="kaggle"):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are
           extracted. for example, cols could be:

          [left_eye_center_x, left_eye_center_y]

    return: None (Save the image and path to given location)

    """
    df = pd.read_csv(os.path.expanduser(fname))
    df["Image"] = (
        df["Image"].apply(lambda im: np.fromstring(im, sep=" ")).astype(np.float32)
    )

    # row with at least one NA columns are removed!
    df = df.fillna(-1)

    for index, row in enumerate(df.iterrows()):
        save_img_landmark(
            row,
            index,
            img_shape[0],
            img_shape[1],
            is_training_set=is_train,
            prefix=prefix,
        )


def main():
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

    prepare_and_save(FTRAIN, img_shape=args.img_shape, is_train=True)
    prepare_and_save(FTEST, img_shape=args.img_shape)


if __name__ == "__main__":
    main()
