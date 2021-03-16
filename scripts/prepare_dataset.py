import os
import numpy as np
import pandas as pd


def load(fname, img_shape=(96, 96), sigma=5, is_train=True):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are 
           extracted. for example, cols could be:

          [left_eye_center_x, left_eye_center_y]

    return: 
    X:  2-d numpy array (Nsample, Ncol*Nrow)
    y:  2-d numpy array (Nsample, Nlandmarks*2) 
        In total there are 15 landmarks. 
        As x and y coordinates are recorded, u.shape = (Nsample,30)
    y0: panda dataframe containins the landmarks

    """
    from sklearn.utils import shuffle
    df = pd.read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    myprint = df.count()
    myprint = myprint.reset_index()
    print(myprint)
    # row with at least one NA columns are removed!
    ## df = df.dropna()
    df = df.fillna(-1)

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    if is_train:  # labels only exists for the training data
        y, y0, nm_landmark = get_y_as_heatmap(df, img_shape[0], img_shape[1], sigma)
        X, y, y0 = shuffle(X, y, y0, random_state=42)  # shuffle data
        y = y.astype(np.float32)
    else:
        y, y0, nm_landmark = None, None, None

    return X, y, y0, nm_landmark


def load2d(fname, img_shape=(96, 96), sigma=5, ):

    re = load(fname, img_shape, sigma)
    X = re[0].reshape(-1, img_shape[1], img_shape[0], 1)
    y, y0, nm_landmarks = re[1:]

    return X, y, y0, nm_landmarks


def _main():
    import argparse
    parser = argparse.ArgumentParser("Parse and prepare dataset from csv")
    parser.add_argument("-d", "--data-dir", type=str, required=True, help="csv path for training dataset")
    parser.add_argument('--sigma', default=5, type=int, help='default sigma value to generate heat-map')
    parser.add_argument('--img_shape', default=(96, 96), type=tuple, help='image size to be accepted by network')
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    FTRAIN = os.path.join(DATA_DIR, "training.csv")
    FTEST = os.path.join(DATA_DIR, "test.csv")
    fid_lookup = os.path.join(DATA_DIR, 'IdLookupTable.csv')
    X_train, y_train, y_train0, nm_landmarks = load2d(FTRAIN, img_shape=args.img_shape, sigma=args.sigma, is_train=True)
    X_test, y_test, _, _ = load2d(FTEST, img_shape=args.img_shape, sigma=args.sigma)


