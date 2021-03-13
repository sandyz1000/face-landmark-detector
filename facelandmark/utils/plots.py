import numpy as np
import matplotlib.pyplot as plt


def plt_img(X_train, idx):
    plt.imshow(X_train[idx, :, :, 0], cmap="gray")
    plt.title("original")
    plt.axis("off")
    plt.show()


def plt_actual_pred(X_train, y_train, y_pred, nplot, lm_colname):
    for i in range(96, 100):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(X_train[i, :, :, 0], cmap="gray")
        ax.axis("off")

        fig = plt.figure(figsize=(20, 3))
        count = 1
        for j, lab in enumerate(lm_colname):
            ax = fig.add_subplot(2, nplot, count)
            ax.set_title(lab[:10] + "\n" + lab[10:-2])
            ax.axis("off")
            count += 1
            ax.imshow(y_pred[i, :, :, j])
            if j == 0:
                ax.set_ylabel("prediction")

        for j, lab in enumerate(lm_colname):
            ax = fig.add_subplot(2, nplot, count)
            count += 1
            ax.imshow(y_train[i, :, :, j])
            ax.axis("off")
            if j == 0:
                ax.set_ylabel("true")
        plt.show()


def plt_multiple(X_train, y_train, nplot, lm_colname, num_row=3):
    # TODO: Fix this method

    for i in range(num_row):
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(2, nplot / 2, 1)
        ax.imshow(X_train[i, :, :, 0], cmap="gray")
        ax.set_title("input")
        for j, lab in enumerate(lm_colname):
            ax = fig.add_subplot(2, nplot / 2, j + 2)
            ax.imshow(y_train[i, :, :, j], cmap="gray")
            ax.set_title(str(j) + "\n" + lab)
        plt.show()


def show_transformed(X_train, y_train, lm_colname, Nplot, iexample=0, transform_fn=None):
    count = 1
    Nhm = 10
    fig = plt.figure(figsize=[Nhm * 2.5, 2 * Nplot])
    for _ in range(Nplot):
        x_batch, y_batch = transform_fn(X_train[[iexample]], y_train[[iexample]])
        ax = fig.add_subplot(Nplot, Nhm + 1, count)
        ax.imshow(x_batch[0, :, :, 0], cmap="gray")
        ax.axis("off")
        count += 1

        for ifeat in range(Nhm):
            ax = fig.add_subplot(Nplot, Nhm + 1, count)
            ax.imshow(y_batch[0, :, :, ifeat], cmap="gray")
            ax.axis("off")
            if count < Nhm + 2:
                ax.set_title(lm_colname[ifeat])
            count += 1
    plt.show()


