import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in range(5000)]

# def plt_img(X_train, idx):
#     plt.imshow(X_train[idx, :, :, 0], cmap="gray")
#     plt.title("original")
#     plt.axis("off")
#     plt.show()


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


# TODO: Fix this below visualization method
def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], : legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    fused_img = (inp_img / 2 + seg_img / 2).astype('uint8')
    return fused_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):
    # TODO: Fix this method
    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        original_h = inp_img.shape[0]
        original_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img, (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img
