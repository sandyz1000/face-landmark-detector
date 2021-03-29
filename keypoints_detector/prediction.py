import os
import cv2
import json
import six
import numpy as np
from .data.generator import get_image_array
from .utils.plots import visualize_segmentation, class_colors
from .data.config import IMAGE_ORDERING


def video_predict(facedetector_fn, landmark_model):
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        rects = facedetector_fn(img)

        for rect in rects:
            marks = detect_marks(img, landmark_model, rect)
            draw_marks(img, marks)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict(model, img, out_fname):
    out = model.predict_segmentation(inp=img, out_fname=out_fname)
    return out


def model_from_checkpoint_path(checkpoints_path):
    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    status = model.load_weights(latest_weights)

    if status is not None:
        status.expect_partial()

    return model



def show_transformed(X_train, y_train, num_plot, lm_colname, iexample=0, transform_fn=None):
    count = 1
    Nhm = 10
    fig = plt.figure(figsize=[Nhm * 2.5, 2 * num_plot])
    for _ in range(num_plot):
        x_batch, y_batch = transform_fn(X_train[[iexample]], y_train[[iexample]])
        ax = fig.add_subplot(num_plot, Nhm + 1, count)
        ax.imshow(x_batch[0, :, :, 0], cmap="gray")
        ax.axis("off")
        count += 1

        for ifeat in range(Nhm):
            ax = fig.add_subplot(num_plot, Nhm + 1, count)
            ax.imshow(y_batch[0, :, :, ifeat], cmap="gray")
            ax.axis("off")
            if count < Nhm + 2:
                ax.set_title(lm_colname[ifeat])
            count += 1
    plt.show()


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None, read_image_type=1):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)

    assert (len(inp.shape) == 3 or len(inp.shape) == 1 or len(inp.shape) == 4), "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr