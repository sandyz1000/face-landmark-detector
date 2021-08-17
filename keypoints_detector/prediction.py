import os
import cv2
import json
import six
import typing
import numpy as np
import tensorflow as tf
from keypoints_detector.data.generator import get_image_array
from keypoints_detector.utils.plots import visualize_keypoints, class_colors, draw_marks
from keypoints_detector.data.config import IMAGE_ORDERING
from keypoints_detector.training import find_latest_checkpoint

# TODO: Fix this module


def detect_marks(img, model, face):
    """
    Find the facial landmarks in an image from the faces

    Parameters
    ----------
    img : np.uint8
        The image in which landmarks are to be found
    model : Tensorflow model
        Loaded facial landmark model
    face : list
        Face coordinates (x, y, x1, y1) in which the landmarks are to be found

    Returns
    -------
    marks : numpy array
        facial landmark points

    """

    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x, top_y, right_x, bottom_y = box

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        # Height > width, a slim box.
        elif diff > 0:
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        # Width > height, a short box.
        else:
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    def move_box(box, offset):
        """Move the box to direction specified by vector offset
        """
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    offset_y = int(abs((face[3] - face[1]) * 0.1))
    box_moved = move_box(face, [0, offset_y])
    facebox = get_square_box(box_moved)

    face_img = img[facebox[1]: facebox[3],
                   facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    predictions = model.signatures["predict"](tf.constant(np.expand_dims(face_img, axis=0), dtype=tf.uint8))['output']
    # face_img_tensor = tf.expand_dims(tf.convert_to_tensor(face_img, dtype=tf.uint8), axis=0)
    # predictions = model.signatures["predict"](face_img_tensor)['output']
    # Convert predictions to landmarks i.e to 68 landmark.
    marks = np.array(predictions).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))

    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    marks = marks.astype(np.uint)

    return marks


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


def model_from_checkpoint_path(checkpoints_path: str) -> tf.keras.Model:
    from .networks.basic_models import LANDMARKS_MODELS
    assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = LANDMARKS_MODELS[model_config['model_class']](
        model_config['n_classes'],
        input_height=model_config['input_height'],
        input_width=model_config['input_width']
    )
    print("loaded weights ", latest_weights)
    status = model.load_weights(latest_weights)

    if status is not None:
        status.expect_partial()

    return model


def show_transformed(X_train, y_train, num_plot, lm_colname, iexample=0, transform_fn=None):
    import matplotlib.pyplot as plt
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


def keypts_predict(
        model: tf.keras.Model = None,
        inp: typing.Union[np.ndarray, str] = None,
        out_fname: str = None,
        checkpoints_path: str = None, overlay_img: bool = False,
        class_names=None, show_legends: bool = False, colors: typing.List[tuple] = class_colors,
        pred_dim: typing.Tuple[int] = None, read_image_type=1
) -> np.ndarray:
    if model is None and checkpoints_path is None:
        raise ValueError("Both model and checkpoint_path cannot be empty")

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None), "Invalid input, should be either directory, ndarray or image path"
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
    if os.path.isdir(inp):
        # TODO: Iterate and predict for all image in a directory
        pass

    _prediction(
        model, inp,
        input_width, input_height,
        output_height, output_width,
        n_classes, colors, show_legends, class_names,
        pred_dim, overlay_img, out_fname
    )


def _prediction(
        model: tf.keras.Model, inp: np.ndarray,
        input_width: int, input_height: int,
        output_height: int, output_width: int,
        n_classes: int, colors: typing.List[typing.Tuple],
        show_legends: bool, class_names: typing.List[str],
        pred_dim: typing.Tuple[int], overlay_img: bool, out_fname: str
):
    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = visualize_keypoints(
        pr, inp, n_classes=n_classes,
        colors=colors, overlay_img=overlay_img,
        show_legends=show_legends,
        class_names=class_names,
        pred_dim=pred_dim,
    )

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr
