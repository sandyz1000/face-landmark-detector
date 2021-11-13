# %%

import os

from keypoints_detector.data.generator import (
    image_keypoints_generator, custom_image_keypts_generator, 
    plot_img_hm_pair, get_image_array
)
from keypoints_detector import training

# %%
data_dir = "/Volumes/SDM/Dataset/fcn_dataset/keypts_dataset"
train_data_dir, valid_data_dir = [os.path.join(data_dir, x) for x in ('train', 'val')]
train_dataset = training.DatasetCfg(img_dirpath=train_data_dir, keypts_dirpath=train_data_dir)
valid_dataset = training.DatasetCfg(img_dirpath=valid_data_dir, keypts_dirpath=valid_data_dir)

gen = image_keypoints_generator(
    train_dataset.img_dirpath,
    train_dataset.keypts_dirpath,
    n_classes=68, do_augment=True
)
# %%
data_dir = "/Volumes/SDM/Dataset/fcn_dataset/facial-keypoints-detection"
train_data_dir, valid_data_dir = [os.path.join(data_dir, x) for x in ('training', 'test')]
train_dataset = training.DatasetCfg(img_dirpath=train_data_dir, keypts_dirpath=train_data_dir)
valid_dataset = training.DatasetCfg(img_dirpath=valid_data_dir, keypts_dirpath=valid_data_dir)

gen = custom_image_keypts_generator(
    train_dataset.img_dirpath,
    train_dataset.keypts_dirpath,
    n_classes=15, do_augment=True
)
# %%
X, y = next(gen)
# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# %%
from keypoints_detector.data.generator import letterbox_image
# %%
img_path = '/Volumes/SDM/Dataset/fcn_dataset/keypts_dataset/train/3187108438_1.jpg'
grayscale = False

im = np.array(
    Image.open(img_path).convert('L')
    if grayscale else Image.open(img_path), dtype='uint8'
)

im = letterbox_image(im, size=(800, 800))
plt.imshow(im)
# %%
n_classes = 68

LANDMARKS_MODELS = {
    'fcn_8_resnet50',
    'fcn_8_mobilenet',
    'fcn_8_vgg',
    'default'
}
img_dim = (640, 640)
trainer = training.Train(net='fcn_8_mobilenet', train_dataset=train_dataset, valid_dataset=valid_dataset, n_classes=n_classes, img_dim=img_dim)
trainer.init_train(epochs=1, batch_size=8)