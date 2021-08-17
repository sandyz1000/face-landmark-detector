# %%
# TODO:
# - Verify datagenerator -- Doing
# - Verify training and evals script -- Doing
# %%
import os
from keypoints_detector.data.generator import image_keypoints_generator, custom_image_keypts_generator
from keypoints_detector import training

data_dir = ""
train_data_dir, valid_data_dir = [os.path.join(data_dir, x) for x in ('train', 'valid')]
train_dataset = training.DatasetCfg(img_dirpath=train_data_dir, keypts_dirpath=train_data_dir)
valid_dataset = training.DatasetCfg(img_dirpath=valid_data_dir, keypts_dirpath=valid_data_dir)

gen = image_keypoints_generator()