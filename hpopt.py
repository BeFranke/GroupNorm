import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization

from layer import GroupNormalization
from train import build_model

import numpy as np

lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
seeds = [42, 43, 44, 45, 46]
norms = [GroupNormalization, BatchNormalization]

# manual validation split, we do not want to search on the test set
(train_imgs, train_lbls), _ = tf.keras.datasets.cifar10.load_data()
train_imgs = train_imgs / 255.0

train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls)).take()

# seed
np.random.seed(seed)
tf.random.set_seed(seed)