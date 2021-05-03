import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization

from train import build_model

import numpy as np

"""
A short search for the optimal learning rate. To keep closer to He & Wu, parameter selection is done for 
a network with BatchNorm (Wu & He did not optimize parameters for their new Normalization).
Further, I do not optimize learning rates separately for each batch size, as Wu & He did not do this either.
For both, hyperparameter search and evaluation, training budget was 100 epochs like in Wu & He.
"""


lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
seeds = [1, 2, 3, 4, 5]

# 4 GPU training on BWunicluster
strategy = tf.distribute.MirroredStrategy()

# manual validation split, we do not want to search on the test set
(train_imgs, train_lbls), _ = tf.keras.datasets.cifar10.load_data()
train_imgs = train_imgs / 255.0

val_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls)).take(6000).batch(32)
train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls)).skip(6000).batch(32)

results = {
    "seed": [],
    "lr": [],
    "val_acc": []
}

for seed in seeds:
    for lr in lrs:
        # seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        with strategy.scope():
            model = build_model(BatchNormalization, lr=lr)

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr / (10 ** (epoch // 30))
        )

        model.fit(train_data, epochs=100, callbacks=[lr_schedule])
        _, acc = model.evaluate(val_data)

        results['seed'].append(seed)
        results['lr'].append(lr)
        results['val_acc'].append(acc)

        pd.DataFrame(results).to_csv("hparams.csv", index=False)
