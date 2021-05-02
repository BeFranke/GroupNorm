import pandas as pd
import shutil
import os

import tensorflow as tf
import numpy as np

from layer import GroupNormalization

"""
Hyperparameters taken from Wu & H:
- weight decay = 0.0001
- epochs = 100
- data augmentation

Learning rate was tuned with hpopt.py, but decays as in Wu & He.
As Wu & He did not describe their data augmentation in detail (and neither the paper they referenced), 
my augmentation could deviate slightly. I used random zoom and rotation.
"""


def augmentation(x):
    x = tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0, -0.2), width_factor=(0, -0.2))(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.2, 0.2))(x)
    return x


def build_model(norm=tf.keras.layers.BatchNormalization, lr=0.001):
    """
    ResNet18, but without the first pooling layer, and the first convolution does not use strides as resolution would
    be lost too fast.
    :return: compiled model
    """
    kwargs = {'kernel_size': 3,
              'padding': 'same',
              'kernel_regularizer': tf.keras.regularizers.L2(0.0001),
              'bias_regularizer': tf.keras.regularizers.L2(0.0001)}

    def block(x, filters, downscale=False):
        stride = int(downscale) + 1
        c_in = x.shape[-1]
        y = tf.keras.layers.Conv2D(filters, strides=stride, **kwargs)(x)
        y = tf.keras.layers.LeakyReLU()(y)
        y = norm()(y)
        y = tf.keras.layers.Conv2D(filters, **kwargs)(y)
        if c_in != filters or downscale:
            # shortcut connection
            x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride,
                                       **{key: kwargs[key] for key in kwargs if key != 'kernel_size'})(x)

        # norm before addition, like recommended here: http://torch.ch/blog/2016/02/04/resnets.html
        y = norm()(y)
        y = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LeakyReLU()(y)
        return y

    inp = tf.keras.Input((32, 32, 3))
    x = augmentation(inp)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.0001),
                               bias_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    x = norm()(x)

    # resolution:
    #          32  32  16   16   8    8    4    4
    filters = [64, 64, 128, 128, 256, 256, 512, 512]
    for i, f in enumerate(filters):
        downscale = i != 0 and filters[i - 1] < f
        x = block(x, f, downscale)

    # output block
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.L2(0.0001),
                              bias_regularizer=tf.keras.regularizers.L2(0.0001))(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    # SGD with nesterov momentum, like recommended here: http://torch.ch/blog/2016/02/04/resnets.html
    # (this is a blog cited by the facebookresearch-github page cited by the Wu & He Paper)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


if __name__ == "__main__":

    # tuned using hpopt.py
    LEARNING_RATE = 0.001

    # 4 GPU training on BWunicluster
    strategy = tf.distribute.MirroredStrategy()

    # clear logdir
    shutil.rmtree("logs")
    os.mkdir("logs")

    ((train_imgs, train_lbls), (test_imgs, test_lbls)) = tf.keras.datasets.cifar10.load_data()
    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test_lbls))

    res = {'seed': [],
           'batch_size': [],
           'norm': [],
           'accuracy': []}

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * epoch // 30
    )

    for batch_size in [32, 16, 8, 4, 2]:
        for norm in [GroupNormalization, tf.keras.layers.BatchNormalization]:
            for seed in [1, 2, 3, 4, 5]:
                model_id = f"BS{batch_size}-{'GN' if norm == GroupNormalization else 'BN'}-S{seed}"
                tboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{model_id}")

                # seed
                np.random.seed(seed)
                tf.random.set_seed(seed)

                with strategy.scope():
                    model = build_model(norm, lr=LEARNING_RATE)

                train_data_batch = train_data.batch(batch_size)
                test_data_batch = test_data.batch(batch_size)

                model.fit(train_data_batch, epochs=100, callbacks=[tboard])

                _, acc = model.evaluate(test_data_batch)

                res['seed'].append(seed)
                res['batch_size'].append(batch_size)
                res['norm'].append("Group Norm" if norm == GroupNormalization else 'Batch Norm')
                res['accuracy'].append(acc)

                pd.DataFrame(res).to_csv("results.csv", index=False)

                model.save(
                    f"models/{model_id}"
                )
