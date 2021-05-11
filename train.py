import pandas as pd
import shutil
import os

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from layer import GroupNormalization

"""
Hyperparameters taken from Wu & H:
- weight decay = 0.0001
- epochs = 100
- data augmentation

Learning rate was taken from ResNet-Paper, but decays as in Wu & He.
As Wu & He did not describe their data augmentation in detail (and neither the paper they referenced), 
my augmentation could deviate slightly. I used random zoom, flip and rotation.
"""


def augmentation(x, adapt_data):
    # learned normalization instead of simple scaling by dividing by 255
    # requires call to adapt()
    normalize_input = tf.keras.layers.experimental.preprocessing.Normalization()
    normalize_input.adapt(adapt_data)
    x = normalize_input(x)
    x = tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0, -0.3), width_factor=(0, -0.3))(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.2, 0.2))(x)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal")(x)
    return x


def build_model(adapt_data, norm=tf.keras.layers.BatchNormalization, lr=0.001):
    """
    ResNet32 for CIFAR-10 like described in https://arxiv.org/pdf/1512.03385.pdf, section 4.2 (here, n=5)
    Only deviation is that I use projection shortcuts when dimensions change (they only omitted it to have the same
    number of parameters as the non-residual baseline).
    :return: compiled model
    """
    kwargs = {
        'kernel_size': 3,
        'padding': 'same',
        'kernel_regularizer': tf.keras.regularizers.L2(0.0001),
        'bias_regularizer': tf.keras.regularizers.L2(0.0001)
    }

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
    x = augmentation(inp, adapt_data)
    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L2(0.0001)
    )(x)
    x = norm()(x)

    # resolution:
    #          32  32  32  32  32  16  16  16  16  16   8   8   8   8   8
    filters = [16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]
    for i, f in enumerate(filters):
        downscale = i != 0 and filters[i - 1] < f
        x = block(x, f, downscale)

    # output block
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.L2(0.0001),
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

    # LR like described in (https://arxiv.org/pdf/1512.03385.pdf)
    LEARNING_RATE = 0.1

    # 4 GPU training on BWunicluster
    strategy = tf.distribute.MirroredStrategy()

    # clear logdir
    shutil.rmtree("logs")
    os.mkdir("logs")

    ((train_imgs, train_lbls), (test_imgs, test_lbls)) = tf.keras.datasets.cifar10.load_data()

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test_lbls))

    res = {
        'seed': [],
        'batch_size': [],
        'norm': [],
        'accuracy': []
    }

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE / (10 ** (epoch // 30))
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
                    model = build_model(train_imgs, norm, lr=LEARNING_RATE)

                train_data_batch = train_data.batch(batch_size)
                test_data_batch = test_data.batch(batch_size)

                model.fit(train_data_batch, epochs=100, callbacks=[tboard, lr_schedule])

                _, acc = model.evaluate(test_data_batch)

                res['seed'].append(seed)
                res['batch_size'].append(batch_size)
                res['norm'].append("Group Norm" if norm == GroupNormalization else 'Batch Norm')
                res['accuracy'].append(acc)

                exit(0)

                pd.DataFrame(res).to_csv("results.csv", index=False)

                model.save(
                    f"models/{model_id}"
                )
