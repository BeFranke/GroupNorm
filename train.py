import pandas as pd
import shutil
import os

import tensorflow as tf
import numpy as np

from layer import GroupNormalization


def build_model(norm=tf.keras.layers.BatchNormalization, lr=0.001):
    """
    builds what is basically a fully-convolutional ResNet.
    The "regular" resnets are rather aggressive about reducing the feature map sizes,
    which is not ideal for 32x32 images
    :return: compiled model
    """

    def block(x, filters):
        y = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", strides=2)(x)
        y = tf.keras.layers.LeakyReLU()(y)
        y = norm()(y)
        y = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same")(y)
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding="same", strides=2)(x)  # shortcut connection
        y = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LeakyReLU()(y)
        y = norm()(y)
        return y

    inp = tf.keras.Input((32, 32, 3))
    x = inp
    for f in [64, 128, 256, 512]:
        x = block(x, f)

    # output block
    x = tf.keras.layers.Conv2D(512, kernel_size=3, padding="same", stride=2)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = norm(x)

    # final 1x1 convolution to achieve n_filters = n_classes
    x = tf.keras.layers.Conv2D(10, kernel_size=1, padding="same")(x)

    # Flatten() squeezes away the singleton dimensions
    x = tf.keras.layers.Flatten()(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


if __name__ == "__main__":

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
           'loss_curve': [],
           'accuracy': []}

    for batch_size in [32, 16, 8, 4, 2]:
        for norm in [GroupNormalization, tf.keras.layers.BatchNormalization]:
            for seed in [42, 43, 44, 45, 46]:
                model_id = f"BS{batch_size}-{'GN' if norm == GroupNormalization else 'BN'}-S{seed}"
                tboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{model_id}")

                # seed
                np.random.seed(seed)
                tf.random.set_seed(seed)

                with strategy.scope():
                    model = build_model(norm)

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
