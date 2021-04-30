import json

import tensorflow as tf
import numpy as np

from layer import GroupNormalization

# 4 GPU training on BWunicluster
strategy = tf.distribute.MirroredStrategy()


def build_model(norm=tf.keras.layers.BatchNormalization):
    """
    :return: compiled model
    """
    inp = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = norm()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = norm()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = norm()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = norm()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


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
    # for norm in [tf.keras.layers.BatchNormalization]:
        for seed in [42, 43, 44, 45, 46]:
            # seed
            np.random.seed(seed)
            tf.random.set_seed(seed)

            with strategy.scope():
                model = build_model(norm)

            train_data_batch = train_data.batch(batch_size)
            test_data_batch = test_data.batch(batch_size)

            history = model.fit(train_data_batch, epochs=50)

            _, acc = model.evaluate(test_data_batch)

            res['seed'].append(seed)
            res['batch_size'].append(batch_size)
            res['norm'].append("Group Norm" if norm == GroupNormalization else 'Batch Norm')
            res['loss_curve'].append(history.history['loss'])
            res['accuracy'].append(acc)

            with open("results.json", "w+") as fp:
                json.dump(res, fp)
