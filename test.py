import tensorflow as tf
import json
from matplotlib import pyplot as plt

from layer import GroupNormalization

# basically a simplified copy of main.py

strategy = tf.distribute.MirroredStrategy()


def residual_block(x, downsample, filters, norm=tf.keras.layers.BatchNormalization):
    """
    inspired by https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
    :param x: input tensor
    :param downsample: True if image resolution should be halved
    :param filters: number of filters
    :return: output tensor
    """
    y = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=int(downsample) + 1,
                               padding="same", activation="relu")(x)
    y = norm()(y)
    y = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(y)
    if downsample:
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=2, padding="same")(x)

    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.layers.ReLU()(out)
    out = norm()(out)
    return out


def build_model(norm=tf.keras.layers.BatchNormalization):
    """
    also inspired by https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
    :return:
    """
    with strategy.scope():
        inputs = tf.keras.Input(shape=(32, 32, 3))
        num_filters = 64

        if norm == GroupNormalization:
            t = norm(G=3)(inputs)
        else:
            t = norm()(inputs)

        t = tf.keras.layers.Conv2D(kernel_size=3,
                                   strides=1,
                                   filters=num_filters,
                                   padding="same", activation="relu")(t)
        t = norm()(t)

        num_blocks_list = [2, 5]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters, norm=norm)
            num_filters *= 2

        t = tf.keras.layers.AveragePooling2D(4)(t)
        t = tf.keras.layers.Flatten()(t)
        outputs = tf.keras.layers.Dense(10)(t)


        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    return model

((train_imgs, train_lbls), (test_imgs, test_lbls)) = tf.keras.datasets.cifar10.load_data()
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

model_bn = build_model(
    GroupNormalization
)
model_bn.fit(train_imgs, train_lbls, validation_split=0.1, epochs=5, batch_size=16)

model_bn.evaluate(test_imgs, test_lbls)
