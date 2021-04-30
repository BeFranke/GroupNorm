import json

import tensorflow as tf

from layer import GroupNormalization

# 4 GPU training on BWunicluster
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

    num_blocks_list = [2, 5, 5, 2]
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
            with strategy.scope():
                model_bn = build_model(norm)

            train_data_batch = train_data.batch(batch_size)
            test_data_batch = test_data.batch(batch_size)

            history = model_bn.fit(train_data_batch, epochs=50)

            _, acc = model_bn.evaluate(test_data_batch)

            res['seed'].append(seed)
            res['batch_size'].append(batch_size)
            res['norm'].append("Group Norm" if norm == GroupNormalization else 'Batch Norm')
            res['loss_curve'].append(history.history['loss'])
            res['accuracy'].append(acc)

            with open("results.json", "w+") as fp:
                json.dump(res, fp)
