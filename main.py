import json

import tensorflow as tf
import numpy as np

from layer import GroupNormalization

# 4 GPU training on BWunicluster
strategy = tf.distribute.MirroredStrategy()


def build_model(norm=tf.keras.layers.BatchNormalization):
    """
    load MobileNet, optionally replace all BN layers with GN
    I use MobileNet instead of MobileNetV2 because MobileNetV2 has extremely unconventional feature map dimensions
    (e.g. 144), that makes Grouping them in even Groups that are not too small much more challenging and therefore makes
    it harder to compare the results to teh original paper
    :return: compiled model
    """
    model = tf.keras.applications.MobileNet(input_shape=(32, 32, 3), weights=None, classes=10,
                                            classifier_activation=None)
    # model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), weights=None, classes=10,
    #                                        classifier_activation=None)
    if norm == GroupNormalization:
        # layer replacements as shown in:
        # https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
        layers = [l for l in model.layers]
        x = layers[0].output
        replaced = 0
        for i in range(1, len(layers)):
            if isinstance(layers[i], tf.keras.layers.BatchNormalization):
                x = GroupNormalization()(x)
                # x = tf.keras.layers.LayerNormalization()(x)
                replaced += 1
            else:
                x = layers[i](x)

        print(f"replaced {replaced} BatchNorm layers!")

        model = tf.keras.Model(layers[0].input, x)

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
            # seed
            np.random.seed(seed)
            tf.random.set_seed(seed)

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
