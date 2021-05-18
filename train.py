import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from layer import GroupNormalization

from argparse import ArgumentParser


def build_model(adapt_data: tf.Tensor,
                seed: int,
                norm: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
                lr: float = 0.001) -> tf.keras.Model:
    """
    ResNet20 for CIFAR-10 like described in https://arxiv.org/pdf/1512.03385.pdf, section 4.2 (here, n=5)
    Only deviation is that I use projection shortcuts when dimensions change (they only omitted it to have the same
    number of parameters as the non-residual baseline).
    :param adapt_data: training data to compute mean and variance for initial normalization
    :param seed: random seed for reproducibility
    :param norm: norm to use, should be one of BatchNormalization or GroupNormalization (class object, NOT instance!)
    :param lr: initial learning rate to use
    :return: compiled model
    """
    kwargs = {
        'kernel_size': 3,
        'padding': 'same',
        'kernel_regularizer': tf.keras.regularizers.L2(0.0001),
        'bias_regularizer': tf.keras.regularizers.L2(0.0001),
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.HeNormal(seed)
    }

    def augmentation(x: tf.Tensor, adapt_data: tf.Tensor, seed: int) -> tf.Tensor:
        """
        handles data augmentation & preprocessing
        :param x: input tensor
        :param adapt_data: entire training data to compute mean & variance
        :param seed: random seed
        :return: x after preprocessing
        """
        # learned normalization instead of simple scaling by dividing by 255
        # requires call to adapt()
        normalize_input = tf.keras.layers.experimental.preprocessing.Normalization()
        normalize_input.adapt(adapt_data)
        x = normalize_input(x)
        # pad 4 pixels on each side, then random crop to 32x32 like described in the ResNet Paper
        x = tf.image.pad_to_bounding_box(x, offset_height=4, offset_width=4, target_height=40, target_width=40)
        x = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=seed)(x)
        x = tf.keras.layers.experimental.preprocessing.RandomCrop(32, 32, seed=seed)(x)
        return x

    def block(x: tf.Tensor,
              filters: int,
              norm: tf.keras.layers.Layer,
              downscale: bool = False) -> tf.Tensor:
        """
        Single ResNet-Block with projection shortcut on dimension change.
        :param x: input tensor
        :param filters: number of filters (per layer) in this block
        :param norm: normalization layer to use
        :param downscale: True if spatial resolution should be reduced
        :return: output tensor
        """
        stride = int(downscale) + 1
        c_in = x.shape[-1]
        y = tf.keras.layers.Conv2D(filters, strides=stride, **kwargs)(x)
        y = tf.keras.layers.ReLU()(y)
        y = norm()(y)
        y = tf.keras.layers.Conv2D(filters, **kwargs)(y)
        if c_in != filters or downscale:
            # shortcut connection
            x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride,
                                       **{key: kwargs[key] for key in kwargs if key != 'kernel_size'})(x)

        # norm before addition, like recommended here: http://torch.ch/blog/2016/02/04/resnets.html
        # Wu & He initialize the last norm of each convolutional block with gamma=0 s.t. the initial state is identity
        if norm == GroupNormalization:
            y = norm(gamma_initializer='zeros')(y)
        else:
            y = norm()(y)

        y = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.ReLU()(y)
        return y

    inp = tf.keras.Input((32, 32, 3))
    x = augmentation(inp, adapt_data, seed=seed)
    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L2(0.0001), use_bias=False,
        kernel_initializer=tf.keras.initializers.HeNormal(seed)
    )(x)
    x = norm()(x)

    # resolution:
    #          32  32  32  16  16  16   8   8   8
    filters = [16, 16, 16, 32, 32, 32, 64, 64, 64]
    for i, f in enumerate(filters):
        downscale = i != 0 and filters[i - 1] < f
        x = block(x, f, norm, downscale)

    # output block
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.L2(0.0001),
                              bias_regularizer=tf.keras.regularizers.L2(0.0001))(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    # SGD with nesterov momentum, like recommended here: http://torch.ch/blog/2016/02/04/resnets.html
    # (this is a blog cited by the facebookresearch-github page cited by the Wu & He Paper)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--restart", action="store_true", default=False, help="remove old results before training")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1], help="specify all seeds to run")

    args = ap.parse_args()

    # LR like described in (https://arxiv.org/pdf/1512.03385.pdf)
    get_lr = lambda batch_size: 0.1 * batch_size / 32

    # 4 GPU training on BWunicluster
    strategy = tf.distribute.MirroredStrategy()

    if args.restart:
        # clear logdir
        shutil.rmtree("logs", ignore_errors=True)
        os.mkdir("logs")

    ((train_imgs, train_lbls), (test_imgs, test_lbls)) = tf.keras.datasets.cifar10.load_data()

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test_lbls))

    res = pd.DataFrame({
        'seed': [],
        'batch_size': [],
        'norm': [],
        'accuracy_final': [],
        'accuracy_5': []
    })

    if not args.restart:
        try:
            res = pd.read_csv("results.csv")
        except:
            pass

    for seed in args.seeds:
        for batch_size in [128, 32, 16, 8, 4, 2]:
            for norm in [GroupNormalization, tf.keras.layers.BatchNormalization]:
                norm_str = "Group Norm" if norm == GroupNormalization else "Batch Norm"
                if ((res["seed"] == seed) & (res["batch_size"] == batch_size) & (res["norm"] == norm_str)).any():
                    print(f"Skipping S{seed}-BS{batch_size}-{norm_str}!")
                else:
                    print(f"Training S{seed}-BS{batch_size}-{norm_str}!")

                lr = get_lr(batch_size)

                lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: lr / (10 ** (epoch // 30))
                )

                model_id = f"BS{batch_size}-{'GN' if norm == GroupNormalization else 'BN'}-S{seed}"
                tboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{model_id}")

                # seed
                np.random.seed(seed)
                tf.random.set_seed(seed)

                with strategy.scope():
                    model = build_model(train_imgs, seed=seed, norm=norm, lr=lr)

                train_data_batch = train_data.batch(batch_size)
                test_data_batch = test_data.batch(batch_size)

                # as I did not do any parameter tuning, I can set validation set = test set for convenience
                history = model.fit(train_data_batch, epochs=100, callbacks=[tboard, lr_schedule],
                                    validation_data=test_data_batch)

                _, acc = model.evaluate(test_data_batch)

                # average of last 5 epochs
                acc_5 = np.mean(history.history['val_accuracy'][-5:])

                res_tmp = {
                    'seed': seed,
                    'batch_size': batch_size,
                    'norm': norm_str,
                    'accuracy_final': acc,
                    'accuracy_5': acc_5
                }

                res = res.append(res_tmp, ignore_index=True)
                res.to_csv("results.csv")

                model.save(
                    f"models/{model_id}"
                )
