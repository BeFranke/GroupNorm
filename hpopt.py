import numpy as np
import tensorflow as tf
import pandas as pd

from layer import GroupNormalization
from train import build_model

"""
This is just a small script to test 3 different values of G on a random 10% validation cut of CIFAR-10
results are written to hp.csv
"""

if __name__ == "__main__":
    # I deliberately left out group sizes that would transform GN layers into IN or LN
    Gs = [8, 4, 2]

    lr = 0.1
    res = {'G': [],
           'accuracy': []}
    seed = 1
    batch_size = 32

    (train_imgs, train_lbls), _ = tf.keras.datasets.cifar10.load_data()

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls)).shuffle(60000, seed=seed)
    test_data = train_data.take(6000)
    train_data = train_data.skip(6000)


    for G in Gs:
        norm = lambda: GroupNormalization(groups=G)

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr / (10 ** (epoch // 30))
        )

        # seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model = build_model(train_imgs, seed=seed, norm=norm, lr=lr)

        train_data_batch = train_data.batch(batch_size)
        test_data_batch = test_data.batch(batch_size)

        # setting validation set = test set for convenience
        history = model.fit(train_data_batch, epochs=100, callbacks=[lr_schedule],
                            validation_data=test_data_batch)

        _, acc = model.evaluate(test_data_batch)

        res['G'].append(G)
        res['accuracy'].append(acc)

        pd.DataFrame(res).to_csv("hp.csv", index=False)
