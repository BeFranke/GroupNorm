import numpy as np
import tensorflow as tf
import json

from layer import GroupNormalization
from train import build_model

if __name__ == "__main__":
    # I deliberately left out group sizes that would transform GN layers into IN or LN
    Gs = [8, 4, 2]

    lr = 0.1

    (train_imgs, train_lbls), _ = tf.keras.datasets.cifar10.load_data()

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    test_data = train_data.take(6000)
    train_data = train_data.skip(6000)

    res = []
    seed = 1
    batch_size = 32


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
        history = model.fit(train_data_batch, epochs=50, callbacks=[lr_schedule],
                            validation_data=test_data_batch)

        _, acc = model.evaluate(test_data_batch)

        res.append(acc)

        with open("hp.json", "w+") as fp:
            json.dump({'group_size': Gs, 'accuracy': res}, fp)

    print("------------------------")
    print("Optimization Results:")
    print("groups: validation accuracy")
    print("----------------------------")
    for i, g in enumerate(Gs):
        print(f"{g}: {res[i]} ")
