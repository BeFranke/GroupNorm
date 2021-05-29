import os

import tensorflow as tf
import pandas as pd

from plot import plot_results

results_new = pd.DataFrame({
    "seed": [],
    "batch_size": [],
    "norm": [],
    "accuracy": []
})

_, (test_imgs, test_lbls) = tf.keras.datasets.cifar10.load_data()

test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test_lbls))

if __name__ == "__main__":
    for folder in os.listdir("models"):
        path = os.path.join(os.getcwd(), "models", folder)
        bs, n, s = folder.split("-")
        batch_size = int(bs.replace("BS", ""))
        norm = "Group Norm" if n == "GN" else "Batch Norm"
        seed = int(s.replace("S", ""))
        model = tf.keras.models.load_model(path)

        test_data_batch = test_data.batch(batch_size)

        print(f"evaluating {folder}!")
        _, acc = model.evaluate(test_data_batch)

        results_new = results_new.append({
            "seed": seed,
            "batch_size": batch_size,
            "norm": norm,
            "accuracy": acc
        }, ignore_index=True)

    results_new.to_csv("results_reproduced.csv", index=False)

    plot_results(fname="results_reproduced.csv")
