import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_results(path_for_result_png=None, with_std=True):
    df = pd.read_csv("results_old_cluster.csv")
    df["error_rate"] = (1 - df["accuracy"]) * 100
    ax = sns.lineplot(
        x="batch_size",
        y="error_rate",
        hue="norm",
        marker='o',
        data=df
    )
    ax.invert_xaxis()
    ax.set_xlabel("batch size")
    ax.set_ylabel("error (%)")
    ax.set_xscale("log")
    plt.show()


if __name__ == "__main__":
    plot_results()

