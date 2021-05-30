import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_results(fname="results.csv", resname="plot.png"):
    sns.set_theme()
    df = pd.read_csv(fname)
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
    ax.set_ylabel("test error (%)")
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.legend(title="Normalization Type")
    fig = plt.gcf()
    fig.savefig(resname)
    plt.show()


if __name__ == "__main__":
    plot_results()

