# External libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_barplot(data, title="", out=None):
    # Setting plot background
    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

    # Plotting barplot
    sns.barplot(x=list(range(len(data))), y=data, orient="v",
                palette=sns.cubehelix_palette(20, reverse=True), ax=ax)

    # Setting horizontal line
    ax.axhline(0, color="k", clip_on=False)

    # Setting up
    ax.set_ylabel("Samples")
    ax.set_title(title, fontsize=20)

    # Set up x labels
    ax.set_xticklabels(data.index, rotation=65, fontsize=14)

    # Set up values inside the bars
    for i, v in enumerate(data.iteritems()):
        ax.text(i, 50, "{:,}".format(v[1]), color='w', va='baseline',
                rotation=0, ha="center", fontsize=14)

    # Lines and yticks
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)

    # Output Image
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()
