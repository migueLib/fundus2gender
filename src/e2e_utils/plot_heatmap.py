# Local libraries
# Built-in libraries
# External libraries
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(data, title="", out=None):
    # Print heat-map of the results
    sns.set(style="white")

    # Setting axes
    f, ax = plt.subplots(figsize=(12, 12))

    # Ploting with seaborn
    sns.heatmap(data, annot=True, linewidths=0, ax=ax, annot_kws={"size": 16})

    # Set pretty labels
    ax.set_yticklabels(data.index, rotation=55, fontsize=14)
    ax.set_xticklabels(data.index, rotation=55, fontsize=14)
    ax.set_title(title, fontsize=20)

    # Output Image
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()
