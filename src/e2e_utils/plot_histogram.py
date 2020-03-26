import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data, n, out=None):
    # Setting plot background
    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plotting distribution
    sns.distplot(data, ax=ax)
    plt.title(f"Distribution on {n} bootstrapped samples")

    # Show histogram or save it to a file
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()
