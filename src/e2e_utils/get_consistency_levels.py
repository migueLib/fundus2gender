# Local librares
from e2e_utils.logger import set_logger as sl

# External
import pandas as pd


def get_consistency_levels(df, classes, logger=None):
    """
    Calculates the consistency of prediction on a given
    DataFrame.
    """
    assert isinstance(df, pd.DataFrame), logger.error("Expected DataFrame")
    assert hasattr(classes, '__contains__'), logger.error("Expected Iterable")

    # Sort classes alphabetically for consistency
    classes = sorted(classes)

    # Get means of the dataframe for a classification
    df_labels = df.sum(axis=1)
    labels = sorted(df_labels.value_counts().index)

    # Generate new names for labels
    new_labels = dict()
    for i in labels:
        if i == min(labels):
            new_labels[i] = classes[0]
        elif i == max(labels):
            new_labels[i] = classes[-1]
        else:
            if max(i - max(labels), min(labels) - i) == min(labels) - i:
                new_labels[
                    i] = f"{classes[0]}{max(i - max(labels), min(labels) - i)}"
            else:
                new_labels[
                    i] = f"{classes[-1]}{max(i - max(labels), min(labels) - i)}"

    df_labels = df_labels.apply(lambda x: new_labels[x])
    return df_labels

