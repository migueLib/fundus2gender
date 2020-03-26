# External libraries
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def get_distance_matrix(data, title="", metric="jaccard"):
    """
    Calculates a similarity matrix using sklearn pairwise_distances

    """
    distance_matrix = 1-pairwise_distances(data.T.astype(bool), metric=metric)
    distance_matrix = pd.DataFrame(distance_matrix, index=data.columns,
                                   columns=data.columns)

    return distance_matrix
