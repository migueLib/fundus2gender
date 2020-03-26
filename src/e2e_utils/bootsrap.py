# External libraries
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from numpy.random import RandomState


def get_bootstrap(true, pred, n_bootstraps=2000, seed=None):
    """
    :param true: true label stored as numpy array
    :param pred: predicted score stored as numpy array
    :param n_bootstraps: Number of bootsraps.
    :param seed: Random seed for result reproducibility.
    :return:
    :return:
    """
    # Get accuracy and printing
    original_acc = accuracy_score(true, pred)
    original_roc = roc_auc_score(true, pred)

    # Generating random numbers from  seed for reproducibility
    rs = RandomState(seed)

    # Start bootstrapping, initialize bootstrap_accuracies
    btstrp_accs = []
    for i in range(n_bootstraps):

        # bootstrap by sampling with replacement on the prediction indices
        indices = rs.random_integers(0, len(pred)-1, len(pred))

        # We need at least one positive and one negative sample for ROC AUC
        # to be defined: reject the sample
        if len(np.unique(true[indices])) >= 2:
            btstrp_accs.append(accuracy_score(true[indices], pred[indices]))

    # obtain the 95 % CI from the results
    sorted_accuracies = np.array(btstrp_accs)
    sorted_accuracies.sort()

    # Get upper and lower bounds
    conf_low = sorted_accuracies[int(0.025 * len(sorted_accuracies))]
    conf_up = sorted_accuracies[int(0.975 * len(sorted_accuracies))]

    return original_acc, original_roc, sorted_accuracies, conf_low, conf_up
