# Local libraries


def get_class_ratio(real, predictions, reverse=False, logger=None):
    # Get length of real and predicted labels
    total = real.size
    n_pre = sum(predictions)

    # If reverse then we are assessing for the other class
    if reverse:
        ratio = 100*(total - n_pre)/total
        logger.debug(f"get_class_ratio {total - n_pre}/{total} = {ratio:.2f}")
        return ratio
    else:
        ratio = 100*n_pre/total
        logger.debug(f"get_class_ratio {n_pre}/{total} = {ratio:.2f}")
        return ratio
