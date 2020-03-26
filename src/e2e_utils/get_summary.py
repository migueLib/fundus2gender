# Local libraries
from e2e_utils.get_class_ratio import get_class_ratio

# External libraries
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def get_summary(true, pred, logger=None):
    # First line of summary
    text = list()
    text.append("#############################################################")

    # First label ratio
    male = get_class_ratio(true, pred, logger=logger)
    text.append(f"Percentage of individuals classified as Male: {male:.2f}%")
    logger.debug(text[-1])

    # Second label ratio
    female = get_class_ratio(true, pred, reverse=True, logger=logger)
    text.append(f"Percentage of individuals classified as Female {female:.2f}%")
    logger.debug(text[-1])

    # Accuracy
    acc = accuracy_score(true, pred)
    text.append(f"Accuracy of the model {100*acc:.2f}")
    logger.debug(text[-1])

    # AUC
    auc = roc_auc_score(true, pred)
    text.append(f"Area Under the ROC Curve: {100*auc:.2f}")
    logger.debug(text[-1])

    # Summary
    summary = classification_report(true, pred)
    text.append(f"\n{summary}")
    logger.debug(text[-1])

    # Concatenate text
    text = "\n".join(text)

    return male, female, acc, auc, summary, text
