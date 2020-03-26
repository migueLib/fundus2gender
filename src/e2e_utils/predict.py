# External libraries
from tqdm import tqdm
import torch
import numpy as np


def predict(model, device, loader):
    """
    model torchvision.models: Pre-trained model
    device torch.device: Device where the calculations will be performed
    [GPU|CPU]
    test_loader torch.utils.data.dataloader.DataLoader: input data
    """

    # Sets the module in evaluation mode.
    model.eval()

    # Initializing variables
    y_true = []
    y_pred = []
    f_path = []

    # Context-manager that disabled gradient calculation.
    with torch.no_grad():

        # replaced volatile=True in old version
        for data, labels, path in tqdm(loader):

            # Replaced .cuda() in old version
            data, labels = data.to(device), labels.to(device)

            # Model the data using inception v3
            try:
                outputs, aux = model(data)
            except ValueError:
                outputs = model(data)

            # Returns the maximum value of all elements in the input tensor.
            _, predicted = torch.max(outputs, 1)

            # Amount of corrected predicted labels
            y_true.extend(labels.cpu())
            y_pred.extend(predicted.cpu())
            f_path.extend(path)

    # Construct numpy array based on results
    y_true = np.asarray(y_true).astype(int, copy=True)
    y_pred = np.asarray(y_pred).astype(int, copy=True)

    # Return labels
    return y_true, y_pred, f_path
