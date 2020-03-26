# Local
from constants.grayscale import WEIGHTS as gsw

# External
import numpy as np
import torch


def get_grayscale(mimg, mode, color):
    if mode == "noise":
        # Choose torch device (CUDA if available CPU otherwise)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create empty matrix with pytorch
        x = torch.zeros(mimg.shape, device=device)
        n = torch.randn(mimg[..., color].shape, device=device)*255

        # Fill the matrix
        for i in range(3):
            x[:, :, i] = torch.from_numpy(mimg[..., color]) if i == color else n

        # Move tensor to CPU and transform it to Numpy Array
        x = x.cpu().numpy()

    elif mode == "triplicate":
        # Triplicate single channel image.
        x = np.sum((gsw[mode][i] * mimg[..., color] for i in range(3)), axis=0)
    else:
        # Gray or normalized
        x = np.sum((gsw[mode][i] * mimg[..., i] for i in range(3)), axis=0)

    return x.astype('uint8')
