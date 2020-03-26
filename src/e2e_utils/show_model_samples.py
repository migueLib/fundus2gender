# Built-in
import datetime

# External
import matplotlib.pyplot as plt
import torchvision
import numpy as np



def show_image(data_loader, class_names, plot_prefix=None, title=None, save=True):
    
    # From the data loaders get inputs and classes
    inputs, classes = next(iter(data_loader))
    inputs, classes = inputs[0:8], classes[0:8]

    # Get image sub-sample
    inp = torchvision.utils.make_grid(inputs)
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)

    # Set title    
    if title is  None:
        title = "            ".join([class_names[x] for x in classes])

    # Configure canvas
    plt.figure(figsize=(12,3))
    plt.imshow(inp)
    plt.axis("off")
    plt.title(title)

    # Check plot prefix
    if plot_prefix is None:
        plot_prefix = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')

    # Save file
    if save:
        plt.savefig(f"{plot_prefix}_original_img_norm.png", bbox_inches='tight')
    else:
        plt.show()
