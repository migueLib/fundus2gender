# Local libraries
from constants.grayscale import FORMATS as valid_formats
from constants.grayscale import COLOR as color_hash
from e2e_utils.get_grayscale import get_grayscale

# Standard libraries
import os
import glob
import time
from PIL import Image
from random import randint
import argparse
from argparse import RawDescriptionHelpFormatter as RDF

# External libraries
import numpy as np
import torch


def get_args():
    description = """
    convert_2grayscale allows the user to convert a folder of images to 
    greyscale images
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RDF)
    # Required Input
    required = parser.add_argument_group(title="Required arguments",
                                         description="Required arguments")
    required.add_argument("-i", "-input", dest="input", required=True,
                          help="Folder containing the images to process")
    required.add_argument("-o", "-output", dest="output", required=True,
                          help="Folder to output the images")

    # Optional Input
    optional = parser.add_argument_group(title="Optional arguments",
                                         description="No required arguments")
    optional.add_argument("-m", "-mode", required=False, default="gray",
                          choices=("gray","noise","normalized","triplicate"),
                          dest="mode", help="Type of grayscale mode")
    optional.add_argument("-c", "--color", required=False, default=None,
                          dest="color", choices=("R", "G", "B"),
                          help="Colour channel for triplicate and  "
                               "singleWithNoise mode conversion")

    args = parser.parse_args()

    return args


def main(in_dir, out_dir="grayscale", mode="gray", color=None, seed=369):
    """
    :param seed: Seed to generate random numbers, ensures reproducibility
    :param in_dir: folder containing the images
    :param out_dir: destination folder
    :param mode: Gray - Grayscale with normalization (default),
    Triplicate- Single channel tripled,  (otherwise)
     (Grayscale without normalization)
    :param color:colour options for triplicate mode,
    int 0 - R, int 1 - G, int 2 B, None - random channel chosen
    :return: None
    """
    # Uses seed to generate random state
    np.random.seed(seed)

    # Sets input path to allow recursive search
    path = str(in_dir) + "/**"

    # Set color to use (only triplicate and singleWnoise)
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Color channel used: {color if color is not None else 'random'}")
    color = color_hash.get(color, randint(0, 2))

    # Counter for logger
    n = 0

    # If output directory does not exist create it.
    if not os.path.isdir(out_dir):
        print(f"Creating new directory: {out_dir}")
        os.makedirs(out_dir)

    # Iterating over files recursively
    for f in glob.glob(path, recursive=True):

        # Check if the file format is valid
        try:
            assert valid_formats[f.split(".")[1]], \
                print(f"{f} Incompatible Format")
            n += 1
        except IndexError:
            print(f"Open directory: {f}")
            continue

        # Open image and convert it to numpy array
        img = Image.open(f)
        mimg = np.array(img)

        # Select mode to perform the grayscale conversion
        img_gray = get_grayscale(mimg, mode, color)

        # Transform np.array with the processed image data to image
        img = Image.fromarray(img_gray)

        # Use RGB mode
        img = img.convert(mode='RGB')

        # Create an out name for each subdirectory
        out_name = os.path.join(str(out_dir), os.path.relpath(f))
        sub_folder = os.path.join(str(out_dir),
                                  os.path.dirname(os.path.relpath(f)))

        # Create out sub-folders
        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)

        # Convert all images to png after the processing
        img.save(os.path.splitext(out_name)[0]+".png", format="png")

    print(f"{n} images converted to grayscale")


if __name__ == "__main__":
    # Calling in arguments
    arg = get_args()

    # Run converter and time it
    start_time = time.time()
    main(arg.input, arg.output, arg.mode, arg.color)
    run_time = time.time() - start_time
    print(f"Script run time: {run_time // 60:.0f}m {run_time % 60:.0f}s\n")
