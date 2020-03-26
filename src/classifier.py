# Local libraries
from e2e_utils.predict import predict
from e2e_utils.logger import set_logger as sl
from constants.model import DATA_TRANSFORMS as DT
from e2e_utils.get_summary import get_summary
from e2e_utils.bootsrap import get_bootstrap
from e2e_utils.plot_histogram import plot_histogram
from classes.ImageFolderWithPaths import  ImageFolderWithPaths

# Standard libraries
import time
import argparse
from argparse import RawDescriptionHelpFormatter

# External libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models


def get_args():
    description = """
    eye2gene classifier allows the user to classify a folder of funduscopy
    images against a pre-trained inception v3 network
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawDescriptionHelpFormatter
                                     )
    # Required input
    required = parser.add_argument_group(title="Required arguments",
                                         description="Necessary arguments")
    required.add_argument("-m", "--model", dest="model", action="store",
                          required=False, help="Pathway to the model")
    required.add_argument("-i", "--img", dest="img", action="store",
                          required=False,
                          help="Folder to the  test set images")

    optional = parser.add_argument_group(title="Optional arguments",
                                         description="Optional arguments")
    optional.add_argument("-b", "--batch-size", dest="batch_size",
                          action="store", required=False, default=30,
                          type=int, help="Batch size to read files")
    optional.add_argument("-cn", "--classes", dest="classes", action="store",
                          required=False, help="""classes names, separated by a
                          coma. ie male,female """, default="female,male")
    optional.add_argument("-t", "--bootstrap", dest="bootstrap", action="store",
                          required=False, default=2000, help=""""Number of 
                          re-sampling iterations for bootstrapping""")
    optional.add_argument("-p", "-padding", dest="padding", required=False,
                          action="store_true", help="Boolean for padding")
    optional.add_argument("-o", "--output", dest="output", action="store",
                          required=False, default=False,
                          help="Pathway to output file")

    context = parser.add_argument_group(title="Contextual arguments",
                                        description="Contextual arguments")
    context.add_argument("-n", "--name", dest="name", required=False,
                         help="Name of the dataset")
    context.add_argument("-c", "--crop", dest="crop", required=False,
                         help="Size after center crop")
    context.add_argument("-ps", "--pad-size", dest="pad_size", required=False,
                         help="Padding Size")

    test = parser.add_argument_group(title="Test mode")
    test.add_argument("-T", "--TEST", dest="test", default=False,
                      action="store_true", help="run script in test mode.")

    # Check if personalized options are available
    args = parser.parse_args()

    if args.test:
        args = parser.parse_args([
        "-mD:\\Sciebo\\ukbb_fundus_images\\Inceptionv3_Models\\"
        "inceptionv3_norm_ukbb140219.pth",
        "-iD:\\Sciebo\\ukbb_fundus_images\\test_sets\\normalized\\ukbb_test_set",
        "-b30",
        "-cfemale,male",
        "-ntest",
        "-oC:\\Users\\darkg\\Downloads\\"])

    # Standardize paths
    # args.model = os.path.abspath(args.model)
    # args.data = os.path.abspath(args.data)
    args.batch_size = int(args.batch_size)

    # Get classes
    args.classes = args.classes.split(",")

    return args


################################################################################
def main(args):
    # Check if CUDA available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")

    # Load data (image folders)
    dataset = ImageFolderWithPaths(args.img, DT["finetune"])
    logger.info(f"Data folder loaded: {args.img}")

    # Load image folders by batch size
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4)
    logger.info(f"Data loaded with batch size {args.batch_size}")

    # Load model (inception v3) (same one as in the paper)
    model_ft = models.inception_v3(pretrained=False)
    logger.info(f"Model loaded {args.model}")

    # Number of inputs for your lineal layer
    n_feats = model_ft.fc.in_features
    logger.debug(f"Number of features: {n_feats}")

    # Apply a linear transformation to the incomming data
    model_ft.fc = nn.Linear(n_feats, len(args.classes))
    logger.info(f"Linear transformation done")
    logger.debug(model_ft.fc)

    # Send processing to device
    model_ft = model_ft.to(device)

    # Load a dictionary containing parameters and persistent buffers
    model_ft.load_state_dict(torch.load(args.model, map_location=device))

    # Prediction
    logger.info(f"Start classification")
    y_true, y_pred, paths = predict(model_ft, device, loader)
    logger.info(f"Classification finished on {len(y_pred)} images")
    logger.debug(f"True labels:      {y_true}")
    logger.debug(f"Predicted labels: {y_pred}")

    # Get summary
    male, female, acc, auc, summary, text = get_summary(y_true, y_pred, logger)

    # Bootstrap
    logger.info(F"Bootstrapping with {args.bootstrap} samples")
    acc, roc, accs, low_b, up_b = get_bootstrap(y_true, y_pred, seed=500,
                                                n_bootstraps=args.bootstrap)
    logger.info(f"ACC with 95% confidence interval: {acc * 100:.2f}% "
                f"({low_b * 100:.2f} - {up_b * 100:.2f})")

    # Plot histogram
    logger.info("Plotting histogram")
    plot_histogram(accs, args.bootstrap, out=args.output+"histogram.png")

    # Output results
    logger.info(f"Writing results to {args.output}")
    if args.output:
        # Summary
        with open(args.output+"summary.txt", "w") as SUMMARY:
            print(text, file=SUMMARY)
        # True labels
        with open(args.output+"labels_true.txt", "w") as TRUE:
            np.savetxt(TRUE, y_true, fmt="%d")
        # Predicted labels
        with open(args.output+"labels_predicted.txt", "w") as PREDICTED:
            np.savetxt(PREDICTED, y_pred, fmt="%d")
        # File Paths
        with open(args.output+"file_paths.txt", "w") as PATHS:
            for path in paths:
                print(path, file=PATHS)


if __name__ == '__main__':
    # This is only for testing purposes
    arg = get_args()

    # Set logger
    logger = sl("info")
    logger.info(f"""
    Data:  {arg.img}
    Model: {arg.model}
    Batch size: {arg.batch_size}
    Classes: {arg.classes}
    Name: {arg.name}
    """)

    # Run script
    st = time.time()
    main(arg)
    rt = time.time() - st
    logger.info(f"Run time: {rt//60:.0f}m {rt%60:.0f}s -- {rt:.2f}s")
