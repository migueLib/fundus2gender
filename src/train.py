# Local libraries
import argparse
# Built-in libraries
import os
import sys
import time
from argparse import RawDescriptionHelpFormatter as RaDeHeFo

import numpy as np
# External libraries
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import datasets, models

from constants.model import DATA_TRANSFORMS_TRAIN as DATA_TRANSFORMS
from e2e_utils.logger import set_logger as sl
from e2e_utils.show_model_samples import show_image
from e2e_utils.train_model import train_model
from e2e_utils.visualize_model import visualize_model


def get_args():
    description = """
    Trains a neural network using inceptionV3 
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RaDeHeFo)
    required = parser.add_argument_group(title="Required arguments",
                                         description="Necessary information to "
                                                     "run the script")
    required.add_argument("-m", "--model", dest="model", required=True,
                          help="Path to save model file")
    required.add_argument("-i", "--img", dest="img", required=True,
                          help="Folder from where to take images")
    
    optional = parser.add_argument_group(title="Optional arguments",
                                         description="Optional arguments with "
                                                     "predefined values")
    optional.add_argument("-p", "--pre-trained", dest="pre_trained",
                          required=False, default=False, action="store_true",
                          help="If you want to use a pre-trained model")
    optional.add_argument("-b", "--batch-size", dest="batch_size",
                          required=False, type=int, default=80,
                          help="Batch size to train the model")
    optional.add_argument("-e", "--epoch", dest="epoch", required=False,
                          type=int, default=10,
                          help="Number of Epochs to train the model")
    optional.add_argument("-l", "--learning-rate", dest="learning_rate",
                          required=False, type=float, default=0.01,
                          help="Learning rate of the model")
    optional.add_argument("-s", "--seed", dest="seed", required=False,
                          type=int, default=155,
                          help="Random seed, ensures reproducibility")
    optional.add_argument("-o", "--momentum", dest="momentum", required=False,
                          type=float, default=0.9, help="Momentum 4 optimizer")
    optional.add_argument("-lpm", "--load-pre-trained-model", dest="load_mod",
                          required=False, type=str, default=None,
                          help="Load previously generated model as a state "
                               "dictionary for and start point for this model")
    optional.add_argument("-c", "--criterion", dest="criterion", required=False,
                          default="CrossEntropyLoss", help="Select an specific loss function")

    args = parser.parse_args()

    return args


def main(args):
    # TODO: Split between training and validation set on script
    # Setting torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading data
    data_dir = os.path.abspath(args.img)
    logger.info(f"Training model on data {data_dir}")

    # Place to save the model
    mdl_path = os.path.abspath(args.model)
    logger.info(f"Path to the model {mdl_path}")

    # Load Image folders for training and finetune (validation)
    logger.info(f"Loading images")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              DATA_TRANSFORMS[x]) for x in
                      ["train", "finetune"]}

    data_loaders = {x: data.DataLoader(image_datasets[x], args.batch_size,
                                       shuffle=True, num_workers=4) for x in
                    ["train", "finetune"]}

    dataset_sizes = {x: len(image_datasets) for x in ["train", "finetune"]}
    class_names = image_datasets["train"].classes


    # Make a grid from batch/ export sub-sample image from the data being used
    # Get a batch of training data
    logger.info(f"Sub-sampling images")
    show_image(data_loaders["finetune"], class_names)

    # Initialize model
    # Check if model will be  pre-trained or not
    model_ft = models.inception_v3(pretrained=args.pre_trained)
    logger.info(f"Pre-trained model: {args.pre_trained}")
    
    # Get number of features
    num_features = model_ft.fc.in_features
    logger.info(f"Number of model features: {num_features}")
    
    # Set the output size of each sample
    logger.info(f"Output size: {len(class_names)}")
    model_ft.fc = nn.Linear(num_features, len(class_names))
    
    # Send processing to device
    model_ft = model_ft.to(device)

    # Use a previously trained model as a starting point for current one
    logger.info(f"Loading state_dict: {args.load_mod}")

    # TODO: Check if this part of the code works or not... but latter
    if args.load_mod is not None:
        logger.info(f"Loading state dictionary from previously trained model")
        model_ft.load_state_dict(torch.load(args.load_mod, map_location=device))

    # Choose criterion
    if args.criterion == "MSELoss":
        logger.info(f"Using MSELoss function")
        criterion = nn.MSELoss()
    elif args.criterion == "CrossEntropyLoss":
        logger.info(f"Using CrossEntropyLoss function")
        criterion = nn.CrossEntropyLoss()
    else:
        logger.info(f"Using BCELoss function")
        criterion = nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.learning_rate,
                             momentum=args.momentum)

    # Decay of learning rate by gamma every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Train the model baby
    logger.info(f"Training model")
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           model_save_path=args.model,
                           data_loaders=data_loaders,
                           dataset_sizes=dataset_sizes,
                           scheduler=scheduler,
                           num_epochs=args.epoch,
                           logger=logger)#,
                           #replace_w_noise=False,
                           #retain_channel=0)

    # Save a visualization of the model
    logger.info(f"Save visualization of the model")
    prefix = os.path.splitext(mdl_path)[0]
    visualize_model(model_ft, plot_name=prefix,  class_names=class_names,
                    device=device, data_loaders=data_loaders, num_images=8)


if __name__ == '__main__':
    # Calling in arguments
    arg = get_args()

    # Set logger
    logger = sl("info")
    logger.info(f"System version {sys.version}")

    # Logging CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA is available")
        logger.info(f"Training on gpu {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"Running script in CPU")

    # Set random seeds
    logger.info(f"Setting random seeds (Ensures reproducibility)")
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)

    # Run main script
    start_time = time.time()
    main(arg)
    run_time = time.time() - start_time
    logger.info(f"Run time: {run_time // 60:.0f}m {run_time % 60:.0f}s\n")
