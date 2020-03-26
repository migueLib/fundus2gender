# Standard Libraries
import os
from os.path import abspath, join
import re

# External Libraries
import pandas as pd

# Local Libraries
from e2e_utils.logger import set_logger as sl

###########################################################################
# Set up logger
logg = sl(level="debug")

##############################################################################
# Set working directory
ROOT = abspath("D:\\Sciebo\\ukbb_fundus_images\\results_by_images")
logg.info(f"Set main directory at {ROOT}")

# Load data
# Iterate over files in ROOT folder and get df for paths and labels
labels_df = pd.concat([pd.read_csv(join(ROOT, file), names=[file.split(".")[0]])
                for file in os.listdir(ROOT) if re.match(".+label.?", file)], axis=1)

paths_df = pd.concat([pd.read_csv(join(ROOT, file), names=[file.split(".")[0]])
                for file in os.listdir(ROOT) if re.match(".+paths.?", file)], axis=1)

# Getting "true labels"
true = labels_df[[c for c in labels_df.columns if "true" in c]].mean(axis="columns").astype(int)
logg.debug(f"Are reference labels consistent: {all((true == 0) | (true == 1))}")

# Drop "true labels"
labels_df.drop([c for c in labels_df.columns if "true" in c], axis="columns", inplace=True)
logg.debug(f"Dropped the reference labels from df")

# Rename labels and paths df
labels_df.rename({c: c[:-10] for c in labels_df.columns if "predicted" in c}, axis="columns", inplace=True)
paths_df.rename({c: c[:-5] for c in paths_df.columns if "file" in c}, axis="columns", inplace=True)

logg.debug(f"Processed images: {labels_df.shape[0]}")
logg.debug(f"Processed modes : {labels_df.shape[1]}")
logg.info(f"Finish file importation")



# "Clean"  Column names
