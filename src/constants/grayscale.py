# GLOBAL VARIABLES

# Grayscale image with standard normalization
# Grayscale without standard normalization
# https://en.wikipedia.org/wiki/Grayscale

WEIGHTS = {"normalized": (0.2989, 0.587, 0.114),
           "gray": (1 / 3, 1 / 3, 1 / 3),
           "triplicate": (0.0, 0.0, 1.0),
           "noise": (1.0, 1.0, 1.0)}

FORMATS = {"jpg": True,
           "JPG": True,
           "png": True,
           "PNG": True,
           "jpeg": True,
           "JPEG": True,
           "tif": True}

COLOR = {"R": 0,
         "G": 1,
         "B": 2}
