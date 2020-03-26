# Local libraries
from e2e_utils.logger import set_logger as sl

# Built in libraries
import os
import re
import warnings

# External libraries
import pandas as pd


def get_df_from_folder(folder, sub="", names=None, nosub=False, logger=None):
    # Test input types
    assert isinstance(folder, str), logger.error(
        f"folder is not the expected type: str")
    assert isinstance(sub, str), logger.error(
        f"sub is not the expected type: str")
    assert isinstance(nosub, bool), logger.error(
        f"nosub is not the expected type: bool")

    # If especific names provided then reverse the list
    try:
        names = list(reversed(names)) if names is not None else type(names)
    except TypeError:
        logger.error(f"Expecting iterable {names} found")
        raise

    df = pd.DataFrame()
    for file in os.listdir(folder):
        # Creating a regular expression
        rx_search = re.search(f"(.*){sub}(.*)", os.path.splitext(file)[0])

        # Only create the dataframe if a file complies with the regex
        if rx_search:
            with open(os.path.join(folder, file), "r") as FILE:
                # Remove substring from the name
                if nosub:
                    file = rx_search.group(1) + rx_search.group(2)

                # Using an specific list of names provided by the user
                try:
                    file = names.pop()
                except IndexError:
                    logger.debug(f"Empty list")
                    warnings.warn(
                        "Script will try to proceed on empty list for 'names'")
                except AttributeError:
                    pass

                # Concatenating DataFrame
                df = pd.concat([df, pd.read_csv(FILE, names=[file])], axis=1)

    del logger
    return df
