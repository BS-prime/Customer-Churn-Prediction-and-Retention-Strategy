# import modules
import pickle
from pathlib import Path
import sys
import dill
import random
import os
from exception import CustomException

# import libraries
import numpy as np


def save_object(filepath: Path, obj: pickle):
    """
    This function saves object in a pickle file.
    :param filepath:
    :param obj:
    :return: Nothing
    """
    try:
        dir_name = Path(filepath.parent).name
        Path(dir_name).mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
