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

# locate the root directory
ROOT_DIR = Path(__file__).parent.parent.parent


def save_object(filepath: Path, obj: pickle):
    """
    This function saves object in a pickle file.
    :param filepath: str
    :param obj: object
    :return: Nothing
    """
    try:
        dir_name = ROOT_DIR / filepath.parent.name
        Path(dir_name).mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def set_seed(seed):
    """
    This function sets random seed.
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
