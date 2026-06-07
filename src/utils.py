# import modules
import sys
from pathlib import Path

import dill

from src.exception import CustomException


def save_object(filepath: Path | str, obj):
    """
    This function saves an object directly at the root node directory
    if a filename or relative path is provided.
    :param filepath: Path | str
    :param obj: object
    :return: Nothing
    """
    try:
        ROOT_DIR = Path(__file__).parent.parent

        path_obj = Path(filepath)

        if not path_obj.is_absolute():
            file_path = (ROOT_DIR / path_obj).resolve()
        else:
            file_path = path_obj.resolve()

        dir_path = file_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
