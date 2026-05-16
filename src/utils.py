# import modules
import sys
import dill
from pathlib import Path

from exception import CustomException


def save_object(filepath: Path | str, obj):
    """
    This function saves an object directly at the root node directory
    if a filename or relative path is provided.
    :param filepath: Path | str
    :param obj: object
    :return: Nothing
    """
    try:
        # locate the root directory
        ROOT_DIR = Path(__file__).parent.parent

        # create the file object
        path_obj = Path(filepath)

        # check if the path is root
        if not path_obj.is_absolute():
            file_path = (ROOT_DIR / path_obj).resolve()
        else:
            file_path = path_obj.resolve()

        # 2. Extract and create the directory component cleanly
        dir_path = file_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        # 3. Save the binary file payload
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
