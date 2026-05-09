import logging
import sys
from pathlib import Path
from datetime import datetime

from exception import CustomException

# 1. Locate the project root
PROJECT_DIR = Path(__file__).resolve().parents[1]

# 2. create the 'logs' DIRECTORY
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Create the folder, not the file

# 3. Define the full path to the log FILE
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE_NAME

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException(e, sys)
