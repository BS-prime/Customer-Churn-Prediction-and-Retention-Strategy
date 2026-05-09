import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    train_data_path: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    test_data_path: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Read the dataset
            df = pd.read_excel(self.ingestion_config.raw_data_path)
            logging.info('Read the dataset as dataframe')

            # Create the 'artifacts' directory
            self.ingestion_config.train_data_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train test data saved to csv")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
