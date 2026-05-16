# import modules
import sys
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.config import load_config

# Import the functions
from components.data_ingestion import load_excel
from components.data_cleaning import data_cleaner
from components.feature_engineering import feature_engineering
from components.model_trainer import model_trainer
from components.model_evaluation import evaluate_models
from utils import save_object

# Locate the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def run_training_pipeline():
    """
    Use different functions to run the training pipeline
    :return: best model with metrics
    """
    try:
        config = load_config()

        # -----------------------------------------------------------------------------------------------
        # --- 1. Data Ingestion ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Data Ingestion")
        df = load_excel()

        # -----------------------------------------------------------------------------------------------
        # --- 2. Data Cleaning ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Data Cleaning")
        cleaned_df = data_cleaner(df)

        # -----------------------------------------------------------------------------------------------
        # --- 3. Feature Engineering ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Feature Engineering")
        X_train, X_test, y_train, y_test = feature_engineering(cleaned_df)

        # -----------------------------------------------------------------------------------------------
        # --- 4. Model Training ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Model Training")
        trained_models = model_trainer(X_train, y_train)

        # -----------------------------------------------------------------------------------------------
        # --- 5. Model Evaluation ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Model Evaluation")
        best_model, metrics = evaluate_models(trained_models, X_test, y_test)

        # -----------------------------------------------------------------------------------------------
        # --- 6. save the model ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Saving the model.")

        # Create model directory
        file_dir = ROOT_DIR / Path(config["output"]["model_path"])
        file_dir.parent.mkdir(parents=True, exist_ok=True)

        save_object(
            filepath=ROOT_DIR / config["output"]["model_path"],
            obj=best_model
        )

        logging.info("Training Pipeline executed successfully!")

        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
