# Import modules
import sys
from pathlib import Path

from components import threshold_optimization
# Import the functions
from components.data_ingestion import load_excel
from components.data_cleaning import data_cleaner
from components.data_preprocessing import preprocess_data
from components.feature_engineering import feature_engineering
from components.model_evaluation import evaluate_models
from components.model_trainer import model_trainer
from components.threshold_optimization import find_optimal_threshold

from src.config import load_config
from src.exception import CustomException
from src.logger import logging
from utils import save_object

# Locate the root directory
ROOT_DIR = Path(__file__).resolve().parents[2]


def run_training_pipeline():
    """
    Use different functions to run the training pipeline and save the models.
    :return: best model metrics
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
        X, y = feature_engineering(cleaned_df)

        # -----------------------------------------------------------------------------------------------
        # --- 4. Data Preprocessing ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Data Preprocessing")
        X_train, X_test, y_train, y_test, preprocessor_obj = preprocess_data(X=X, y=y)

        # -----------------------------------------------------------------------------------------------
        # --- 5. Save the preprocessor ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Saving Preprocessor")

        # Create model directory
        preprocessor_dir = ROOT_DIR / Path(config["output"]["preprocessor_path"])
        preprocessor_dir.parent.mkdir(parents=True, exist_ok=True)

        # save the preprocessor
        save_object(
            filepath=preprocessor_dir,
            obj=preprocessor_obj
        )

        logging.info(f"Preprocessor saved at: {preprocessor_dir}")

        # -----------------------------------------------------------------------------------------------
        # --- 5. Model Training ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Model Training")
        trained_models = model_trainer(X_train, y_train)

        # -----------------------------------------------------------------------------------------------
        # --- 6. Model Evaluation ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Model Evaluation")
        best_model, metrics = evaluate_models(trained_models, X_test, y_test)

        # -----------------------------------------------------------------------------------------------
        # --- 7. save the model ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Saving the model.")

        # Create model directory
        model_dir = ROOT_DIR / Path(config["output"]["model_path"])
        model_dir.parent.mkdir(parents=True, exist_ok=True)

        save_object(
            filepath=model_dir,
            obj=best_model
        )

        logging.info(f"Model saved at: {model_dir}")

        # -----------------------------------------------------------------------------------------------
        # --- 9. Threshold Optimization ---
        # -----------------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Threshold Optimization")

        threshold_info = find_optimal_threshold(model=best_model,
                                                x_test=X_test,
                                                y_test=y_test
                                                )

        # -----------------------------------------------------------------------------------------------
        # --- 9. Return statement ---
        # -----------------------------------------------------------------------------------------------

        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
