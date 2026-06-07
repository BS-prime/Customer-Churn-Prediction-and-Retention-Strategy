# import modules
import sys
from pathlib import Path
from typing import Any

# import libraries
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# initiate config
config = load_config()

# locate root directory
ROOT_DIR = Path(__file__).resolve().parents[2]


# =================================================================================================
# --- 1. Perform train test split ---
# =================================================================================================


def _train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config["data"]["test_size"],
    random_state: int = config["data"]["random_state"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Helper function to split data into train and test
    :param X: pandas.DataFrame
    :param y: pandas Series
    :param test_size: float
    :param random_state: int
    :return: X_train, X_test, y_train, y_test
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=random_state
        )
        logging.info(
            f"Dataframe split into train and test: "
            f"{X_train.shape}, "
            f"{X_test.shape}, "
            f"{y_train.shape}, "
            f"{y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 2. Select features based on datatype ---
# =================================================================================================


def _select_datatypes_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Helper to select columns of numerical and categorical data types from a pd dataframe
    :param X: pd dataframe
    :return: categorical, cardinal, and numerical column lists
    """
    try:
        # select cardinal column
        card_columns = ["City"] if "City" in X.columns else []

        # select numerical columns
        num_columns = X.select_dtypes(include=["number"]).columns.tolist()
        num_columns = [col for col in num_columns if col not in card_columns]

        # select categorical columns
        cat_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_columns = [col for col in cat_columns if col not in card_columns]

        logging.info(f"Columns of numerical data: {num_columns}")
        logging.info(f"Columns of cardinal data: {card_columns}")
        logging.info(f"Columns of categorical data: {cat_columns}")

        return cat_columns, card_columns, num_columns

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 3. Create pipelines for different datatypes ---
# =================================================================================================


def _create_pipelines() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Helper function to create pipeline objects of numerical, categorical, and cardinal data
    :return: num_pipeline, cat_pipeline, card_pipeline
    """
    try:
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        logging.info(f"Steps of numerical pipeline: {num_pipeline.steps}")

        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first")),
            ]
        )

        logging.info(f"Steps of categorical pipeline: {cat_pipeline.steps}")

        card_pipeline = Pipeline([("target_encoder", ce.TargetEncoder(smoothing=10))])

        logging.info(f"Steps of cardinal pipeline: {card_pipeline.steps}")

        return num_pipeline, cat_pipeline, card_pipeline

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 4. Create preprocessor out of those pipelines ---
# =================================================================================================


def _create_preprocessor(
    num_pipeline: Pipeline,
    cat_pipeline: Pipeline,
    card_pipeline: Pipeline,
    cat_columns: list[str],
    num_columns: list[str],
    card_columns: list[str],
) -> ColumnTransformer:
    """
    Helper to combine all the pipelines to create a preprocessor
    :param num_pipeline: Pipeline
    :param cat_pipeline: Pipeline
    :param card_pipeline: Pipeline
    :return: Returns a ColumnTransformer object
    """
    try:
        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, num_columns),
                ("cat", cat_pipeline, cat_columns),
                ("card", card_pipeline, card_columns),
            ],
            sparse_threshold=0,
        )

        logging.info("Preprocessor initialized}")

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 5. Using preprocessor to transform the data ---
# =================================================================================================


def _initiate_data_transformation(
    preprocessor_obj: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config_data=config["output"],
) -> tuple[Any, Any, ColumnTransformer]:
    """
    Helper to transform the data, save features to csv files
    :param preprocessor_obj: ColumnTransformer object
    :param X_train: pandas DataFrame
    :param y_train: pandas Series
    :param X_test: pandas DataFrame
    :param y_test: pandas Series
    :param config_data: Path
    :return: transformed train array, transformed test array, and file path to preprocessor object
    """
    try:
        input_feature_train_arr = preprocessor_obj.fit_transform(X_train, y_train)

        input_feature_test_arr = preprocessor_obj.transform(X_test)

        # Concatenate transformed features with their respective target variables
        train_arr = np.c_[input_feature_train_arr, np.array(y_train)]
        test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

        logging.info(f"training features created: {train_arr.shape}")
        logging.info(f"testing features created: {test_arr.shape}")

        # save features
        train_dir = ROOT_DIR / Path(config_data["train_feature_path"]).parent
        train_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(train_arr).to_csv(
            train_dir / "train.csv", index=False, header=False
        )

        test_dir = ROOT_DIR / Path(config_data["test_feature_path"]).parent
        test_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(test_arr).to_csv(test_dir / "test.csv", index=False, header=False)

        return (train_arr, test_arr, preprocessor_obj)

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 6. Final execution ---
# =================================================================================================


def preprocess_data(X, y) -> tuple[Any, Any, Any, Any, Any]:
    """
    Using helpers to preprocess the data
    :param X: pandas Dataframe
    :param y: pandas Series
    :return:
    """
    X_train, X_test, y_train, y_test = _train_test_split(X, y)

    cat_columns, card_columns, num_columns = _select_datatypes_columns(X=X_train)

    num_pipeline, cat_pipeline, card_pipeline = _create_pipelines()

    preprocessor_obj = _create_preprocessor(
        num_pipeline=num_pipeline,
        cat_pipeline=cat_pipeline,
        card_pipeline=card_pipeline,
        card_columns=card_columns,
        num_columns=num_columns,
        cat_columns=cat_columns,
    )

    train_arr, test_arr, preprocessor_obj = _initiate_data_transformation(
        preprocessor_obj=preprocessor_obj,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # train test split the preprocessed data
    X_train = train_arr[:, :-1]
    y_train = train_arr[:, -1]

    X_test = test_arr[:, :-1]
    y_test = test_arr[:, -1]

    return X_train, X_test, y_train, y_test, preprocessor_obj
