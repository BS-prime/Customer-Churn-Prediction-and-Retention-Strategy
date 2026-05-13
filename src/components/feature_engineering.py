# import modules
import sys

from src.exception import CustomException
from src.logger import logging
from src.config import load_config

# import libraries
import pandas as pd
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# load the config
config = load_config()


# =================================================================================================
# --- 1. Separate input and output feature ---
# =================================================================================================

def _input_output_split(df: pd.DataFrame):
    """
    Helper to split data into input and output features
    :param df: pandas.DataFrame
    :return: the input and output dataframe
    """

    try:
        X = df.drop("Churn Value", axis=1)
        y = df["Churn Value"]
        logging.info(f"Dataframe split into input and output feature: {df.shape}, {y.shape}")

        return X, y

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 2. Drop redundant features ---
# =================================================================================================

def _drop_feature(X: pd.DataFrame):
    """
    Helper function to drop redundant features for model training
    :param X: pd dataframe
    :return: pd dataframe
    """

    try:
        X = X.drop(columns=[
            "Churn Label",
            "Churn Reason"
        ])

        return X

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 3. Replace the services columns with number of services ---
# =================================================================================================

def _refactor_services_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to create a new feature from how many services the customers have.
    :param X: pd.DataFrame
    :return: pd.DataFrame
    """

    # get the services feature
    services_feature = [
        'Phone Service',
        'Multiple Lines',
        'Internet Service',
        'Online Security',
        'Online Backup',
        'Device Protection',
        'Tech Support',
        'Streaming TV',
        'Streaming Movies'
    ]

    # Create a new service count feature
    X["Service_count"] = (X[services_feature] != "No").sum(axis=1)

    # drop this redundant columns
    X = X.drop(columns=services_feature)

    return X


# =================================================================================================
# --- 4. Refactor tenure month column into categories ---
# =================================================================================================

def _refactor_tenure_month_feature(X: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to refactor tenure month feature into categories
    :param X: pd dataframe
    :return: pd dataframe
    """

    X["customer_loyalty"] = pd.cut(x=X["Tenure Months"],
                                   bins=[0, 1, 2, 10, 29, 72],
                                   labels=["new", "seeker", "floater", "loyal", "extra_loyal"]
                                   )

# =================================================================================================
# --- 4. Perform train test split ---
# =================================================================================================

def _train_test_split(
        df: pd.DataFrame,
        y: pd.Series,
        test_size: float = config["data"]["test_size"],
        random_state: int = config["data"]["random_state"]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Helper function to split data into train and test
    :param df: pandas.DataFrame
    :param y: pandas Series
    :param test_size: float
    :param random_state: int
    :return:
    """

    try:
        X_train, X_test, y_train, y_test = train_test_split(df,
                                                            y,
                                                            stratify=y,
                                                            test_size=test_size,
                                                            random_state=random_state
                                                            )
        logging.info(f"Dataframe split into train and test: "
                     f"{X_train.shape}, "
                     f"{X_test.shape}, "
                     f"{y_train.shape}, "
                     f"{y_test.shape}"
                     )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 6. Select features based on datatype ---
# =================================================================================================

def _select_datatypes_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Helper to select columns of numerical and categorical data types from a pd dataframe
    :param df: pd dataframe
    :return: pd dataframe
    """

    try:
        df = df.copy()

        # select the features according to datatype
        categorical_columns = df.select_dtypes(include="object").columns.drop(["City"])
        cardinal_columns = ["City"]
        numerical_columns = df.select_dtypes(include="number").columns

        logging.info(f"Columns of numerical data: {numerical_columns}")
        logging.info(f"Columns of categorical data: {cardinal_columns}")
        logging.info(f"Columns of categorical data: {categorical_columns}")

        return categorical_columns, cardinal_columns, numerical_columns


    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 7. Create pipelines for different datatypes ---
# =================================================================================================

def _create_pipelines() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Helper function to create pipeline objects of numerical and categorical data    :return:
    """

    try:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        logging.info(f"Columns of numerical data: {num_pipeline.steps}")

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first")),
        ])

        logging.info(f"Columns of categorical data: {cat_pipeline.steps}")

        card_pipeline = Pipeline([
            ("target_encoder", ce.TargetEncoder(smoothing=10))
        ])

        return num_pipeline, cat_pipeline, card_pipeline

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 8. Create preprocessor out of those pipelines ---
# =================================================================================================

def _create_preprocessor(
        num_pipeline: Pipeline,
        cat_pipeline: Pipeline,
        card_pipeline: Pipeline,
        categorical_columns: list[str],
        numerical_columns: list[str],
        card_columns: list[str],
) -> ColumnTransformer:
    """
    Helper function to combine all the pipelines to create a preprocessor
    :param num_pipeline: Pipeline
    :param cat_pipeline: Pipeline
    :param card_pipeline: Pipeline
    :return: Returns a ColumnTransformer object
    """

    try:
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_columns),
            ("cat", cat_pipeline, categorical_columns),
            ("card", card_pipeline, card_columns),
        ])

        logging.info(f"Steps of preprocessor: {preprocessor.steps}")

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)
