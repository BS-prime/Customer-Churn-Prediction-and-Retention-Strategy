# import libraries
import numpy as np
import pandas as pd


def _null_value_handling(dataframe: pd.DataFrame):
    """
    fill null values in dataframe
    :param dataframe:
    :return: pandas DataFrame
    """

    # fill null values in Churn Reason with N/A
    dataframe["Churn Reason"] = dataframe["Churn Reason"].fillna("N/A")

    # replace the null values with median
    dataframe["Total Charges"] = dataframe["Total Charges"].replace(" ", np.nan)
    dataframe["Total Charges"] = dataframe["Total Charges"].replace(" ", dataframe["Total Charges"].median())

    return dataframe


def _fixing_datatype(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fixing datatypes in dataframe
    :param dataframe:
    :return:
    """

    # change the str datatype of each values
    str_cols = dataframe.select_dtypes(include="str").columns
    dataframe[str_cols] = dataframe[str_cols].astype(object)

    # Change the data type of the Total Charges feature
    dataframe["Total Charges"] = dataframe["Total Charges"].astype(float)

    return dataframe


def _drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns in dataframe
    :param dataframe:
    :return:
    """

    # Drop the redundant columns in the dataframe
    dataframe = dataframe.drop(
        columns=[
            "Count",
            "Country",
            "State",
            "CustomerID",
            "Lat Long",
            "Latitude",
            "Longitude",
            "Zip Code",
        ]
    )
    return dataframe


def data_cleaner(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe with helper functions
    :param dataframe:
    :return: the cleaned dataframe
    """

    dataframe = _null_value_handling(dataframe)
    dataframe = _fixing_datatype(dataframe)
    dataframe = _drop_columns(dataframe)

    return dataframe
