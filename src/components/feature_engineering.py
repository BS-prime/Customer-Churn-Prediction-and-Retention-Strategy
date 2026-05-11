import numpy as np


class FeatureEngineering:
    def __init__(self):
def _null_value_handling(dataframe: pd.DataFrame) -> pd.DataFrame:
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