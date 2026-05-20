import pandas as pd

from src.components.data_cleaning import data_cleaner


def test_data_cleaner_converts_total_charges_and_drops_columns():
    df = pd.DataFrame(
        {
            "Total Charges": [" ", "12.5"],
            "Count": [1, 1],
            "Country": ["United States", "United States"],
            "State": ["California", "California"],
            "CustomerID": ["A", "B"],
            "Monthly Charges": [10.0, 20.0],
        }
    )

    cleaned_df = data_cleaner(df)

    assert cleaned_df["Total Charges"].tolist() == [0.0, 12.5]
    assert cleaned_df["Total Charges"].dtype == float
    assert {"Count", "Country", "State", "CustomerID"}.isdisjoint(cleaned_df.columns)
