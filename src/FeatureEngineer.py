import pandas as pd


class FeatureEngineer:
    def transform(self, df):
        df = df.copy()

        df["tenure_group"] = pd.cut(
            df["Tenure Month"],
            bins=[0, 12, 36, 60, 100],
            labels=["new", "mid", "loyal", "very_loyal"]
        )

        service_cols = [
            "Phone Service", "Multiple Lines", "Internet Service",
            "Online Security", "Online Backup", "Device Protection",
            "Tech Support", "Streaming TV", "Streaming Movies"
        ]

        df["service_count"] = (df[service_cols] != "No").sum(axis=1)

        df.drop([
            "Phone Service", "Multiple Lines", "Internet Service",
            "Online Security", "Online Backup", "Device Protection",
            "Tech Support", "Streaming TV", "Streaming Movies"
        ], axis=1, inplace=True)

        return df
