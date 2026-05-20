# Import modules
from typing import Literal

# Import libraries
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    MonthlyCharges: float | int = Field(default=100, gt=0, title="Monthly Charges")
    TotalCharges: float | int = Field(default=1000, gt=0, title="Total Charges")
    CLTV: float | int = Field(default=4226, gt=0, title="CLTV")
    ServiceCount: int = Field(default=1, gt=0, title="Service count")
    City: Literal[
        "Los Angeles",
        "San Diego",
        "San Jose",
        "Sacramento",
        "San Francisco",
        "Fresno",
        "Long Beach",
        "Oakland",
        "Stockton",
        "Glendale",
        "Bakersfield",
        "Riverside",
    ] = "Los Angeles"
    Gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: Literal["Yes", "No"] = "No"
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["No", "Yes"] = "No"
    Contract: Literal["Month-to-month", "Two year", "One year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    CustomerLoyalty: Literal[
        "new",
        "seeker",
        "floater",
        "loyal",
        "extra_loyal",
    ] = "new"

    def to_model_features(self) -> dict:
        return {
            "Monthly Charges": self.MonthlyCharges,
            "Total Charges": self.TotalCharges,
            "CLTV": self.CLTV,
            "Service_count": self.ServiceCount,
            "City": self.City,
            "Gender": self.Gender,
            "Senior Citizen": self.SeniorCitizen,
            "Partner": self.Partner,
            "Dependents": self.Dependents,
            "Contract": self.Contract,
            "Paperless Billing": self.PaperlessBilling,
            "Payment Method": self.PaymentMethod,
            "customer_loyalty": self.CustomerLoyalty,
        }


class PredictionResponse(BaseModel):
    model_version: str
    churn_probability: float
    will_churn: bool
    profit: int | float
    should_target: bool
