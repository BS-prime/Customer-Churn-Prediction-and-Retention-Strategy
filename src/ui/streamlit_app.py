from typing import Any

import streamlit as st
from pydantic import ValidationError

from src.api.schemas import CustomerData
from src.pipeline.predict_pipeline import ChurnPredictor


YES_NO_OPTIONS = ["No", "Yes"]
GENDER_OPTIONS = ["Male", "Female"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
LOYALTY_OPTIONS = ["new", "seeker", "floater", "loyal", "extra_loyal"]


@st.cache_resource
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor.load_artifacts()


def build_customer_data(form_values: dict[str, Any]) -> CustomerData:
    return CustomerData(**form_values)


def render_prediction(prediction: dict[str, float | bool | str]) -> None:
    probability_percent = prediction["churn_probability"] * 100
    profit = prediction["profit"]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Churn Probability", f"{probability_percent:.2f}%")
    metric_cols[1].metric("Expected Profit", f"${profit:,.2f}")
    metric_cols[2].metric(
        "Target Customer",
        "Yes" if prediction["should_target"] else "No",
    )

    if prediction["will_churn"]:
        st.warning("High churn risk")
    else:
        st.success("Low churn risk")


def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Predictor",
        layout="wide",
    )

    st.title("Customer Churn Predictor")

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Prediction artifacts could not be loaded: {e}")
        st.stop()

    with st.form("customer_churn_form"):
        left_col, middle_col, right_col = st.columns(3)

        with left_col:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.01,
                value=100.0,
                step=1.0,
            )
            total_charges = st.number_input(
                "Total charges",
                min_value=0.01,
                value=1000.0,
                step=10.0,
            )
            cltv = st.number_input(
                "CLTV",
                min_value=0.01,
                value=4226.0,
                step=50.0,
            )
            service_count = st.number_input(
                "Service count",
                min_value=0,
                max_value=9,
                value=1,
                step=1,
            )

        with middle_col:
            city = st.text_input("City", value="Los Angeles")
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            senior_citizen = st.selectbox("Senior citizen", YES_NO_OPTIONS)
            partner = st.selectbox("Partner", YES_NO_OPTIONS)
            dependents = st.selectbox("Dependents", YES_NO_OPTIONS)

        with right_col:
            contract = st.selectbox("Contract", CONTRACT_OPTIONS)
            paperless_billing = st.selectbox(
                "Paperless billing", YES_NO_OPTIONS, index=1
            )
            payment_method = st.selectbox("Payment method", PAYMENT_METHOD_OPTIONS)
            customer_loyalty = st.selectbox("Customer loyalty", LOYALTY_OPTIONS)

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        try:
            customer = build_customer_data(
                {
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "CLTV": cltv,
                    "ServiceCount": service_count,
                    "City": city,
                    "Gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "CustomerLoyalty": customer_loyalty,
                }
            )
            prediction = predictor.predict(customer.to_model_features())
            render_prediction(prediction)

        except ValidationError as e:
            st.error(f"Invalid customer input: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
