# import modules
from pathlib import Path
from retention_strategy.rentention import retention_profit

# import libraries
import streamlit as st
import pandas as pd
import joblib

# locate root directory
ROOT_DIR = Path(__file__).parents[2]

preprocessor_path = ROOT_DIR.joinpath("artifacts", "preprocessor", "preprocessor.pkl")
preprocessor = joblib.load(preprocessor_path)

model_path = ROOT_DIR.joinpath("models", "best_model.pkl")
model = joblib.load(model_path)

st.title("Customer Churn Risk & Retention Advisor")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict"):

    sample = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "InternetService": internet
    }])

    prob = model.predict_proba(sample)[0, 1]
    profit = retention_profit(prob)

    st.metric("Churn Probability", f"{prob:.2%}")

    if profit > 0:
        st.success(f"Target customer — Expected Profit ${profit:.2f}")
    else:
        st.error("Do NOT target — Not profitable")
