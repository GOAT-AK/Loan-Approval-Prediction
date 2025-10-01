import streamlit as st
import pickle
import pandas as pd


with open("Loan Approval Prediction/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Loan Approval Prediction/scaler_loan.pkl", "rb") as f:
    scaler = pickle.load(f)


FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value"
]


st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="centered")
st.title("💳 Loan Approval Prediction App")
st.write("This app uses a Logistic Regression model trained with SMOTE and scaling to predict loan approval status.")


def user_input_features():
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    income_annum = st.number_input("Annual Income", min_value=0, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
    loan_term = st.number_input("Loan Term (in months)", min_value=0, step=6)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=1000)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=1000)

    features = {
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value
    }
    
    return pd.DataFrame([features])


input_df = user_input_features()


if st.button("Predict Loan Status"):
    
    input_df = input_df[FEATURES]

   
    input_scaled = scaler.transform(input_df)
    
    
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {prediction_proba[0]*100:.2f}%)")

    st.write("Approval Probability:", f"{prediction_proba[1]*100:.2f}%")
    st.write("Rejection Probability:", f"{prediction_proba[0]*100:.2f}%")
