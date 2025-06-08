import streamlit as st
import pandas as pd
import joblib
from fraud_detection import preprocess_data, load_data, evaluate_model
from geolocation import get_location_from_coordinates

st.title("Real-Time Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload credit card transactions CSV", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Sample Transactions:")
    st.dataframe(data.head())

    model_choice = st.selectbox("Select Model", ['Random Forest', 'XGBoost'])

    model_path = 'rf_model.pkl' if model_choice == 'Random Forest' else 'xgb_model.pkl'
    model = joblib.load(model_path)

    if st.button("Detect Fraud"):
        X = data.drop(['Class'], axis=1)
        preds = model.predict(X)
        data['Prediction'] = preds
        st.write("Prediction Results:")
        st.dataframe(data[['Prediction']])

        frauds = data[data['Prediction'] == 1]
        st.success(f"{len(frauds)} fraudulent transactions detected.")

        if 'lat' in data.columns and 'lon' in data.columns:
            st.write("Fraudulent Transactions Locations:")
            for i, row in frauds.iterrows():
                address = get_location_from_coordinates(row['lat'], row['lon'])
                st.write(f"Location: {address}")
