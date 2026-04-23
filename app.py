import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le_brand = joblib.load("le_brand.pkl")
le_fuel = joblib.load("le_fuel.pkl")
le_trans = joblib.load("le_trans.pkl")

st.title("🚗 Car Price Prediction App")

brand = st.selectbox("Brand", le_brand.classes_)
fuel = st.selectbox("Fuel Type", le_fuel.classes_)
trans = st.selectbox("Transmission", le_trans.classes_)


model_year = st.number_input("Model Year")
engine_size = st.number_input("Engine Size")
mileage = st.number_input("Mileage")
doors = st.number_input("Doors")
owner_count = st.number_input("Owner Count")
horsepower = st.number_input("Horsepower")

if st.button("Predict Price"):

    brand_enc = le_brand.transform([brand])[0]
    fuel_enc = le_fuel.transform([fuel])[0]
    trans_enc = le_trans.transform([trans])[0]

    data = np.array([[brand_enc, fuel_enc, trans_enc,
                      model_year, engine_size, mileage,
                      doors, owner_count, horsepower]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    st.success(f"Estimated Price: {prediction:.2f} $")