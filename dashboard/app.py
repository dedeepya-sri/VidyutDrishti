import streamlit as st
import requests

st.title("⚡ VidyutDrishti Dashboard")

temp = st.slider("Temperature", 20, 45, 30)
wind = st.slider("Wind Speed", 0, 15, 5)
irr = st.slider("Irradiance", 500, 1000, 800)
hum = st.slider("Humidity", 20, 80, 40)

if st.button("Predict"):
    url = f"http://127.0.0.1:8000/predict?temp={temp}&wind={wind}&irr={irr}&hum={hum}"
    response = requests.get(url).json()

    st.success(f"P50: {response['P50']} MW")
    st.info(f"P10: {response['P10']} MW")
    st.info(f"P90: {response['P90']} MW")