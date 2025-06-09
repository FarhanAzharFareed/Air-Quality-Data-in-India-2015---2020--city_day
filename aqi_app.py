# AQI Streamlit Web App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("🌫️ Air Quality Index (AQI) Prediction App")

uploaded_file = st.file_uploader("📤 Upload your 'city_day.csv' file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Raw Data Preview")
    st.write(df.head())

    df.fillna(df.median(numeric_only=True), inplace=True)
    df = df.dropna(subset=['AQI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['City_Code'] = df['City'].astype('category').cat.codes

    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                'Benzene', 'Toluene', 'Xylene', 'City_Code', 'Year', 'Month', 'Day']
    target = 'AQI'

    df_sorted = df.sort_values(by=['City', 'Date'])
    X = df_sorted[features]
    y = df_sorted[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader(" Model Evaluation")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    st.subheader(" Forecasting Graph")
    city_selected = st.selectbox("Select a City", df['City'].unique())
    future_df = df[df["City"] == city_selected].sort_values("Date")
    future_X = future_df[features]
    future_pred = model.predict(future_X)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(future_df['Date'], future_df['AQI'], label='Actual AQI', alpha=0.7)
    ax.plot(future_df['Date'], future_pred, label='Predicted AQI', linestyle='--', color='red')
    ax.set_title(f"AQI Forecasting for {city_selected}")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning(" Please upload a valid 'city_day.csv' file to proceed.")
