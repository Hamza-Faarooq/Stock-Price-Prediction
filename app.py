import streamlit as st
import pandas as pd

# Load Dataset
@st.cache
def load_data():
    # Replace with your dataset path
    data = pd.read_csv('World-Stock-Prices-Dataset.csv')
    data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_localize(None)
    return data

data = load_data()

# Title and Description
st.title("Stock Price Prediction Dashboard")
st.write("Explore stock price trends and predictions for different brands.")

# Brand Selection
selected_brand = st.selectbox("Select a Brand", data['Brand_Name'].unique())
brand_data = data[data['Brand_Name'] == selected_brand]

# Plot Stock Prices Over Time
st.subheader(f"Stock Prices Over Time for {selected_brand}")
st.line_chart(brand_data[['Date', 'Close']].set_index('Date'))
