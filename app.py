import streamlit as st
import pandas as pd
import plotly.express as px

# Load Dataset
@st.cache
def load_data():
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

# Check if 'y_pred' column exists for predicted prices
if 'y_pred' not in brand_data.columns:
    st.warning("Predicted stock prices (y_pred) are not available in the dataset.")
else:
    # Reshape Data for Plotting
    plot_data = brand_data[['Date', 'Close', 'y_pred']].melt(id_vars=['Date'], 
                                                              var_name='Type', 
                                                              value_name='Price')

    # Plot with Plotly
    fig = px.line(plot_data, x='Date', y='Price', color='Type', 
                  title=f"Actual vs Predicted Prices for {selected_brand}",
                  labels={'Price': 'Stock Price', 'Date': 'Time'})

    # Display in Streamlit
    st.plotly_chart(fig)
st.subheader("📉 Adidas Actual vs Predicted Prices")
st.write(adidas_data.tail())  # Show predictions in Streamlit
