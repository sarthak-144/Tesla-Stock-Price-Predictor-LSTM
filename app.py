import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# --- UI Layout ---
st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")
st.title("📈 Tesla Stock Price Predictor (LSTM)")

# --- 1. Load Data ---
# We use st.cache_data so we don't reload the file every time the app refreshes
@st.cache_data
def load_data():
    # Make sure 'Tasla_Stock.csv' is in the same folder
    df = pd.read_csv('Tasla_Stock.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(df.tail())

# Plot Raw Data
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Date, df.Close, label='Close Price')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
st.pyplot(fig)


# --- 2. Prepare Data for Model ---
# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Filter only Close column
data_close = df.filter(['Close'])
dataset = data_close.values

# Scale the data
scaled_data = scaler.transform(dataset)

# Define time_step (must match the trained model)
time_step = 60 

# Split into X_test (Simulation)
# We will just predict on the last 20% of the data to show performance
training_data_len = int(np.ceil(len(dataset) * 0.8))

test_data = scaled_data[training_data_len - time_step: , :]
x_test = []
y_test = dataset[training_data_len:, :] # Actual values

for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# --- 3. Load Model & Predict ---
st.subheader('Model Predictions')
try:
    model = load_model('stock_prediction_model.keras')
    
    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate validation data for plotting
    train = data_close[:training_data_len]
    valid = data_close[training_data_len:]
    valid['Predictions'] = predictions

    # Plot
    fig2 = plt.figure(figsize=(16, 8))
    plt.title('Model Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df['Date'][:training_data_len], train['Close'], label='Training Data')
    plt.plot(df['Date'][training_data_len:], valid['Close'], label='Actual Value')
    plt.plot(df['Date'][training_data_len:], valid['Predictions'], label='Predicted Value')
    plt.legend(loc='lower right')
    st.pyplot(fig2)
    
    # Show the actual vs predicted numbers
    st.write("Compare Actual vs Predicted Prices:")
    st.write(valid.tail())

except Exception as e:
    st.error(f"Error loading model: {e}. Did you save the model first?")


# --- 4. Future Prediction (Bonus) ---
st.subheader("🔮 Predict Next Day's Price")

# Get the last 60 days of data
last_60_days = data_close[-60:].values
# Scale it using the same scaler
last_60_days_scaled = scaler.transform(last_60_days)
# Reshape
X_test_new = []
X_test_new.append(last_60_days_scaled)
X_test_new = np.array(X_test_new)
X_test_new = np.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], 1))

# Predict
pred_price = model.predict(X_test_new)
pred_price_unscaled = scaler.inverse_transform(pred_price)

st.success(f"Based on the last 60 days, the predicted price for the next trading day is: **${pred_price_unscaled[0][0]:.2f}**")