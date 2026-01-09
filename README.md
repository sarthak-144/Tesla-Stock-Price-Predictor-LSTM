# 📈 Tesla Stock Price Predictor

A web-based machine learning application that predicts the future closing price of Tesla (TSLA) stock. This project uses a **Long Short-Term Memory (LSTM)** neural network trained on historical stock data to forecast trends and visualize performance.



[Image of Stock Market Analysis Chart]


## 🚀 Live Demo
[Click here to view the App]([YOUR_STREAMLIT_APP_LINK_HERE](https://tesla-stock-price-predictor-lstm.streamlit.app/))

## 🧠 Model Architecture
The core of this project is a Deep Learning model built with **TensorFlow/Keras**:
* **Type:** Recurrent Neural Network (RNN) - LSTM
* **Input:** Past 60 days of stock closing prices
* **Layers:** * LSTM Layer (50 units, return sequences)
    * LSTM Layer (50 units)
    * Dense Layer (25 units)
    * Output Layer (1 unit)
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)

## 🛠️ Technologies Used
* **Python**: Core programming language.
* **Streamlit**: For the web interface and interactivity.
* **TensorFlow & Keras**: For building and training the LSTM model.
* **Pandas & NumPy**: For data manipulation and preprocessing.
* **Matplotlib**: For visualization of stock trends and predictions.
* **Scikit-Learn**: For data normalization (MinMaxScaler).

## 📂 Project Structure
```text
├── app.py                      # The main Streamlit application
├── stock_prediction_model.keras # The saved pre-trained LSTM model
├── scaler.pkl                  # The saved MinMaxScaler object
├── Tasla_Stock.csv             # Historical stock data (Dataset)
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation

```

## 💻 How to Run Locally

1. **Clone the repository:**
```bash
git clone [https://github.com/sarthak-144/Tesla-Stock-Price-Predictor-LSTM.git](https://github.com/sarthak-144/Tesla-Stock-Price-Predictor-LSTM.git)
cd Tesla-Stock-Price-Predictor-LSTM

```


2. **Create a virtual environment (Optional but Recommended):**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run the application:**
```bash
streamlit run app.py

```



## 📊 Features

* **Raw Data Visualization:** View the historical dataframe of Tesla stock.
* **Interactive Charts:** Dynamic line charts showing closing prices over time.
* **Prediction vs Actual:** Visual comparison of the model's predictions against real market data.
* **Future Forecasting:** Predicts the *next* trading day's price based on the most recent data.

## 📝 Dataset

The model was trained on historical data ranging from **2015 to 2024**. The dataset (`Tasla_Stock.csv`) includes Open, High, Low, Close, and Volume columns, though the model focuses exclusively on the 'Close' price.


## 📜 License

This project is open-source and available under the [MIT License](LICENSE)
