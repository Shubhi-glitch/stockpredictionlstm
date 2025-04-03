import argparse
import copy
import numpy as np
import os
import random
import tensorflow as tf
import streamlit as st
from datetime import date
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

tf.compat.v1.disable_eager_execution()

def leaky_relu(features, alpha=0.2, name=None):
    return tf.maximum(alpha * features, features)

class AWLSTM:
    def __init__(self, data, parameters, seq=5, epochs=50, batch_size=256, gpu=False, att=1, hinge=1, adv=0):
        self.data = data
        self.paras = copy.copy(parameters)
        self.seq, self.epochs, self.batch_size, self.gpu = seq, epochs, batch_size, gpu
        self.att, self.hinge, self.adv_train = bool(att), bool(hinge), bool(adv)
        self.preprocess_data()

    def preprocess_data(self):
        prices = self.data['Close'].dropna().values
        n = len(prices)
        tra_end, val_end = int(n * 0.7), int(n * 0.85)

        def create_sequences(data):
            return np.array([data[i:i+self.seq] for i in range(len(data) - self.seq)]), \
                   np.array([data[i+self.seq] for i in range(len(data) - self.seq)])

        self.tra_pv, self.tra_gt = create_sequences(prices[:tra_end])
        self.val_pv, self.val_gt = create_sequences(prices[tra_end:val_end])
        self.tes_pv, self.tes_gt = create_sequences(prices[val_end:])

        self.tra_pv, self.val_pv, self.tes_pv = map(lambda x: x.reshape(-1, self.seq, 1), [self.tra_pv, self.val_pv, self.tes_pv])
        self.tra_gt, self.val_gt, self.tes_gt = map(lambda x: x.reshape(-1, 1), [self.tra_gt, self.val_gt, self.tes_gt])

    def construct_graph(self):
        device_name = '/gpu:0' if self.gpu else '/cpu:0'
        with tf.device(device_name):
            tf.compat.v1.reset_default_graph()

            self.gt_var = tf.compat.v1.placeholder(tf.float32, [None, 1])
            self.pv_var = tf.compat.v1.placeholder(tf.float32, [None, self.paras['seq'], 1])

            lstm_layer = tf.keras.layers.LSTM(self.paras['unit'], return_sequences=False)
            outputs = lstm_layer(self.pv_var)

            self.pred = tf.keras.layers.Dense(1, activation=None)(outputs)
            self.loss = tf.compat.v1.losses.mean_squared_error(self.gt_var, self.pred)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.paras['lr']).minimize(self.loss)

    def predict(self, future_dates):
        self.construct_graph()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            last_sequence = self.tes_pv[-1:].copy()
            predictions = []
            for _ in future_dates:
                pred = sess.run(self.pred, feed_dict={self.pv_var: last_sequence})
                predictions.append(pred[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred[0, 0]
        return np.array(predictions)

# Streamlit App
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="STOCKLENS", page_icon="📉", layout="centered")
st.markdown("<h3 style='text-align: center; color: #4CAF50; font-size: 40px;'>Stock Prediction App</h3>", unsafe_allow_html=True)

stocks = ("AAPL", "GOOG", "MSFT", "GME", "NVDA", "TSLA", "BTC-USD", "ETH-USD", "META", "JPM", "AMZN")
selected_stocks = st.selectbox("Select stock:", stocks)
n_years = st.slider("Prediction years:", 1, 4, key='slider', help="Select years to forecast.")
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data.reset_index()

data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close', line=dict(color='#ff7f0e')))
    fig.update_layout(title_text="Stock Prices Over Time", xaxis_rangeslider_visible=True, plot_bgcolor="#f9f9f9")
    st.plotly_chart(fig)

plot_raw_data()

with st.expander("Click here to view Forecasting Results"):
    if 'Close' in data.columns:
        parameters = {'seq': 5, 'unit': 32, 'lr': 1e-2}
        awlstm = AWLSTM(data, parameters)
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=period + 1, freq='D')[1:]
        predictions = awlstm.predict(future_dates)

        forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': predictions})
        st.subheader("Forecast Data")
        st.write(forecast_df.tail())

        st.markdown("#### Forecast Plot")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical', line=dict(color='#1f77b4')))
        fig1.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(color='#ff7f0e')))
        st.plotly_chart(fig1, use_container_width=False, height=500, width=850)
    else:
        st.error("The 'Close' column is missing in the data.")
