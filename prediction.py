import numpy as np
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo LSTM
model = load_model("models/tatum/model_rnn_tatum.h5", compile=False)
model.compile(loss='mse', optimizer='adam')

# Función para realizar la predicción
def predict_points(pts):
    # Convertir a array de NumPy
    pts = np.array(pts).reshape(-1, 1)

    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    pts_scaled = scaler.fit_transform(pts)

    # Preparar los datos para el modelo LSTM
    input_data = np.reshape(pts_scaled, (1, 5, 1))

    # Hacer la predicción
    predicted_scaled = model.predict(input_data)

    # Invertir la normalización para obtener los puntos originales
    predicted_pts = scaler.inverse_transform(predicted_scaled)[0][0]

    return predicted_pts