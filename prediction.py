from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def predict_points(player_key):
    # Construir las rutas de los archivos basadas en el jugador seleccionado
    model_path = f"models/{player_key}-lstm.h5"
    data_path = f"data/{player_key}.csv"

    # Cargar el modelo LSTM correspondiente
    model = load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='adam')

    # Cargar los datos del jugador
    df = pd.read_csv(data_path)

    # Tomar los últimos 5 partidos
    last_5_games = df["PTS"].tail(5).values

    # Convertir a array de NumPy y normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_5_games = np.array(last_5_games).reshape(-1, 1)
    pts_scaled = scaler.fit_transform(last_5_games)

    # Preparar los datos para el modelo LSTM
    input_data = np.reshape(pts_scaled, (1, 5, 1))

    # Hacer la predicción
    predicted_scaled = model.predict(input_data)

    # Invertir la normalización para obtener los puntos originales
    predicted_pts = scaler.inverse_transform(predicted_scaled)[0][0]

    return predicted_pts
