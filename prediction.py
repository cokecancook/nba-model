import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def predict_points(player_key):
    # Build file paths based on the selected player
    model_path = f"models/{player_key}-lstm.h5"
    data_path = f"data/{player_key}.csv"

    # Load the corresponding LSTM model
    model = load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='adam')

    # Load player data
    df = pd.read_csv(data_path)

    # Take the last 5 games
    last_5_games = df["PTS"].tail(5).values

    # Convert to NumPy array and normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_5_games = np.array(last_5_games).reshape(-1, 1)
    pts_scaled = scaler.fit_transform(last_5_games)

    # Prepare the data for the LSTM model
    input_data = np.reshape(pts_scaled, (1, 5, 1))

    # Make the prediction
    predicted_scaled = model.predict(input_data)

    # Inverse the normalization to get the original points
    predicted_pts = scaler.inverse_transform(predicted_scaled)[0][0]

    # Round the predicted points to 2 decimal places
    predicted_pts = round(predicted_pts, 2)

    return predicted_pts

