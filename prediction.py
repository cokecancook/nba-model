import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def predict_points_lstm(player_key):
    """
    Predicts points using the LSTM model based on the last 5 games' PTS.
    """
    # Build file paths based on the selected player
    model_path = f"models/model-lstm-{player_key}.h5"
    data_path = f"data/{player_key}.csv"

    # Load the corresponding LSTM model
    model = load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='adam')

    # Load player data
    df = pd.read_csv(data_path)

    # Take the last 5 games for the PTS column
    last_5_games = df["PTS"].tail(5).values

    # Convert to NumPy array and reshape for scaling
    last_5_games = np.array(last_5_games).reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    pts_scaled = scaler.fit_transform(last_5_games)

    # Prepare the data for the LSTM model: [samples, time steps, features]
    input_data = np.reshape(pts_scaled, (1, 5, 1))

    # Make the prediction
    predicted_scaled = model.predict(input_data)

    # Inverse the normalization to get the original scale
    predicted_pts = scaler.inverse_transform(predicted_scaled)[0][0]
    predicted_pts = round(predicted_pts, 2)

    return predicted_pts

def predict_points_mlp(player_key, week_day, rest_days, opponent_id, home):
    """
    Predicts points using the MLP model based on features:
    WEEK_DAY, REST_DAYS, OPPONENT_ID, and HOME from the last game.
    """
    # Build file paths based on the selected player
    model_path = f"models/model-mlp-{player_key}.h5"
    data_path = f"data/{player_key}.csv"

    # Load the corresponding MLP model
    model = load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='adam')

    # Load player's historical data for scaling
    df = pd.read_csv(data_path)
    
    # Define feature columns and fit the feature scaler on historical data
    feature_cols = ['WEEK_DAY', 'REST_DAYS', 'OPPONENT_ID', 'HOME']
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(df[feature_cols].values)

    # Prepare the input features from the function arguments and scale them
    input_features = np.array([[week_day, rest_days, opponent_id, home]])
    input_features_scaled = scaler_X.transform(input_features)

    # Prepare the target scaler for PTS (to invert the scaling after prediction)
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    pts = df['PTS'].values.reshape(-1, 1)
    scaler_Y.fit(pts)

    # Make the prediction
    predicted_scaled = model.predict(input_features_scaled)
    predicted_pts = scaler_Y.inverse_transform(predicted_scaled)[0][0]
    predicted_pts = round(predicted_pts, 2)

    return predicted_pts


def predict_points_combined(player_key, week_day, rest_days, opponent_id, home):
    """
    Combines predictions from both LSTM and MLP models.
    Here, we simply average the two predictions.
    """
    lstm_prediction = predict_points_lstm(player_key)
    mlp_prediction = predict_points_mlp(player_key, week_day, rest_days, opponent_id, home)

    # Simple logic: average the two predictions
    combined_prediction = round((lstm_prediction + mlp_prediction) / 2, 2)

    print(f"LSTM Prediction: {lstm_prediction}")
    print(f"MLP Prediction: {mlp_prediction}")
    print(f"Combined Prediction: {combined_prediction}")

    return combined_prediction

# Example usage:
if __name__ == "__main__":
    player_key = "jayson-tatum"  # Replace with the desired player key
    final_prediction = predict_points_combined(player_key)
    print("Final Combined Predicted Points:", final_prediction)
