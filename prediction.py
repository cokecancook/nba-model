import numpy as np
import pandas as pd
import pickle
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
    predicted_points = scaler.inverse_transform(predicted_scaled)

    return predicted_points[0][0]

def predict_points_mlp(player_key, week_day, rest_days, opponent_id, home):
    """
    Predicts points using the MLP model based on features:
    WEEK_DAY, REST_DAYS, OPPONENT_ID, and HOME from the last game.
    """
    # Build file paths based on the selected player
    model_path = f"models/model-mlp-{player_key}.h5"
    scaler_X_path = model_path.replace('.h5', '_scaler_X.pkl')
    scaler_Y_path = model_path.replace('.h5', '_scaler_Y.pkl')

    # Load the MLP model
    model = load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='adam')

    # Load the scalers
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_Y_path, 'rb') as f:
        scaler_Y = pickle.load(f)

    # Create input array
    input_data = np.array([[week_day, rest_days, opponent_id, home]])

    # Scale the input data (only transform, don't fit)
    scaled_input = scaler_X.transform(input_data)

    # Make the prediction
    prediction = model.predict(scaled_input)

    # Inverse transform the prediction
    predicted_points = scaler_Y.inverse_transform(prediction)

    return predicted_points[0][0]


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
    # Example parameters
    player_key = "jayson-tatum"
    week_day = 2        # Wednesday
    rest_days = 1       # 1 day of rest
    opponent_id = 5     # Example opponent team ID
    home = 1           # Home game
    
    # Make the prediction with all required parameters
    final_prediction = predict_points_combined(
        player_key=player_key,
        week_day=week_day,
        rest_days=rest_days,
        opponent_id=opponent_id,
        home=home
    )
    
    print("\nPrediction Complete!")