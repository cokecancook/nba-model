import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

class BaseModel:
    def __init__(self, file_path, target_column='PTS', look_back=5, train_size=0.8):
        """
        Initializes the BaseModel with common parameters.
        
        Parameters:
        - file_path: Path to the CSV data file.
        - target_column: Column name to be used as the target variable.
        - look_back: Number of previous time steps to use as input.
        - train_size: Proportion of data to be used for training.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.look_back = look_back
        self.train_size = train_size
        
        # Initialize placeholders
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None
        self.train, self.test = None, None
        self.trainX, self.trainY = None, None
        self.testX, self.testY = None, None
        self.trainY_orig, self.testY_orig = None, None

    def load_data(self):
        """Loads data from CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        self.df = pd.read_csv(self.file_path)
        if self.target_column not in self.df.columns:
            raise ValueError(f"Column '{self.target_column}' not found in the dataset.")
        print("Data loaded successfully.")

    def preprocess_data(self):
        """Selects target column and scales the data."""
        self.dataset = self.scaler.fit_transform(self.df[[self.target_column]].values)
        print("Data preprocessing completed.")

    def split_data(self):
        """Splits the dataset into training and testing sets."""
        train_size = int(len(self.dataset) * self.train_size)
        self.train, self.test = self.dataset[:train_size], self.dataset[train_size:]
        print(f"Data split into train ({len(self.train)}) and test ({len(self.test)}) sets.")

    def create_sequences(self, reshape=True):
        """Creates input-output sequences for the model."""
        X, Y = [], []
        for i in range(len(self.train) - self.look_back):
            X.append(self.train[i:i+self.look_back])
            Y.append(self.train[i+self.look_back])
        self.trainX, self.trainY = np.array(X), np.array(Y)
        
        X_test, Y_test = [], []
        for i in range(len(self.test) - self.look_back):
            X_test.append(self.test[i:i+self.look_back])
            Y_test.append(self.test[i+self.look_back])
        self.testX, self.testY = np.array(X_test), np.array(Y_test)
        
        self.trainY_orig = self.scaler.inverse_transform(self.trainY.reshape(-1, 1))
        self.testY_orig = self.scaler.inverse_transform(self.testY.reshape(-1, 1))
        
        if reshape:
            # Reshape input for LSTM [samples, time steps, features]
            self.trainX = self.trainX.reshape((self.trainX.shape[0], self.trainX.shape[1], 1))
            self.testX = self.testX.reshape((self.testX.shape[0], self.testX.shape[1], 1))
            print("Sequences created and reshaped for LSTM.")
        else:
            # Reshape for MLP [samples, features]
            self.trainX = self.trainX.reshape((self.trainX.shape[0], self.trainX.shape[1]))
            self.testX = self.testX.reshape((self.testX.shape[0], self.testX.shape[1]))
            print("Sequences created and reshaped for MLP.")

class LSTMModel(BaseModel):
    def __init__(self, file_path, target_column='PTS', look_back=10, train_size=0.8, 
                epochs=50, batch_size=1, model_save_path='model_lstm.h5'):
        super().__init__(file_path, target_column, look_back, train_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.model = None
        self.trainPredict, self.testPredict = None, None

    def create_sequences(self):
        super().create_sequences(reshape=True)

    def build_model(self):
        """Builds and compiles the LSTM model."""
        self.model = Sequential([
            LSTM(50, input_shape=(self.look_back, 1), return_sequences=False),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print("LSTM model built and compiled.")

    def train_model(self):
        """Trains the LSTM model."""
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            self.trainX, self.trainY,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[early_stop],
            validation_data=(self.testX, self.testY)
        )
        print("LSTM model training completed.")
        self.plot_training_history(history)

    def make_predictions(self):
        """Makes predictions and inverses the scaling."""
        self.trainPredict = self.scaler.inverse_transform(self.model.predict(self.trainX))
        self.testPredict = self.scaler.inverse_transform(self.model.predict(self.testX))
        print("LSTM model predictions made on training and testing data.")

    def evaluate_model(self):
        """Evaluates the LSTM model using RMSE and MAE metrics."""
        rmse_train = np.sqrt(mean_squared_error(self.trainY_orig, self.trainPredict))
        rmse_test = np.sqrt(mean_squared_error(self.testY_orig, self.testPredict))
        mae = mean_absolute_error(self.testY_orig, self.testPredict)
        
        print(f'LSTM Train RMSE: {rmse_train:.2f}')
        print(f'LSTM Test RMSE: {rmse_test:.2f}')
        print(f'LSTM MAE: {mae:.2f}')
        
        return {'Train RMSE': rmse_train, 'Test RMSE': rmse_test, 'MAE': mae}

    def save_model(self):
        """Saves the trained LSTM model to the specified path."""
        self.model.save(self.model_save_path)
        print(f"LSTM model saved successfully at {self.model_save_path}.")

    def plot_predictions(self):
        """Plots the original vs predicted values for the test set."""
        plt.figure(figsize=(12,6))
        plt.plot(self.testY_orig, label='Actual')
        plt.plot(self.testPredict, label='Predicted')
        plt.title('LSTM: Actual vs Predicted PTS')
        plt.xlabel('Time')
        plt.ylabel('PTS')
        plt.legend()
        plt.show()

    def plot_training_history(self, history):
        """Plots the training and validation loss."""
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM: Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def run_pipeline(self):
        """Executes the entire LSTM pipeline."""
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.create_sequences()
        self.build_model()
        self.train_model()
        self.make_predictions()
        metrics = self.evaluate_model()
        self.save_model()
        self.plot_predictions()
        return metrics

class MLPModel(BaseModel):
    def __init__(self, file_path, target_column='PTS', parameters=[],train_size=0.8,
                epochs=80, batch_size=16, model_save_path='model_mlp.h5'):
        # We no longer need the look_back parameter for MLP
        super().__init__(file_path, target_column, look_back=None, train_size=train_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.model = None
        self.trainPredict, self.testPredict = None, None
        self.parameters = parameters

    def preprocess_data(self):
        """
        Preprocess the data using the selected feature columns for X and the target column for Y.
        X: WEEK_DAY, REST_DAYS, OPPONENT_ID, HOME
        Y: PTS
        """
        features = ['WEEK_DAY', 'REST_DAYS', 'OPPONENT_ID', 'HOME']
        # Check that all required columns exist
        missing_features = [col for col in features if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing features in the dataset: {missing_features}")
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        # Create separate scalers for features and target
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_Y = MinMaxScaler(feature_range=(0, 1))

        # Scale the feature columns and target column
        self.X_scaled = self.scaler_X.fit_transform(self.df[features].values)
        self.Y_scaled = self.scaler_Y.fit_transform(self.df[[self.target_column]].values)
        print("MLP model preprocessing completed using selected features.")

    def split_data(self):
        """Splits the preprocessed data into training and testing sets."""
        train_size = int(len(self.X_scaled) * self.train_size)
        self.trainX = self.X_scaled[:train_size]
        self.testX = self.X_scaled[train_size:]
        self.trainY = self.Y_scaled[:train_size]
        self.testY = self.Y_scaled[train_size:]
        # Inverse-transform the target values for later evaluation
        self.trainY_orig = self.scaler_Y.inverse_transform(self.trainY)
        self.testY_orig = self.scaler_Y.inverse_transform(self.testY)
        print(f"MLP model data split into train ({len(self.trainX)}) and test ({len(self.testX)}) sets.")

    def build_model(self):
        """Builds and compiles the MLP model."""
        self.model = Sequential([
            Dense(64, input_dim=self.trainX.shape[1], activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        optimizer = RMSprop(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        print("MLP model built and compiled.")

    def train_model(self):
        """Trains the MLP model."""
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            self.trainX, self.trainY,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[early_stop],
            validation_data=(self.testX, self.testY)
        )
        print("MLP model training completed.")
        self.plot_training_history(history)

    def make_predictions(self, parameters=[]):
        """Makes predictions and inverses the scaling for the target variable."""
        self.trainPredict = self.scaler_Y.inverse_transform(self.model.predict(self.trainX))
        self.testPredict = self.scaler_Y.inverse_transform(self.model.predict(self.testX))
        print("MLP model predictions made on training and testing data.")

    def evaluate_model(self):
        """Evaluates the MLP model using RMSE and MAE metrics."""
        rmse_train = np.sqrt(mean_squared_error(self.trainY_orig, self.trainPredict))
        rmse_test = np.sqrt(mean_squared_error(self.testY_orig, self.testPredict))
        mae = mean_absolute_error(self.testY_orig, self.testPredict)

        print(f'MLP Train RMSE: {rmse_train:.2f}')
        print(f'MLP Test RMSE: {rmse_test:.2f}')
        print(f'MLP MAE: {mae:.2f}')

        return {'Train RMSE': rmse_train, 'Test RMSE': rmse_test, 'MAE': mae}

    def save_model(self):
        """Saves the trained MLP model and scalers."""
        self.model.save(self.model_save_path)

        # Guardar los scalers con pickle
        with open(self.model_save_path.replace('.h5', '_scaler_X.pkl'), 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(self.model_save_path.replace('.h5', '_scaler_Y.pkl'), 'wb') as f:
            pickle.dump(self.scaler_Y, f)
        
        print(f"MLP model and scalers saved successfully at {self.model_save_path}.")

    def plot_predictions(self):
        """Plots the original vs. predicted values for the test set."""
        plt.figure(figsize=(12,6))
        plt.plot(self.testY_orig, label='Actual')
        plt.plot(self.testPredict, label='Predicted')
        plt.title('MLP: Actual vs. Predicted PTS')
        plt.xlabel('Sample')
        plt.ylabel('PTS')
        plt.legend()
        plt.show()

    def plot_training_history(self, history):
        """Plots the training and validation loss."""
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('MLP: Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def run_pipeline(self):
        """Executes the entire MLP pipeline."""
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.build_model()
        self.train_model()
        self.make_predictions(self.parameters)
        metrics = self.evaluate_model()
        self.save_model()
        self.plot_predictions()
        return metrics


# Example usage
if __name__ == "__main__":
    PLAYER = 'jayson-tatum'
    
    # File path to your CSV
    file_path = f'data/{PLAYER}.csv'
    
    # Paths to save models
    lstm_save_path = f'models/model-lstm-{PLAYER}.h5'
    mlp_save_path = f'models/model-mlp-{PLAYER}.h5'
    
    # Initialize and run LSTMModel
    lstm_model = LSTMModel(
        file_path=file_path,
        target_column='PTS',
        look_back=5,
        train_size=0.80,
        epochs=100,  # Increased epochs for potential better training
        batch_size=32,  # Adjusted batch size for efficiency
        model_save_path=lstm_save_path
    )
    lstm_metrics = lstm_model.run_pipeline()
    
    # Initialize and run MLPModel
    mlp_model = MLPModel(
        file_path=file_path,
        target_column='PTS',
        train_size=0.80,
        epochs=80,  # As per your MLP setup
        batch_size=16,  # As per your MLP setup
        model_save_path=mlp_save_path
    )
    mlp_metrics = mlp_model.run_pipeline()
    
    # Optionally, compare metrics
    print("\n=== Model Performance ===")
    print("LSTM Metrics:", lstm_metrics)
    print("MLP Metrics:", mlp_metrics)