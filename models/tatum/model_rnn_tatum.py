import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar datos
file_path = 'Jayson-Tatum-27012025.csv'
df = pd.read_csv(file_path)

# Usar solo la columna de puntos (PTS) y normalizar
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df[['PTS']].values)

# Dividir en conjuntos de entrenamiento y prueba
train_size = int(len(dataset) * 0.80)
train, test = dataset[:train_size], dataset[train_size:]

# Función para crear secuencias de datos
def create_sequences(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

look_back = 5
trainX, trainY = create_sequences(train, look_back)
testX, testY = create_sequences(test, look_back)

# Redimensionar para LSTM
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Construir modelo LSTM
model = Sequential([
    LSTM(10, input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1)

# Hacer predicciones
trainPredict = scaler.inverse_transform(model.predict(trainX))
testPredict = scaler.inverse_transform(model.predict(testX))
trainY_orig = scaler.inverse_transform(trainY)
testY_orig = scaler.inverse_transform(testY)

# Calcular errores
rmse_train = np.sqrt(mean_squared_error(trainY_orig, trainPredict))
rmse_test = np.sqrt(mean_squared_error(testY_orig, testPredict))
mae = mean_absolute_error(testY_orig, testPredict)

print(f'Train RMSE: {rmse_train:.2f}')
print(f'Test RMSE: {rmse_test:.2f}')
print(f'MAE: {mae:.2f}')

# Guardar modelo entrenado
model.save("model_rnn_tatum.h5")
print("Modelo guardado exitosamente.")