import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar datos
file_path = '/Users/mauro/Documents/MIA/IA_PROJECT2/ia_project_2/models/tatum/lebron/LeBron-James-10022025.csv'
df = pd.read_csv(file_path)

# Normalizar la columna de puntos (PTS)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df[['PTS']].values)

# Dividir en entrenamiento y prueba
train_size = int(len(dataset) * 0.80)
train, test = dataset[:train_size], dataset[train_size:]

# Crear secuencias
def create_sequences(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

look_back = 5
trainX, trainY = create_sequences(train, look_back)
testX, testY = create_sequences(test, look_back)

# Aplanar entradas para MLP
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
testX = testX.reshape(testX.shape[0], testX.shape[1])

# Construir modelo MLP optimizado
model = Sequential([
    Dense(64, input_dim=look_back, activation='relu'),
    Dropout(0.2), 

    Dense(32, activation='relu'),
    Dropout(0.2),  

    Dense(1)
])

# Compilar el modelo con RMSprop
optimizer = RMSprop(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

# Entrenar el modelo
model.fit(trainX, trainY, epochs=80, batch_size=16, verbose=1, validation_data=(testX, testY))

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
model.save("/Users/mauro/Documents/MIA/IA_PROJECT2/ia_project_2/models/tatum/lebron/model_mlp_tatum_v1.h5")
print("Modelo guardado exitosamente.")
