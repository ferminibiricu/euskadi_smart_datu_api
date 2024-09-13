import os
import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Cargar el mapeo de estaciones desde el archivo JSON
station_mapping_file = '../raw/Listado estaciones de calidad del aire.json'

with open(station_mapping_file, 'r', encoding='utf-8') as f:
    station_mapping = json.load(f)

# Crear un diccionario para mapear nombres de estaciones a sus IDs
station_name_to_id = {}
for feature in station_mapping['features']:
    station_name = feature['properties']['name'].replace(' ', '_').upper()
    station_id = feature['properties']['id']
    station_name_to_id[station_name] = int(station_id)  # Guardar ID como entero

# Directorio de datos preprocesados
data_dir = '../processed/lstm_data'

# Función para formatear el nombre del archivo basado en el nombre de la estación
def format_station_name_for_file(station_name):
    return re.sub(r'[^a-zA-Z0-9]', '_', station_name)

# Función para cargar los datos de una estación
def load_station_data(station_name):
    formatted_station_name = format_station_name_for_file(station_name)
    
    # Cargar los archivos X y y correspondientes a la estación
    X = np.load(os.path.join(data_dir, f'X_{formatted_station_name}.npy'))
    y = np.load(os.path.join(data_dir, f'y_{formatted_station_name}.npy'))
    
    # Obtener el ID de la estación
    station_id = station_name_to_id[station_name]
    
    # Expandir la dimensión de station_id para que coincida con la dimensión de los datos X
    station_id_array = np.full((X.shape[0], X.shape[1], 1), station_id)
    
    # Concatenar el ID de la estación como una característica adicional a los datos de entrada
    X_with_station_id = np.concatenate([X, station_id_array], axis=2)
    
    return X_with_station_id, y

# Lista de estaciones a usar basada en los nombres en el mapeo JSON
stations = list(station_name_to_id.keys())

# Variables para acumular todos los datos
all_X = []
all_y = []

# Cargar los datos de todas las estaciones
for station in stations:
    print(f"Cargando datos para la estación: {station}")
    try:
        X, y = load_station_data(station)
        all_X.append(X)
        all_y.append(y)
    except FileNotFoundError as e:
        print(f"Archivo no encontrado para la estación {station}: {e}")

# Concatenar todos los datos en matrices finales
all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))  # Salida para la predicción del índice de calidad del aire

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model.save('../model/lstm_air_quality_model.h5')

print("Entrenamiento completado y modelo guardado.")
