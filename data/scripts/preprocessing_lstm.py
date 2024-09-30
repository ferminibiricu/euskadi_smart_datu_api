# preprocessing_lstm.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Cargar los datos históricos desde el archivo procesado con valores llenados
input_file = '../processed/combined_air_quality_data_filled.csv'
df = pd.read_csv(input_file)

# Asegurarse de que 'DateTime' es de tipo datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Definir la variable objetivo y las características
target_column = 'ICA Estación'
feature_columns = [col for col in df.columns if col not in ['DateTime', 'StationName', 'StationId', target_column]]

# Verificar si hay valores NaN en las características
nan_columns = df[feature_columns].columns[df[feature_columns].isna().any()].tolist()
if nan_columns:
    print(f"\nLas siguientes columnas contienen valores NaN y serán rellenadas con la media: {nan_columns}")
    # Rellenar valores NaN con la media de la columna
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
else:
    print("\nNo se encontraron valores NaN en las características.")

# Eliminar columnas con todos los valores NaN
df.dropna(axis=1, how='all', inplace=True)

# Actualizar feature_columns después de eliminar columnas
feature_columns = [col for col in df.columns if col not in ['DateTime', 'StationName', 'StationId', target_column]]

# Normalización de las características
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Guardar el escalador para su uso futuro
scaler_file = '../processed/scaler.pkl'
joblib.dump(scaler, scaler_file)
print(f"\nEscalador guardado en {scaler_file}")

# Asegurarse de que las etiquetas están en el rango 0-5
valid_labels = [0, 1, 2, 3, 4, 5]
df = df[df[target_column].isin(valid_labels)]

# Estaciones
stations = df['StationId'].unique()
print(f"\nNúmero de estaciones procesadas: {len(stations)}")

n_steps = 24  # 24 horas hacia atrás

# Directorio de salida para los datos preprocesados
output_dir = '../processed/lstm_data'
os.makedirs(output_dir, exist_ok=True)

# Preparamos un diccionario para guardar las fechas
dates_dict = {}

# Definir función para crear secuencias de tiempo
def create_sequences(features, target, dates, n_steps):
    X, y, seq_dates = [], [], []
    for i in range(len(features) - n_steps):
        X.append(features[i:i + n_steps])
        y.append(target[i + n_steps])  # Valor objetivo correspondiente al final de la ventana
        seq_dates.append(dates[i + n_steps])  # Fecha correspondiente al valor objetivo
    return np.array(X), np.array(y), np.array(seq_dates)

# Preprocesar los datos por estación
for station_id in stations:
    station_df = df[df['StationId'] == station_id].copy()
    station_df.sort_values('DateTime', inplace=True)  # Asegurar que los datos están ordenados temporalmente

    features = station_df[feature_columns].values
    target = station_df[target_column].values
    dates = station_df['DateTime'].values

    # Crear las secuencias de tiempo
    X, y, seq_dates = create_sequences(features, target, dates, n_steps)

    # Verificar si hay suficientes datos
    if len(X) == 0 or len(y) == 0:
        print(f"No hay suficientes datos para la estación {station_id}. Se omitirá.")
        continue

    # Asegurarse de que las etiquetas están en el rango 0-5
    valid_indices = np.isin(y, valid_labels)
    X = X[valid_indices]
    y = y[valid_indices]
    seq_dates = seq_dates[valid_indices]

    # Guardar los datos preprocesados para cada estación
    np.save(os.path.join(output_dir, f'X_{station_id}.npy'), X)
    np.save(os.path.join(output_dir, f'y_{station_id}.npy'), y)
    np.save(os.path.join(output_dir, f'dates_{station_id}.npy'), seq_dates)  # Guardar las fechas

    print(f"Datos guardados para la estación {station_id}: X.shape = {X.shape}, y.shape = {y.shape}")

print(f"\nPreprocesamiento completado. Datos guardados en {output_dir}")
