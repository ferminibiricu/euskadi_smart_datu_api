import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import json
import unicodedata
import re
import joblib

def normalize_station_name(name):
    # Convertir a mayúsculas
    name = name.upper()
    # Reemplazar caracteres especiales y acentos
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')
    # Reemplazar espacios y guiones bajos por nada
    name = name.replace(' ', '').replace('_', '')
    # Eliminar caracteres no alfanuméricos
    name = re.sub(r'[^A-Z0-9]', '', name)
    return name

# Cargar los datos históricos desde el archivo procesado con valores llenados
input_file = '../processed/combined_air_quality_data_filled.csv'
df = pd.read_csv(input_file)

# Cargar datos de las estaciones desde el archivo JSON
station_file = '../raw/Listado estaciones de calidad del aire.json'
with open(station_file, 'r', encoding='utf-8') as f:
    station_data = json.load(f)

# Crear un diccionario para mapear nombres de estaciones a sus IDs
station_map = {station['properties']['name']: int(station['properties']['id']) for station in station_data['features']}

# Normalizar los nombres de las estaciones en el DataFrame
df['StationName_norm'] = df['StationName'].apply(normalize_station_name)

# Normalizar los nombres de las estaciones en el diccionario de mapeo
station_map_norm = {normalize_station_name(k): v for k, v in station_map.items()}

# Diccionario manual para estaciones restantes
manual_station_map = {
    'ALGORTABBIZI2': 90,
    'AVDATOLOSA': 66,
    'ZELAIETAPARQUE': 94,
    'MDIAZHARO': 81,
    'BOROAMETEO': 205  # Asignar StationId para BOROA_METEO
}

# Combinar los diccionarios de mapeo
full_station_map_norm = {**station_map_norm, **manual_station_map}

# Mapear StationId usando los nombres normalizados
df['StationId'] = df['StationName_norm'].map(full_station_map_norm)

# Identificar estaciones sin StationId
estaciones_sin_id = df[df['StationId'].isna()]['StationName'].unique()
if len(estaciones_sin_id) > 0:
    print("\nEstaciones sin StationId asignado:", estaciones_sin_id)
else:
    print("\nTodas las estaciones tienen StationId asignado.")

# Eliminar filas con StationId NaN
df.dropna(subset=['StationId'], inplace=True)

# Convertir 'StationId' a entero
df['StationId'] = df['StationId'].astype(int)

# Eliminar la columna 'StationName_norm' si no es necesaria
df.drop(columns=['StationName_norm'], inplace=True)

# Verificar las columnas antes de eliminar
print("\nColumnas antes de eliminar columnas no numéricas:")
print(df.columns.tolist())

# Identificar columnas con valores de texto
remaining_columns_with_strings = df.select_dtypes(include=['object']).columns.tolist()

# Excluir 'StationName' y 'DateTime' de las columnas a eliminar
columns_to_drop = [col for col in remaining_columns_with_strings if col not in ['StationName', 'DateTime']]

# Si hay columnas a eliminar, se imprimen y se eliminan
if columns_to_drop:
    print(f"\nLas siguientes columnas contienen valores no numéricos y serán eliminadas: {columns_to_drop}")
    df.drop(columns=columns_to_drop, inplace=True)
else:
    print("\nNo se encontraron columnas adicionales de tipo 'object' para eliminar.")

# Verificar las columnas después de eliminar
print("\nColumnas después de eliminar columnas no numéricas:")
print(df.columns.tolist())

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

    # Convertir station_id a entero por si acaso
    station_id_int = int(station_id)

    # Guardar los datos preprocesados para cada estación
    np.save(os.path.join(output_dir, f'X_{station_id_int}.npy'), X)
    np.save(os.path.join(output_dir, f'y_{station_id_int}.npy'), y)
    np.save(os.path.join(output_dir, f'dates_{station_id_int}.npy'), seq_dates)  # Guardar las fechas

    print(f"Datos guardados para la estación {station_id_int}: X.shape = {X.shape}, y.shape = {y.shape}")

print(f"\nPreprocesamiento completado. Datos guardados en {output_dir}")
