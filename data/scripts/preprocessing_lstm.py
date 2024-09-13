import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Cargar los datos históricos
input_file = '../processed/combined_air_quality_data.csv'
df = pd.read_csv(input_file)

# Asegurarse de que DateTime es un objeto de tipo datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Crear un diccionario para mapear los valores categóricos a números
category_map = {
    'Muy bueno / Oso ona': 5,
    'Bueno / Ona': 4,
    'Regular / Erregularra': 3,
    'Malo / Txarra': 2,
    'Muy malo / Oso txarra': 1,
    'Sin datos / Daturik gabe': 0  # Para los valores sin datos
}

# Identificar las columnas categóricas
categorical_columns = [col for col in df.columns if 'ICA' in col or col == 'ICA Estación']

# Reemplazar los valores categóricos por sus equivalentes numéricos
for col in categorical_columns:
    df[col] = df[col].map(category_map)

# Revisar si quedan columnas con valores de texto
remaining_columns_with_strings = df.select_dtypes(include=['object']).columns.tolist()

# Si hay columnas con valores de texto, se imprimen para ver cuáles quedan
if remaining_columns_with_strings:
    print(f"Las siguientes columnas contienen valores no numéricos: {remaining_columns_with_strings}")

# Normalización de los datos
scaler = MinMaxScaler()

# Columnas a escalar (exceptuando 'DateTime', 'StationName' y las columnas categóricas ya convertidas)
columns_to_scale = df.columns.difference(['DateTime', 'StationName'] + categorical_columns)
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Definir función para crear secuencias de tiempo
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Estaciones
stations = df['StationName'].unique()
n_steps = 24  # 24 horas hacia atrás

# Directorio de salida para los datos preprocesados
output_dir = '../processed/lstm_data'
os.makedirs(output_dir, exist_ok=True)

# Preprocesar los datos por estación
for station in stations:
    station_data = df[df['StationName'] == station].drop(columns=['StationName']).set_index('DateTime')
    
    # Convertir los datos en una matriz de numpy
    station_data_values = station_data.values
    
    # Crear las secuencias de tiempo
    X, y = create_sequences(station_data_values, n_steps)
    
    # Guardar los datos preprocesados para cada estación
    np.save(os.path.join(output_dir, f'X_{station}.npy'), X)
    np.save(os.path.join(output_dir, f'y_{station}.npy'), y)

print(f"Preprocesamiento completado. Datos guardados en {output_dir}")
