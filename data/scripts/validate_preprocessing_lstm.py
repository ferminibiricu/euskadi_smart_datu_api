import numpy as np
import os

# Directorio de los archivos npy
data_dir = '../processed/lstm_data'

# Estaciones para revisar (puedes ajustar esto según las estaciones que has generado)
stations = ['VALDEREJO', 'JAIZKIBEL', 'PUYO', 'AVDA_TOLOSA', 'ZIERBENA_Puerto']  

for station in stations:
    try:
        X = np.load(os.path.join(data_dir, f'X_{station}.npy'))
        y = np.load(os.path.join(data_dir, f'y_{station}.npy'))
        
        print(f"Estación: {station}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Primera secuencia X: {X[0]}")
        print(f"Primer valor y: {y[0]}")
        print("="*40)
    except FileNotFoundError:
        print(f"Archivos para la estación {station} no encontrados.")

