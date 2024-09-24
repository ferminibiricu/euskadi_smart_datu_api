# Cargar datos de la estación 205
import numpy as np
X_205 = np.load('../processed/lstm_data/X_205.npy')
y_205 = np.load('../processed/lstm_data/y_205.npy')

print(f"Tamaño de X_205: {X_205.shape}")
print(f"Tamaño de y_205: {y_205.shape}")
