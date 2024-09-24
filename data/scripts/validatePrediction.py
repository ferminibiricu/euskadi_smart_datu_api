import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Cargar el modelo
model_file = '../model/lstm_air_quality_model.h5'
model = load_model(model_file)

# Cargar el escalador
scaler_file = '../processed/scaler.pkl'
scaler = joblib.load(scaler_file)

# Verificar la forma de entrada del modelo
print(f"Input shape esperado: {model.input_shape}")

# Simula datos de entrada escalados con la forma adecuada (1 muestra, 24 pasos de tiempo, 31 características)
X_test_sample = np.random.rand(1, 24, 31)

# Escalar los datos
X_test_sample_scaled = scaler.transform(X_test_sample.reshape(-1, X_test_sample.shape[-1])).reshape(X_test_sample.shape)

# Realiza la predicción
pred = model.predict(X_test_sample_scaled)
predicted_class = np.argmax(pred, axis=1)
print(f"Predicción: {pred}")
print(f"Clase predicha: {predicted_class}")
