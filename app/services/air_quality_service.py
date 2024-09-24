import logging
import numpy as np
import joblib
import requests
from fastapi import HTTPException
from tensorflow.keras.models import load_model
from config.config import air_quality_base_url, hourly_measurements_endpoint, exclusion_threshold

# Cargar el modelo LSTM
model_file = 'data/model/lstm_air_quality_model.h5'
model = load_model(model_file)
logging.info(f"Modelo LSTM cargado desde {model_file}")

# Cargar el escalador
scaler_file = 'data/processed/scaler.pkl'
scaler = joblib.load(scaler_file)
logging.info(f"Escalador cargado desde {scaler_file}")

def get_air_quality_data(station_code: str, hours: int):
    # Calcular el rango de tiempo para los datos
    start_time, end_time = calculate_time_range(hours)
    url = f"{air_quality_base_url}{hourly_measurements_endpoint.format(station_code=station_code, start_time=start_time, end_time=end_time)}"
    
    logging.info(f"Fetching air quality data from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Data successfully retrieved for station ID {station_code}")
        data = response.json()
        
        # Registrar el contenido de los datos recibidos para depuración
        logging.debug(f"Datos de calidad del aire recibidos - len(data): {len(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            return data
        else:
            logging.warning(f"Unexpected data structure for station ID {station_code}: {data}")
            raise HTTPException(status_code=404, detail="Unexpected data structure for air quality data.")
        
    except requests.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except requests.RequestException as e:
        logging.error(f"Request error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching air quality data.")

def preprocess_data_for_lstm(data):
    """
    Preprocess data for LSTM model input, including scaling the features.
    """
    try:
        # Definir las características utilizadas durante el entrenamiento
        feature_names = [
            'D.vien', 'D.vien_dup',
            'H', 'H_dup',
            'NO', 'NO_dup',
            'NO2', 'NO2_dup',
            'NOX', 'NOX_dup',
            'P', 'P_dup',
            'PM10', 'PM10_dup',
            'PM2,5', 'PM2,5_dup',
            'R', 'R_dup',
            'SH2', 'SH2_dup',
            'SO2', 'SO2_dup',
            'Tº', 'Tº_dup',
            'V.vien', 'V.vien_dup',
            'NO2_ICA', 'PM10_ICA', 'PM2,5_ICA', 'SO2_ICA',
            'airQualityStation'
        ]
        expected_num_features = len(feature_names)

        # Mapeo de niveles de calidad del aire
        airquality_mapping = {
            "Muy malo": 1,
            "Malo": 2,
            "Regular": 3,
            "Bueno": 4,
            "Muy bueno": 5,
            "Sin datos": 0
        }

        # Lista para almacenar los vectores de características por hora
        feature_vectors = []

        for d in data:
            station_data = d.get('station', [{}])[0]
            measurements = station_data.get('measurements', [])
            airQualityStation = station_data.get('airQualityStation', 'Sin datos')
            airQualityStation_value = airquality_mapping.get(airQualityStation, np.nan)

            measurement_dict = {}
            counts = {}
            for m in measurements:
                name = m['name']
                value = m.get('value', np.nan)
                if name in counts:
                    counts[name] += 1
                    key = f"{name}_dup"
                else:
                    counts[name] = 1
                    key = name
                measurement_dict[key] = value
                # Si existe el campo 'airquality', mapearlo a un valor numérico
                if 'airquality' in m:
                    airquality = m['airquality']
                    ica_value = airquality_mapping.get(airquality, np.nan)
                    measurement_dict[f"{name}_ICA"] = ica_value

            # Añadir airQualityStation
            measurement_dict['airQualityStation'] = airQualityStation_value

            # Crear el vector de características para este punto en el tiempo
            feature_vector = []
            for feature in feature_names:
                value = measurement_dict.get(feature, np.nan)
                feature_vector.append(value)

            feature_vectors.append(feature_vector)

        # Convertir la lista de vectores en un array de numpy
        features = np.array(feature_vectors, dtype=np.float32)

        # Verificar si el número de características coincide con el esperado
        if features.shape[1] != expected_num_features:
            logging.error(f"El número de características ({features.shape[1]}) no coincide con el esperado ({expected_num_features}).")
            raise ValueError(f"El número de características ({features.shape[1]}) no coincide con el esperado ({expected_num_features}).")

        # Manejar valores faltantes
        col_means = np.nanmean(features, axis=0)
        inds = np.where(np.isnan(features))
        features[inds] = np.take(col_means, inds[1])

        # Asegurarse de que tenemos 24 pasos de tiempo
        n_timesteps_expected = 24
        if features.shape[0] >= n_timesteps_expected:
            features = features[-n_timesteps_expected:, :]
        else:
            # Si tenemos menos de 24 pasos, rellenamos con la primera fila
            n_missing = n_timesteps_expected - features.shape[0]
            padding = np.repeat(features[0:1, :], n_missing, axis=0)
            features = np.concatenate((padding, features), axis=0)

        # Redimensionar a 3D para el LSTM (samples, timesteps, features)
        features_reshaped = features.reshape((1, features.shape[0], features.shape[1]))

        # Escalar las características usando el escalador previamente cargado
        n_timesteps = features_reshaped.shape[1]
        n_features = features_reshaped.shape[2]
        features_scaled = scaler.transform(features_reshaped.reshape(-1, n_features)).reshape(1, n_timesteps, n_features)

        return features_scaled

    except Exception as e:
        logging.error(f"Error while preprocessing data: {e}")
        raise HTTPException(status_code=500, detail="Error in preprocessing air quality data for LSTM model.")

def postprocess_lstm_output(prediction):
    """
    Postprocess the raw output of the LSTM model into a readable air quality category.
    """
    categories = ["Sin datos", "Muy malo", "Malo", "Regular", "Bueno", "Muy bueno"]
    predicted_category_index = np.argmax(prediction)
    return categories[predicted_category_index]

def calculate_time_range(hours: int = 24):
    from datetime import datetime, timedelta
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")
    return start_time_str, end_time_str

def calculate_air_quality_summary(data, hours: int):
    air_quality_levels = {
        "Muy malo": 1,
        "Malo": 2,
        "Regular": 3,
        "Bueno": 4,
        "Muy bueno": 5
    }
    
    air_quality_values = []
    total_data_points = len(data)
    weight_sum = 0
    weighted_sum = 0
    
    # Asignar pesos decrecientes
    weights = [(total_data_points - i) / total_data_points for i in range(total_data_points)]
    
    for index, hour_data in enumerate(data):
        if 'station' in hour_data and isinstance(hour_data['station'], list) and len(hour_data['station']) > 0:
            air_quality = hour_data['station'][0].get('airQualityStation')
            if air_quality in air_quality_levels:
                air_quality_values.append(air_quality)
                weighted_sum += air_quality_levels[air_quality] * weights[index]
                weight_sum += weights[index]
    
    missing_data_count = total_data_points - len(air_quality_values)
    missing_data_percentage = (missing_data_count / total_data_points) * 100
    if missing_data_percentage > exclusion_threshold:
        return "Datos no disponibles"

    if weight_sum > 0:
        average_score = weighted_sum / weight_sum
        if average_score <= 1.5:
            return "Muy malo"
        elif average_score <= 2.5:
            return "Malo"
        elif average_score <= 3.5:
            return "Regular"
        elif average_score <= 4.5:
            return "Bueno"
        else:
            return "Muy bueno"
    else:
        return "Datos no disponibles"

def predict_air_quality(current_air_quality_data):
    """
    Uses the LSTM model to predict the air quality based on current data.
    """
    logging.info(f"Predicting air quality based on current air quality data.")
    
    # Preprocesar los datos actuales de calidad del aire para el modelo
    features_scaled = preprocess_data_for_lstm(current_air_quality_data)
    
    # Generar la predicción utilizando el modelo LSTM
    prediction = model.predict(features_scaled)
    
    # Postprocesar la predicción para obtener la categoría de calidad del aire
    predicted_summary = postprocess_lstm_output(prediction)
    
    return predicted_summary
