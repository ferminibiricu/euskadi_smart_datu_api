import logging
import numpy as np
import joblib
import requests
import pandas as pd
from fastapi import HTTPException
from tensorflow.keras.models import load_model
from config.config import air_quality_base_url, hourly_measurements_endpoint, exclusion_threshold
from datetime import datetime, timedelta

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas absolutas o relativas según la estructura de tu proyecto
MODEL_FILE = 'data/model/lstm_air_quality_best_model.keras'
SCALER_FILE = 'data/model/scaler.pkl'

# Cargar el modelo LSTM
try:
    model = load_model(MODEL_FILE)
    logging.info(f"Modelo LSTM cargado desde {MODEL_FILE}")
except Exception as e:
    logging.error(f"Error al cargar el modelo LSTM desde {MODEL_FILE}: {e}")
    raise

# Cargar el escalador
try:
    scaler = joblib.load(SCALER_FILE)
    logging.info(f"Escalador cargado desde {SCALER_FILE}")
except Exception as e:
    logging.error(f"Error al cargar el escalador desde {SCALER_FILE}: {e}")
    raise

def get_air_quality_data(station_code: str, hours: int):
    """
    Fetch air quality data from the Open Data Euskadi API.
    """
    start_time, end_time = calculate_time_range(hours)
    url = f"{air_quality_base_url}{hourly_measurements_endpoint.format(station_code=station_code, start_time=start_time, end_time=end_time)}"
    
    logging.info(f"Fetching air quality data from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Data successfully retrieved for station ID {station_code}")
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            return data
        else:
            logging.warning(f"No data returned for station ID {station_code}")
            return None
        
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
        # Usar las características del escalador para garantizar la consistencia
        feature_names = scaler.feature_names_in_
    
        measurement_name_mapping = {
            "CO (mg/m3)": ["CO", "CO(mg/m3)", "CO (mg/m3)"],
            "CO 8h (mg/m3)": ["CO 8h", "CO8h", "CO 8h (mg/m3)"],
            "NO (µg/m3)": ["NO", "NO (µg/m3)"],
            "NO2 (µg/m3)": ["NO2", "NO2 (µg/m3)"],
            "NO2 - ICA": ["NO2 - ICA", "NO2 ICA", "NO2-ICA"],
            "NOX (µg/m3)": ["NOX", "NOX (µg/m3)"],
            "PM10 (µg/m3)": ["PM10", "PM10 (µg/m3)"],
            "PM10 - ICA": ["PM10 - ICA", "PM10 ICA", "PM10-ICA"],
            "PM2,5 (µg/m3)": ["PM2,5", "PM2.5", "PM2,5 (µg/m3)", "PM2.5 (µg/m3)"],
            "PM2,5 - ICA": ["PM2,5 - ICA", "PM2,5 ICA", "PM2.5 - ICA", "PM2.5 ICA"],
            "SO2 (µg/m3)": ["SO2", "SO2 (µg/m3)"],
            "SO2 - ICA": ["SO2 - ICA", "SO2 ICA", "SO2-ICA"],
            "D.vien (grados)": ["D.vien", "D.vien (grados)", "Direccion viento", "Dirección viento", "DireccionViento"],
            "H (%)": ["H", "H (%)", "Humedad", "Humedad relativa"],
            "Precipitación (l/m2)": ["Precipitación", "Precipitacion", "Precipitación (l/m2)"],
            "Tº (ºC)": ["Tº", "Tº (ºC)", "Temperatura", "Temperatura (ºC)"],
            "V.vien (m/s)": ["V.vien", "V.vien (m/s)", "Velocidad viento", "VelocidadViento"],
            "O3 (µg/m3)": ["O3", "O3 (µg/m3)"],
            "OZONO": ["OZONO"],
            "O3 - ICA": ["O3 - ICA", "O3 ICA", "O3-ICA"],
            "O3 8h (µg/m3)": ["O3 8h", "O3 8h (µg/m3)"],
            "P (mBar)": ["P", "P (mBar)", "Presión", "Presion"],
            "R.UVA (w/m2)": ["R.UVA", "R.UVA (w/m2)"],
            "R (w/m2)": ["R", "R (w/m2)", "Radiación", "Radiacion"],
            "Benceno (µg/m3)": ["Benceno", "Benceno (µg/m3)"],
            "Etilbenceno (µg/m3)": ["Etilbenceno", "Etilbenceno (µg/m3)"],
            "Tolueno (µg/m3)": ["Tolueno", "Tolueno (µg/m3)"],
            "Ortoxileno (µg/m3)": ["Ortoxileno", "Ortoxileno (µg/m3)"],
            "NH3 (µg/m3)": ["NH3", "NH3 (µg/m3)"],
            "SH2 (µg/m3)": ["SH2", "SH2 (µg/m3)"],
            "M-P-XILENO (µg/m3)": ["M-P-Xileno", "M-P-XILENO (µg/m3)", "MP-Xileno", "MP-XILENO"]
        }
    
        feature_vectors = []
    
        for d in data:
            station_data = d.get('station', [{}])[0]
            measurements = station_data.get('measurements', [])
    
            measurement_dict = {}
            for m in measurements:
                name = m['name']
                unit = m.get('unit', '')
                value = m.get('value', np.nan)
                name_with_unit = f"{name} ({unit})" if unit else name
    
                found = False
                for standard_name, aliases in measurement_name_mapping.items():
                    if name == standard_name or name_with_unit == standard_name or name in aliases or name_with_unit in aliases:
                        measurement_dict[standard_name] = value
                        found = True
                        break
                if not found:
                    logging.debug(f"Measurement name '{name}' with unit '{unit}' not mapped.")
    
            feature_vector = []
            for feature in feature_names:
                value = measurement_dict.get(feature, np.nan)
                feature_vector.append(value)
    
            feature_vectors.append(feature_vector)
    
        features_df = pd.DataFrame(feature_vectors, columns=feature_names)
    
        logging.debug(f"Características extraídas antes de manejar NaN:\n{features_df}")
    
        # Reemplazar valores NaN con la media de la columna
        features_df = features_df.fillna(features_df.mean())
        features_df = features_df.fillna(0)
    
        if features_df.isnull().values.any():
            logging.warning("Aún hay valores NaN después de la imputación. Se reemplazarán por ceros.")
            features_df.fillna(0, inplace=True)
    
        n_timesteps_expected = 24
        if len(features_df) >= n_timesteps_expected:
            features_df = features_df.tail(n_timesteps_expected)
        else:
            n_missing = n_timesteps_expected - len(features_df)
            padding = pd.DataFrame([features_df.mean()] * n_missing, columns=features_df.columns)
            features_df = pd.concat([padding, features_df], ignore_index=True)
            logging.info(f"Datos insuficientes para completar {n_timesteps_expected} pasos de tiempo. Se han rellenado {n_missing} pasos con la media.")
    
        # Asegurarse de que las características están en el orden correcto esperado por el escalador
        features_df = features_df[feature_names]
    
        try:
            features_scaled = scaler.transform(features_df)
        except Exception as e:
            logging.error(f"Error al escalar las características: {e}")
            raise HTTPException(status_code=500, detail="Error en el escalado de datos para el modelo LSTM.")
    
        features_scaled = features_scaled.reshape(1, n_timesteps_expected, -1)
    
        logging.debug(f"Características escaladas: {features_scaled}")
    
        return features_scaled
    
    except Exception as e:
        logging.error(f"Error while preprocessing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error en el preprocesamiento de datos para el modelo LSTM.")

def postprocess_lstm_output(prediction):
    # Corrección del mapeo de índices de clase a categorías
    categories = {
        0: 'Extremadamente deficiente',
        1: 'Muy deficiente',
        2: 'Deficiente',
        3: 'Moderado',
        4: 'Bueno',
        5: 'Muy bueno'
    }
    predicted_category_index = np.argmax(prediction)
    return categories.get(predicted_category_index, 'Desconocido')

def calculate_time_range(hours: int = 24):
    end_time = datetime.utcnow() - timedelta(minutes=10)
    start_time = end_time - timedelta(hours=hours)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")
    return start_time_str, end_time_str

def calculate_air_quality_summary(data, hours: int):
    # Actualización del mapeo para que coincida con los índices de clase correctos
    air_quality_levels = {
        "Muy bueno": 5,
        "Bueno": 4,
        "Moderado": 3,
        "Deficiente": 2,
        "Muy deficiente": 1,
        "Extremadamente deficiente": 0,
        "Sin datos": None
    }
    
    air_quality_values = []
    total_data_points = len(data)
    weight_sum = 0
    weighted_sum = 0

    weights = [(total_data_points - i) / total_data_points for i in range(total_data_points)]

    for index, hour_data in enumerate(data):
        if 'station' in hour_data and isinstance(hour_data['station'], list) and len(hour_data['station']) > 0:
            air_quality = hour_data['station'][0].get('airQualityStation')
            if air_quality in air_quality_levels and air_quality_levels[air_quality] is not None:
                air_quality_values.append(air_quality)
                weighted_sum += air_quality_levels[air_quality] * weights[index]
                weight_sum += weights[index]

    missing_data_count = total_data_points - len(air_quality_values)
    missing_data_percentage = (missing_data_count / total_data_points) * 100
    if missing_data_percentage > exclusion_threshold:
        return "Datos no disponibles"

    if weight_sum > 0:
        average_score = weighted_sum / weight_sum
        average_score_rounded = int(round(average_score))
        return next((category for category, value in air_quality_levels.items() if value == average_score_rounded), "Datos no disponibles")
    else:
        return "Datos no disponibles"

def predict_air_quality(current_air_quality_data):
    logging.info("Predicting air quality based on current air quality data.")
    
    try:
        logging.debug(f"Datos actuales de calidad del aire: {current_air_quality_data}")
        features_scaled = preprocess_data_for_lstm(current_air_quality_data)
                
        if features_scaled.shape[1] < 24:
            logging.warning(f"No hay suficientes pasos de tiempo para la predicción. Se requieren 24, pero se encontraron {features_scaled.shape[1]}.")
            return "Predicción no disponible"
        
        prediction = model.predict(features_scaled)
        logging.debug(f"Predicción cruda del modelo: {prediction}")
        
        predicted_summary = postprocess_lstm_output(prediction)
        logging.debug(f"Resumen de calidad del aire predicho: {predicted_summary}")
        
        return predicted_summary

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return "Predicción no disponible"
