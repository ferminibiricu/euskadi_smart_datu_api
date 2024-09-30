# app/services/air_quality_service.py

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
SCALER_FILE = 'data/processed/scaler.pkl'

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
        logging.debug(f"Datos de calidad del aire recibidos para la estación {station_code}: {data}")
        
        if isinstance(data, list):
            return data
        else:
            logging.warning(f"Unexpected data structure for station ID {station_code}: {data}")
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
        # Definir las características utilizadas durante el entrenamiento
        feature_names = [
            'CO (mg/m3)', 'CO 8h (mg/m3)', 'NO (µg/m3)', 'NO2 (µg/m3)', 'NO2 - ICA', 'NOX (µg/m3)',
            'PM10 (µg/m3)', 'PM10 - ICA', 'PM2,5 (µg/m3)', 'PM2,5 - ICA', 'SO2 (µg/m3)', 'SO2 - ICA',
            'D.vien (grados)', 'H (%)', 'Precipitación (l/m2)', 'Tº (ºC)', 'V.vien (m/s)',
            'O3 (µg/m3)', 'OZONO', 'O3 - ICA', 'O3 8h (µg/m3)', 'P (mBar)', 'R.UVA (w/m2)',
            'R (w/m2)', 'Benceno (µg/m3)', 'Etilbenceno (µg/m3)', 'Tolueno (µg/m3)',
            'Ortoxileno (µg/m3)', 'NH3 (µg/m3)', 'SH2 (µg/m3)', 'M-P-XILENO (µg/m3)'
        ]

        # Mapeo de nombres alternativos de mediciones
        measurement_name_mapping = {
            "CO (mg/m3)": ["CO", "CO(mg/m3)"],
            "CO 8h (mg/m3)": ["CO 8h", "CO8h"],
            "NO (µg/m3)": ["NO"],
            "NO2 (µg/m3)": ["NO2"],
            "NO2 - ICA": ["NO2 ICA", "NO2-ICA"],
            "NOX (µg/m3)": ["NOX"],
            "PM10 (µg/m3)": ["PM10"],
            "PM10 - ICA": ["PM10 ICA", "PM10-ICA"],
            "PM2,5 (µg/m3)": ["PM2.5", "PM2,5"],
            "PM2,5 - ICA": ["PM2.5 ICA", "PM2,5-ICA"],
            "SO2 (µg/m3)": ["SO2"],
            "SO2 - ICA": ["SO2 ICA", "SO2-ICA"],
            "D.vien (grados)": ["D.vien", "DireccionViento"],
            "H (%)": ["H"],
            "Precipitación (l/m2)": ["Precipitacion", "Precipitación"],
            "Tº (ºC)": ["Tº", "Temperatura"],
            "V.vien (m/s)": ["V.vien", "VelocidadViento"],
            "O3 (µg/m3)": ["O3"],
            "OZONO": ["OZONO"],
            "O3 - ICA": ["O3 ICA", "O3-ICA"],
            "O3 8h (µg/m3)": ["O3 8h", "O38h"],
            "P (mBar)": ["P"],
            "R.UVA (w/m2)": ["R.UVA"],
            "R (w/m2)": ["R"],
            "Benceno (µg/m3)": ["Benceno"],
            "Etilbenceno (µg/m3)": ["Etilbenceno"],
            "Tolueno (µg/m3)": ["Tolueno"],
            "Ortoxileno (µg/m3)": ["Ortoxileno"],
            "NH3 (µg/m3)": ["NH3"],
            "SH2 (µg/m3)": ["SH2"],
            "M-P-XILENO (µg/m3)": ["M-P-Xileno", "MP-Xileno"]
        }

        # Lista para almacenar los vectores de características por hora
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

                # Buscar el nombre estándar de la medición
                for standard_name, aliases in measurement_name_mapping.items():
                    if name == standard_name or name_with_unit == standard_name or name in aliases or name_with_unit in aliases:
                        measurement_dict[standard_name] = value
                        break

            # Crear el vector de características para este punto en el tiempo
            feature_vector = []
            for feature in feature_names:
                value = measurement_dict.get(feature, np.nan)
                feature_vector.append(value)

            feature_vectors.append(feature_vector)

        # Convertir la lista de vectores en un DataFrame
        features_df = pd.DataFrame(feature_vectors, columns=feature_names)

        # Añadir logs para depuración
        logging.debug(f"Características extraídas antes de manejar NaN:\n{features_df}")

        # Manejar valores faltantes (NaN)
        # Reemplazar NaN en cada columna con el valor medio de la columna
        features_df = features_df.fillna(features_df.mean())

        # Si aún hay NaN (por ejemplo, si toda la columna era NaN), reemplazar con cero
        features_df = features_df.fillna(0)

        # Verificar si aún hay NaN
        if features_df.isnull().values.any():
            logging.warning("Aún hay valores NaN después de la imputación. Se reemplazarán por ceros.")
            features_df.fillna(0, inplace=True)

        # Asegurarse de que tenemos 24 pasos de tiempo
        n_timesteps_expected = 24
        if len(features_df) >= n_timesteps_expected:
            features_df = features_df.tail(n_timesteps_expected)
        else:
            # Si tenemos menos de 24 pasos, rellenamos con el valor medio de las características existentes
            n_missing = n_timesteps_expected - len(features_df)
            padding = pd.DataFrame([features_df.mean()] * n_missing, columns=features_df.columns)
            features_df = pd.concat([padding, features_df], ignore_index=True)

        # Verificar y corregir las características antes de escalar
        expected_features = scaler.feature_names_in_
        current_features = features_df.columns

        missing_features = set(expected_features) - set(current_features)
        extra_features = set(current_features) - set(expected_features)

        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                features_df[feature] = 0.0  # Asignar un valor predeterminado a las características faltantes

        if extra_features:
            logging.warning(f"Extra features: {extra_features}")
            features_df = features_df.drop(columns=list(extra_features))  # Eliminar las características extra

        # Asegurarse de que el orden de las características es correcto
        features_df = features_df[expected_features]

        # Escalar las características usando el escalador previamente cargado
        try:
            features_scaled = scaler.transform(features_df)
        except Exception as e:
            logging.error(f"Error al escalar las características: {e}")
            raise HTTPException(status_code=500, detail="Error in scaling air quality data for LSTM model.")

        # Redimensionar para el LSTM (samples, timesteps, features)
        features_scaled = features_scaled.reshape(1, n_timesteps_expected, -1)

        logging.debug(f"Características escaladas: {features_scaled}")

        return features_scaled

    except Exception as e:
        logging.error(f"Error while preprocessing data: {e}")
        raise HTTPException(status_code=500, detail="Error in preprocessing air quality data for LSTM model.")

def postprocess_lstm_output(prediction):
    """
    Postprocess the raw output of the LSTM model into a readable air quality category.
    """
    categories = {
        0: 'Muy bueno',
        1: 'Bueno',
        2: 'Regular',
        3: 'Malo',
        4: 'Muy malo',
        5: 'Sin datos'
    }
    predicted_category_index = np.argmax(prediction)
    return categories.get(predicted_category_index, 'Desconocido')

def calculate_time_range(hours: int = 24):
    """
    Calculate the time range for fetching air quality data.
    """
    end_time = datetime.utcnow() - timedelta(minutes=10)  # Restar 10 minutos para evitar problemas de sincronización
    start_time = end_time - timedelta(hours=hours)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")
    return start_time_str, end_time_str

def calculate_air_quality_summary(data, hours: int):
    """
    Calculate a weighted average summary of air quality over the past 'hours' hours.
    """
    air_quality_levels = {
        "Muy bueno": 0,
        "Bueno": 1,
        "Regular": 2,
        "Malo": 3,
        "Muy malo": 4,
        "Sin datos": 5
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
        # Redondear al entero más cercano
        average_score_rounded = int(round(average_score))
        return next((category for category, value in air_quality_levels.items() if value == average_score_rounded), "Datos no disponibles")
    else:
        return "Datos no disponibles"

def predict_air_quality(current_air_quality_data):
    """
    Uses the LSTM model to predict the air quality based on current data.
    """
    logging.info("Predicting air quality based on current air quality data.")
    
    try:
        # Preprocesar los datos actuales de calidad del aire para el modelo
        logging.debug(f"Datos actuales de calidad del aire: {current_air_quality_data}")
        features_scaled = preprocess_data_for_lstm(current_air_quality_data)
        logging.debug(f"Características escaladas: {features_scaled}")
        
        # Generar la predicción utilizando el modelo LSTM
        prediction = model.predict(features_scaled)
        logging.debug(f"Predicción cruda del modelo: {prediction}")
        
        # Postprocesar la predicción para obtener la categoría de calidad del aire
        predicted_summary = postprocess_lstm_output(prediction)
        logging.debug(f"Resumen de calidad del aire predicho: {predicted_summary}")
        
        return predicted_summary

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return "Predicción no disponible"
