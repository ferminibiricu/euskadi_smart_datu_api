import logging
import requests
from fastapi import HTTPException
from config.config import air_quality_base_url, hourly_measurements_endpoint, exclusion_threshold

def get_air_quality_data(station_code: str, hours: int = 2):
    # Calculate the time range for the data
    start_time, end_time = calculate_time_range(hours)
    url = f"{air_quality_base_url}{hourly_measurements_endpoint.format(station_code=station_code, start_time=start_time, end_time=end_time)}"
    
    logging.info(f"Fetching air quality data from {url}")
    
    try:
        # Make the request to the API
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Data successfully retrieved for station ID {station_code}")
        
        # Parse the JSON response
        data = response.json()
        logging.debug(f"Full data structure: {data}")

        # Check if the data is a non-empty list
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

def calculate_time_range(hours: int = 2):
    from datetime import datetime, timedelta

    # Calculate the start and end times
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    # Format the times for the API request
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")

    logging.debug(f"Calculated time range: start={start_time_str}, end={end_time_str}")
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
    
    # Asignar pesos decrecientes: las horas más recientes tendrán más peso
    weights = [(i + 1) / total_data_points for i in range(total_data_points)]

    logging.info(f"Total data points received: {total_data_points}")
    
    for index, hour_data in enumerate(data):
        if 'station' in hour_data and isinstance(hour_data['station'], list) and len(hour_data['station']) > 0:
            air_quality = hour_data['station'][0].get('airQualityStation')
            logging.info(f"Air quality at {hour_data['date']}: {air_quality}")
            
            if air_quality in air_quality_levels:
                air_quality_values.append(air_quality)
                weighted_sum += air_quality_levels[air_quality] * weights[index]
                weight_sum += weights[index]
            else:
                logging.warning(f"No valid air quality data at {hour_data['date']} (None or invalid)")
    
    logging.info(f"Valid air quality data points: {len(air_quality_values)}")
    
    # Verificar la política de exclusión: si faltan más del X% de los datos, no devolvemos resumen
    missing_data_count = total_data_points - len(air_quality_values)
    missing_data_percentage = (missing_data_count / total_data_points) * 100
    logging.info(f"Missing data percentage: {missing_data_percentage}% (Exclusion threshold: {exclusion_threshold}%)")
    
    if missing_data_percentage > exclusion_threshold:
        logging.warning(f"More than {exclusion_threshold}% of the data is missing, excluding result.")
        return "Datos no disponibles"

    # Calcular la media ponderada
    if weight_sum > 0:
        average_score = weighted_sum / weight_sum
        logging.info(f"Weighted average air quality score: {average_score}")

        # Determinar el resumen según el puntaje promedio
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
        logging.warning("No valid data available after filtering. Returning 'Datos no disponibles'.")
        return "Datos no disponibles"

def predict_air_quality(current_summary: str, hours: int = 2):
    logging.info(f"Predicting air quality for the next {hours} hours based on current summary: {current_summary}")
    
    # For simplicity, the prediction is just the current summary repeated
    predicted_summary = current_summary

    logging.info(f"Prediction completed with result: {predicted_summary}")
    return predicted_summary
