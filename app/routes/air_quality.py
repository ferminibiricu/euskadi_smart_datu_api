from fastapi import APIRouter, HTTPException
from typing import Optional
from app.utils.air_quality_utils import get_nearest_station, get_all_stations_with_distance
from app.services.air_quality_service import get_air_quality_data, predict_air_quality, calculate_air_quality_summary
import logging

router = APIRouter()

@router.get("/stations_by_distance")
def get_stations_by_distance(
    lat: float, 
    lon: float
):
    logging.info(f"Received request to get stations ordered by distance from lat: {lat}, lon: {lon}.")
    stations = get_all_stations_with_distance(lat, lon)
    
    if not stations:
        logging.warning("No stations found or unable to fetch station data.")
        raise HTTPException(status_code=404, detail="Unable to fetch station data.")
    
    return stations

@router.get("/nearest_station")
def get_nearest_station_by_location(lat: float, lon: float):
    logging.info(f"Received request for nearest air quality station at lat: {lat}, lon: {lon}.")
    station = get_nearest_station(lat, lon)
    if not station:
        logging.warning("No nearby station found.")
        raise HTTPException(status_code=404, detail="No nearby station found.")
    
    nearest_station_code = station["properties"]["id"]
    nearest_station_name = station["properties"]["name"]
    nearest_station_lat = station["geometry"]["coordinates"][1]
    nearest_station_lon = station["geometry"]["coordinates"][0]
    distance = station["distance"]
    
    logging.info(f"Nearest station found: {nearest_station_name} (ID: {nearest_station_code}) at a distance of {distance:.2f} km.")
    return {
        "nearest_station_code": nearest_station_code,
        "nearest_station_name": nearest_station_name,
        "nearest_station_lat": nearest_station_lat,
        "nearest_station_lon": nearest_station_lon,
        "distance": round(distance, 2)
    }

@router.get("/air_quality")
def get_air_quality_by_location(
    lat: float, 
    lon: float, 
):
    logging.info(f"Received request for air quality data at lat: {lat}, lon: {lon}.")

    # Establecer 'summary_hours' para calcular el valor actual ponderado con los datos disponibles de la 6 últimas horas
    summary_hours = 6

    # Establecer 'prediction_hours' de forma que concuerde con los pasos en el modelo
    prediction_hours = 24

    # Obtener la estación más cercana
    station = get_nearest_station(lat, lon)
    if not station:
        logging.warning("No nearby station found.")
        raise HTTPException(status_code=404, detail="No nearby station found.")

    nearest_station_code = station["properties"]["id"]
    nearest_station_name = station["properties"]["name"]
    nearest_station_lat = station["geometry"]["coordinates"][1]
    nearest_station_lon = station["geometry"]["coordinates"][0]
    distance = station["distance"]
    logging.info(f"Nearest station found: {nearest_station_name} (ID: {nearest_station_code})")

    # Obtener datos para el resumen actual
    data_for_summary = get_air_quality_data(nearest_station_code, summary_hours)
    if not data_for_summary:
        logging.warning(f"Air quality data not found for the nearest station ID {nearest_station_code}.")
        raise HTTPException(status_code=404, detail="Air quality data not found for the nearest station.")

    current_air_quality_summary = calculate_air_quality_summary(data_for_summary, summary_hours)

    # Obtener datos para la predicción
    data_for_prediction = get_air_quality_data(nearest_station_code, prediction_hours)
    if not data_for_prediction or len(data_for_prediction) < prediction_hours:
        logging.warning(f"Not enough data for prediction for station ID {nearest_station_code}.")
        predicted_air_quality_summary = "Predicción no disponible"
    else:
        # Predecir la calidad del aire usando los datos actuales de la estación
        try:
            predicted_air_quality_summary = predict_air_quality(data_for_prediction)
        except HTTPException as e:
            logging.error(f"Error in prediction: {str(e)}")
            predicted_air_quality_summary = "Predicción no disponible"

    logging.info(f"Returning air quality data and predictions for station {nearest_station_name} (ID: {nearest_station_code})")
    return {
        "lat": lat,
        "lon": lon,
        "nearest_station_code": nearest_station_code,
        "nearest_station_name": nearest_station_name,
        "nearest_station_lat": nearest_station_lat,
        "nearest_station_lon": nearest_station_lon,
        "distance": round(distance, 2),
        "current_air_quality_summary": current_air_quality_summary,
        "predicted_air_quality_summary": predicted_air_quality_summary
    }