from fastapi import APIRouter, HTTPException
from typing import Optional
from app.utils.air_quality_utils import get_nearest_station
from app.services.air_quality_service import get_air_quality_data, predict_air_quality, calculate_air_quality_summary
import logging

router = APIRouter()

@router.get("/air_quality")
def get_air_quality_by_location(
    lat: float, 
    lon: float, 
    hours: Optional[int] = 6
):
    logging.info(f"Received request for air quality data at lat: {lat}, lon: {lon}, for the past {hours} hours.")
    
    # Paso 1: Buscar la estación más cercana
    station = get_nearest_station(lat, lon)
    if not station:
        logging.warning("No nearby station found.")
        raise HTTPException(status_code=404, detail="No nearby station found.")
    
    nearest_station_code = station["properties"]["id"]
    nearest_station_name = station["properties"]["name"]
    logging.info(f"Nearest station found: {nearest_station_name} (ID: {nearest_station_code})")
    
    # Paso 2: Obtener los datos actuales de calidad del aire para las últimas X horas
    data = get_air_quality_data(nearest_station_code, hours)
    if not data:
        logging.warning(f"Air quality data not found for the nearest station ID {nearest_station_code}.")
        raise HTTPException(status_code=404, detail="Air quality data not found for the nearest station.")
    
    # Paso 3: Calcular el resumen actual de la calidad del aire con ponderación y verificación de umbral
    current_air_quality_summary = calculate_air_quality_summary(data, hours)

    # Paso 4: Generar la predicción para las próximas X horas
    predicted_air_quality_summary = predict_air_quality(current_air_quality_summary, hours)
    
    # Paso 5: Devolver la información, incluyendo latitud y longitud
    logging.info(f"Returning air quality data and predictions for station {nearest_station_name} (ID: {nearest_station_code})")
    return {
        "lat": lat,
        "lon": lon,
        "nearest_station_code": nearest_station_code,
        "nearest_station_name": nearest_station_name,
        "current_air_quality_summary": current_air_quality_summary,
        "predicted_air_quality_summary": predicted_air_quality_summary
    }
