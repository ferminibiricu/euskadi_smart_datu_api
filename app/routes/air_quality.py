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
    hours: Optional[int] = None
):
    logging.info(f"Received request for air quality data at lat: {lat}, lon: {lon}.")
    
    # Asegurar que 'hours' sea al menos 24
    if hours is None or hours < 24:
        hours = 24
        logging.info(f"Adjusting 'hours' to {hours} to meet the model's requirements.")

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
    
    data = get_air_quality_data(nearest_station_code, hours)
    if not data:
        logging.warning(f"Air quality data not found for the nearest station ID {nearest_station_code}.")
        raise HTTPException(status_code=404, detail="Air quality data not found for the nearest station.")
    
    current_air_quality_summary = calculate_air_quality_summary(data, hours)

    # Predecir la calidad del aire usando los datos actuales de la estación
    try:
        predicted_air_quality_summary = predict_air_quality(data)
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
