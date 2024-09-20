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
    """
    Devuelve una lista de estaciones de calidad del aire ordenadas por distancia desde las coordenadas proporcionadas,
    incluyendo el código de estación, nombre, latitud, longitud y distancia.

    :param lat: Latitud de la ubicación.
    :param lon: Longitud de la ubicación.
    :return: Lista de estaciones ordenadas por distancia.
    """
    logging.info(f"Received request to get stations ordered by distance from lat: {lat}, lon: {lon}.")
    
    # Obtener todas las estaciones con sus distancias
    stations = get_all_stations_with_distance(lat, lon)
    
    if not stations:
        logging.warning("No stations found or unable to fetch station data.")
        raise HTTPException(status_code=404, detail="Unable to fetch station data.")
    
    # Devolver el listado de estaciones con sus distancias
    return stations


@router.get("/nearest_station")
def get_nearest_station_by_location(lat: float, lon: float):
    logging.info(f"Received request for nearest air quality station at lat: {lat}, lon: {lon}.")
    
    # Buscar la estación más cercana
    station = get_nearest_station(lat, lon)
    if not station:
        logging.warning("No nearby station found.")
        raise HTTPException(status_code=404, detail="No nearby station found.")
    
    nearest_station_code = station["properties"]["id"]
    nearest_station_name = station["properties"]["name"]
    nearest_station_lat = station["geometry"]["coordinates"][1]  # Latitud de la estación
    nearest_station_lon = station["geometry"]["coordinates"][0]  # Longitud de la estación
    distance = station["distance"]  # Acceder directamente a la clave 'distance'
    
    logging.info(f"Nearest station found: {nearest_station_name} (ID: {nearest_station_code}) at a distance of {distance:.2f} km.")

    # Devolver los datos solicitados
    return {
        "nearest_station_code": nearest_station_code,
        "nearest_station_name": nearest_station_name,
        "nearest_station_lat": nearest_station_lat,  # Añadir latitud
        "nearest_station_lon": nearest_station_lon,  # Añadir longitud
        "distance": round(distance, 2)
    }


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
    nearest_station_lat = station["geometry"]["coordinates"][1]  # Latitud de la estación más cercana
    nearest_station_lon = station["geometry"]["coordinates"][0]  # Longitud de la estación más cercana
    distance = station["distance"]
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
        "nearest_station_lat": nearest_station_lat,
        "nearest_station_lon": nearest_station_lon,
        "distance": round(distance, 2),
        "current_air_quality_summary": current_air_quality_summary,
        "predicted_air_quality_summary": predicted_air_quality_summary
    }


@router.get("/air_quality_stations")
def get_air_quality_for_all_stations(
    station_id: Optional[str] = None,
    hours: Optional[int] = 6
):
    """
    Devuelve la calidad del aire para todas las estaciones o para una estación específica si se proporciona station_id.
    
    :param station_id: ID opcional de la estación de calidad del aire.
    :param hours: Número opcional de horas anteriores para las que se requiere la calidad del aire (por defecto 6).
    :return: Información de calidad del aire para todas las estaciones o la estación específica.
    """
    logging.info(f"Received request for air quality data for all stations or specific station ID: {station_id}, for the past {hours} hours.")
    
    # Si se proporciona un station_id, obtener datos solo para esa estación
    if station_id:
        logging.info(f"Fetching air quality data for station ID: {station_id}")
        data = get_air_quality_data(station_id, hours)
        if not data:
            logging.warning(f"No air quality data found for station ID {station_id}.")
            raise HTTPException(status_code=404, detail=f"Air quality data not found for station ID {station_id}.")
        
        current_air_quality_summary = calculate_air_quality_summary(data, hours)
        predicted_air_quality_summary = predict_air_quality(current_air_quality_summary, hours)
        
        return {
            "station_id": station_id,
            "current_air_quality_summary": current_air_quality_summary,
            "predicted_air_quality_summary": predicted_air_quality_summary
        }
    
    # Si no se proporciona un station_id, obtener datos para todas las estaciones
    logging.info("Fetching air quality data for all stations.")
    all_stations = get_all_stations()
    results = []
    
    for station in all_stations:
        station_code = station["id"]
        station_name = station["name"]
        logging.info(f"Processing station: {station_name} (ID: {station_code})")
        
        data = get_air_quality_data(station_code, hours)
        if data:
            current_air_quality_summary = calculate_air_quality_summary(data, hours)
            predicted_air_quality_summary = predict_air_quality(current_air_quality_summary, hours)
            
            results.append({
                "station_id": station_code,
                "station_name": station_name,
                "current_air_quality_summary": current_air_quality_summary,
                "predicted_air_quality_summary": predicted_air_quality_summary
            })
        else:
            logging.warning(f"No air quality data found for station ID {station_code}.")
    
    if not results:
        logging.warning("No air quality data found for any stations.")
        raise HTTPException(status_code=404, detail="No air quality data found for any stations.")
    
    return results
