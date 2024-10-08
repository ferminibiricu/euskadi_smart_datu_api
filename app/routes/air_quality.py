from fastapi import APIRouter, HTTPException
from typing import Optional
from app.utils.air_quality_utils import (
    get_nearest_station,
    get_all_stations_with_distance,
    get_all_stations,
    get_station_by_id,
)
from app.services.air_quality_service import (
    get_air_quality_data,
    predict_air_quality,
    calculate_air_quality_summary,
)
import logging
import time

router = APIRouter()

@router.get("/stations_by_distance")
def get_stations_by_distance(lat: float, lon: float):
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
        "distance": round(distance, 2),
    }

@router.get("/air_quality")
def get_air_quality_by_location(lat: float, lon: float):
    logging.info(f"Received request for air quality data at lat: {lat}, lon: {lon}.")

    start_time = time.time()
    summary_hours = 6
    prediction_hours = 24

    try:
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

        # Obtener datos para el resumen actual y la predicción (usando el máximo de horas necesarias)
        max_hours = max(summary_hours, prediction_hours)
        air_quality_data = get_air_quality_data(nearest_station_code, max_hours)
        if not air_quality_data:
            logging.warning(f"Air quality data not found for the nearest station ID {nearest_station_code}.")
            current_air_quality_summary = "Datos no disponibles"
            predicted_air_quality_summary = "Predicción no disponible"
            predicted_air_quality_probability = "N/A"
        else:
            # Calcular el resumen actual
            current_air_quality_summary = calculate_air_quality_summary(air_quality_data[:summary_hours], summary_hours)
            logging.debug(f"Current air quality summary: {current_air_quality_summary}")

            # Calcular la predicción
            try:
                logging.debug(f"Data for prediction (station {nearest_station_code}): {air_quality_data[:prediction_hours]}")
                predicted_air_quality_summary, predicted_probability = predict_air_quality(air_quality_data[:prediction_hours])
                logging.debug(f"Predicted air quality summary: {predicted_air_quality_summary}")
                if predicted_probability is not None:
                    predicted_air_quality_probability = f"{predicted_probability * 100:.2f}%"
                else:
                    predicted_air_quality_probability = "N/A"
            except Exception as e:
                logging.error(f"Error in prediction: {str(e)}", exc_info=True)
                predicted_air_quality_summary = "Predicción no disponible"
                predicted_air_quality_probability = "N/A"

        end_time = time.time()
        logging.info(f"Time taken for /air_quality: {end_time - start_time:.2f} seconds")
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
            "predicted_air_quality_summary": predicted_air_quality_summary,
            "predicted_air_quality_probability": predicted_air_quality_probability,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"An error occurred while processing air quality data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing air quality data.")

@router.get("/air_quality_all")
def get_air_quality_for_all_stations(id: Optional[int] = None):
    logging.info(f"Received request for air quality data for all stations or for station ID: {id}.")

    start_time = time.time()
    summary_hours = 6
    prediction_hours = 24

    if id is not None:
        # Procesar solo la estación con el ID proporcionado
        station = get_station_by_id(id)
        if not station:
            logging.warning(f"No station found with ID {id}.")
            raise HTTPException(status_code=404, detail=f"No station found with ID {id}.")

        station_code = station["properties"]["id"]
        station_name = station["properties"]["name"]
        station_lat = station["geometry"]["coordinates"][1]
        station_lon = station["geometry"]["coordinates"][0]

        logging.info(f"Processing station: {station_name} (ID: {station_code})")

        try:
            # Obtener datos para el resumen actual y la predicción (usando el máximo de horas necesarias)
            max_hours = max(summary_hours, prediction_hours)
            air_quality_data = get_air_quality_data(station_code, max_hours)
            if not air_quality_data:
                logging.warning(f"Air quality data not found for station ID {station_code}.")
                current_air_quality_summary = "Datos no disponibles"
                predicted_air_quality_summary = "Predicción no disponible"
                predicted_air_quality_probability = "N/A"
            else:
                # Calcular el resumen actual
                current_air_quality_summary = calculate_air_quality_summary(air_quality_data[:summary_hours], summary_hours)
                logging.debug(f"Current air quality summary for station {station_code}: {current_air_quality_summary}")

                # Calcular la predicción
                try:
                    logging.debug(f"Data for prediction (station {station_code}): {air_quality_data[:prediction_hours]}")
                    predicted_air_quality_summary, predicted_probability = predict_air_quality(air_quality_data[:prediction_hours])
                    logging.debug(f"Predicted air quality summary for station {station_code}: {predicted_air_quality_summary}")
                    if predicted_probability is not None:
                        predicted_air_quality_probability = f"{predicted_probability * 100:.2f}%"
                    else:
                        predicted_air_quality_probability = "N/A"
                except Exception as e:
                    logging.error(f"Error in prediction for station {station_code}: {str(e)}", exc_info=True)
                    predicted_air_quality_summary = "Predicción no disponible"
                    predicted_air_quality_probability = "N/A"

            end_time = time.time()
            logging.info(f"Time taken for /air_quality_all with station ID {id}: {end_time - start_time:.2f} seconds")
            return {
                "station_code": station_code,
                "station_name": station_name,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "current_air_quality_summary": current_air_quality_summary,
                "predicted_air_quality_summary": predicted_air_quality_summary,
                "predicted_air_quality_probability": predicted_air_quality_probability,
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            logging.error(f"An error occurred while processing station {station_code}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing station {station_code}.")
    else:
        # Procesar todas las estaciones
        stations = get_all_stations()
        if not stations:
            logging.warning("No stations found or unable to fetch station data.")
            raise HTTPException(status_code=404, detail="Unable to fetch station data.")

        results = []

        for station in stations["features"]:
            station_code = station["properties"]["id"]
            station_name = station["properties"]["name"]
            station_lat = station["geometry"]["coordinates"][1]
            station_lon = station["geometry"]["coordinates"][0]

            logging.info(f"Processing station: {station_name} (ID: {station_code})")

            try:
                # Obtener datos para el resumen actual y la predicción (usando el máximo de horas necesarias)
                max_hours = max(summary_hours, prediction_hours)
                air_quality_data = get_air_quality_data(station_code, max_hours)
                if not air_quality_data:
                    logging.warning(f"Air quality data not found for station ID {station_code}.")
                    current_air_quality_summary = "Datos no disponibles"
                    predicted_air_quality_summary = "Predicción no disponible"
                    predicted_air_quality_probability = "N/A"
                else:
                    # Calcular el resumen actual
                    current_air_quality_summary = calculate_air_quality_summary(air_quality_data[:summary_hours], summary_hours)
                    logging.debug(f"Current air quality summary for station {station_code}: {current_air_quality_summary}")

                    # Calcular la predicción
                    try:
                        logging.debug(f"Data for prediction (station {station_code}): {air_quality_data[:prediction_hours]}")
                        predicted_air_quality_summary, predicted_probability = predict_air_quality(air_quality_data[:prediction_hours])
                        logging.debug(f"Predicted air quality summary for station {station_code}: {predicted_air_quality_summary}")
                        if predicted_probability is not None:
                            predicted_air_quality_probability = f"{predicted_probability * 100:.2f}%"
                        else:
                            predicted_air_quality_probability = "N/A"
                    except Exception as e:
                        logging.error(f"Error in prediction for station {station_code}: {str(e)}", exc_info=True)
                        predicted_air_quality_summary = "Predicción no disponible"
                        predicted_air_quality_probability = "N/A"

                station_result = {
                    "station_code": station_code,
                    "station_name": station_name,
                    "station_lat": station_lat,
                    "station_lon": station_lon,
                    "current_air_quality_summary": current_air_quality_summary,
                    "predicted_air_quality_summary": predicted_air_quality_summary,
                    "predicted_air_quality_probability": predicted_air_quality_probability,
                }
                results.append(station_result)
            except HTTPException as he:
                raise he
            except Exception as e:
                logging.error(f"An error occurred while processing station {station_code}: {str(e)}", exc_info=True)
                # Continuar con la siguiente estación
                continue

        end_time = time.time()
        logging.info(f"Time taken for /air_quality_all: {end_time - start_time:.2f} seconds")
        return results